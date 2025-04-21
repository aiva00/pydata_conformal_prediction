"""
Logit-free Conformal Prediction for API-only LLMs
Paper: 'API Is Enough: Conformal Prediction for Large Language Models Without Logit-Access' (Su et al., 2024)
Implements the full algorithm (frequency ▸ NE ▸ SS non-conformity) for chat-completion models.
"""

from __future__ import annotations
import os, re, string, math, random, asyncio, time, json
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import openai
from sentence_transformers import SentenceTransformer, util
from tqdm.auto import tqdm

OPENAI_API_KEY = "insert your API key here"

def normalize_text(s: str) -> str:
    """Lower‑case, strip punctuation / articles / excess spaces (SQuAD style)."""
    def rm_articles(t): return re.sub(r"\b(a|an|the)\b", " ", t)
    def rm_punc(t):     return "".join(ch for ch in t if ch not in set(string.punctuation))
    return " ".join(rm_articles(rm_punc(s.lower())).split())

def entropy(probs: np.ndarray) -> float:
    """Shannon entropy normalised to [0,1]."""
    if probs.sum() == 0: return 1.0
    p = probs / probs.sum()
    h = -(p * np.log(p + 1e-12)).sum()
    return h / math.log(len(p) + 1e-12)

class ChatSampler:
    """
    Thin async wrapper around the new openai.chat.completions.create endpoint.
    Supports temperature / top_p and parallel calls.
    """

    def __init__(self,
                 model: str                   = "gpt-4o-mini",
                 n: int                       = 30,
                 temperature: float           = 0.7,
                 top_p: float                 = 1.0,
                 system_prompt: str | None    = None,
                 max_tokens: int              = 64,
                 api_key: str | None          = None,
                 sleep_throttle: float        = 0.0):
        openai.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if openai.api_key is None:
            raise ValueError("Set OPENAI_API_KEY env‑var or pass api_key param.")

        self.client        = openai.AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.model         = model
        self.n             = n
        self.temperature   = temperature
        self.top_p         = top_p
        self.system_prompt = system_prompt
        self.max_tokens    = max_tokens
        self.sleep         = sleep_throttle

    async def _one(self, user_msg: str) -> str:
        # A single sample from the model
        msgs = []
        if self.system_prompt: msgs.append({"role": "system", "content": self.system_prompt})
        msgs.append({"role": "user", "content": user_msg})

        resp = await self.client.chat.completions.create(
            model       = self.model,
            messages    = msgs,
            max_tokens  = self.max_tokens,
            temperature = self.temperature,
            top_p       = self.top_p,
        )
        if self.sleep: time.sleep(self.sleep)           # polite throttle
        return resp.choices[0].message.content.strip()

    async def sample(self, user_msg: str) -> List[str]:
        coros = [self._one(user_msg) for _ in range(self.n)]
        return await asyncio.gather(*coros)

# ---------------- Conformal Predictor core ----------------
@dataclass
class LofreeCPConfig:
    m: int = 30                 # samples per prompt
    alpha: float = 0.2          # desired error‑rate (1‑coverage)
    lambda1: float = 1.0        # weight for normalised entropy (NE)
    lambda2: float = 1.0        # weight for semantic similarity (SS)
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

class LofreeCP:
    """
    Implements calibration ▸ validation ▸ prediction as defined in the paper.
    """

    def __init__(self, cfg: LofreeCPConfig):
        self.cfg   = cfg
        self.q_hat = None   # conformal threshold
        self._embedder = SentenceTransformer(cfg.embed_model)

    # non-conformity scores
    def _scores_for_prompt(self,
                           responses: List[str],
                           freqs: Dict[str, int],
                           ne: float,
                           top_emb: np.ndarray
                          ) -> Dict[str, float]:
        m = self.cfg.m
        scores = {}
        for r in responses:
            f    = freqs[r] / m                 # coarse frequency
            ss   = util.cos_sim(top_emb, self._embedder.encode(r, normalize_embeddings=True))[0][0].item() \
                   if r != responses[0] else 0.0           # SS with top‑1 (0 for top‑1 itself)
            s = -f + self.cfg.lambda1 * ne - self.cfg.lambda2 * ss
            scores[r] = s
        return scores

    # calibration
    async def _collect(self, sampler: ChatSampler, prompts: List[str]) -> Tuple[List[Dict], List[List[str]]]:
        """
        Returns:
            pools: list of dict{prompt, responses(List[str])}
            flat:  flat list of all raw responses (for embedding batching)
        """
        pools, flat = [], []
        for p in tqdm(prompts, desc="sampling"):
            resp = await sampler.sample(p)
            pools.append({"prompt": p, "samples": resp})
            flat.extend(resp)
        return pools, flat

    def _calibrate_scores(self, pools: List[Dict], answers: List[str]) -> List[float]:
        """
        Compute non-conformity scores for calibration answers (may be ∞ if never sampled).
        """
        embeds_all = {
            normalize_text(r): self._embedder.encode(r, normalize_embeddings=True)
            for pool in pools for r in pool["samples"]
        }

        cal_scores = []
        for pool, true_ans in zip(pools, answers):
            freq = Counter([normalize_text(r) for r in pool["samples"]])
            ne   = entropy(np.array(list(freq.values())))
            # top‑1:
            top_resp = max(freq.items(), key=lambda kv: kv[1])[0]
            top_emb  = embeds_all[top_resp]
            scores   = self._scores_for_prompt(
                           responses=list(freq.keys()),
                           freqs=freq,
                           ne=ne,
                           top_emb=top_emb)
            true_norm = normalize_text(true_ans)
            cal_scores.append(scores.get(true_norm, math.inf))   # ∞ if never generated
        return cal_scores

    async def calibrate(self,
                        sampler: ChatSampler,
                        prompts: List[str],
                        answers: List[str]):
        """
        Calibrate q̂.  Prompts & answers must be *paired* calibration set.
        """
        pools, _ = await self._collect(sampler, prompts)
        scores   = self._calibrate_scores(pools, answers)
        # split‑CP quantile
        n = len(scores)
        # r (1‑indexed) then k=r‑1 (0‑indexed), but clamp so k ≤ n‑1
        r = math.ceil((n + 1) * (1 - self.cfg.alpha))
        k = min(r - 1, n - 1)
        self.q_hat = np.partition(scores, k)[k]
        print(f"[lofree‑cp] calibrated q̂ = {self.q_hat:.4f} with α = {self.cfg.alpha}")

    # ---------- prediction ----------
    async def predict(self, sampler: ChatSampler, prompt: str) -> List[str]:
        """
        Return *prediction set* for one prompt.
        """
        if self.q_hat is None:
            raise RuntimeError("Call calibrate() first.")

        resp = await sampler.sample(prompt)
        freq = Counter([normalize_text(r) for r in resp])
        ne   = entropy(np.array(list(freq.values())))

        embeds = {r: self._embedder.encode(r, normalize_embeddings=True) for r in freq.keys()}
        top_resp = max(freq.items(), key=lambda kv: kv[1])[0]
        top_emb  = embeds[top_resp]

        scores = self._scores_for_prompt(list(freq.keys()), freq, ne, top_emb)
        pred_set = [r for r, s in scores.items() if s <= self.q_hat]
        return pred_set if pred_set else [top_resp]    # guarantee non‑empty

# Quick demo
if __name__ == "__main__":
    
    calib_pairs = [
        ("Which American-born Sinclair won the Nobel Prize for Literature in 1930?", "Sinclair Lewis"),
        ("Where in England was Dame Judi Dench born?", "York"),
        ("In which decade did Billboard magazine first publish an American hit chart?", "30s"),
        ("Who wrote 'Pride and Prejudice'?", "Jane Austen"),
        ("What is the capital of France?",    "Paris"),
    ]
    test_questions = [
        "Which country is Europe's largest silk producer?",
        "What is Bruce Willis' real first name?",
        "What is the number that rhymes with the word we use to describe a tall plant?",
        "What are the top 3 ways to fix Greece's financial situation?"
    ]

    # configure hyper‑parameters (tiny for demo)
    cfg = LofreeCPConfig(m=10, alpha=0.1, lambda1=0.5, lambda2=0.5)
    cp  = LofreeCP(cfg)
    sampler = ChatSampler(
        model="gpt-4o-mini",
        n=cfg.m,
        temperature=0.7,
        system_prompt="Answer the question concisely with a short noun phrase.",
        api_key=OPENAI_API_KEY
    )

    async def _run():
        # calibration
        cal_prompts, cal_answers = zip(*calib_pairs)
        await cp.calibrate(sampler, list(cal_prompts), list(cal_answers))

        # prediction
        for q in test_questions:
            pred_set = await cp.predict(sampler, q)
            print(f"\nQ: {q}\nPrediction set (|S|={len(pred_set)}):")
            for p in pred_set:
                print("  •", p)

    asyncio.run(_run())