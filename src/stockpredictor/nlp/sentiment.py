"""Headline sentiment with FinBERT (free, runs locally on CPU).

Score = P(positive) - P(negative), in [-1, 1]. Used by the GitHub Actions
news job; the Mac does not need to install the model.
"""

from __future__ import annotations

MODEL_NAME = "ProsusAI/finbert"


class FinBertScorer:
    def __init__(self, model_name: str = MODEL_NAME, batch_size: int = 32):
        from transformers import pipeline  # heavy import, only when scoring

        self._pipe = pipeline("text-classification", model=model_name, top_k=None,
                              truncation=True)
        self._batch = batch_size

    def __call__(self, texts: list[str]) -> list[float]:
        scores = []
        for out in self._pipe(texts, batch_size=self._batch):
            probs = {d["label"].lower(): d["score"] for d in out}
            scores.append(round(probs.get("positive", 0) - probs.get("negative", 0), 4))
        return scores
