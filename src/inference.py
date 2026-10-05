import os
import re
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "outputs" / "final_model"))
MAX_LENGTH = 128


def normalize_arabic(text: str) -> str:
    """Same preprocessing used during training (must stay in sync with the notebook)."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"[\u064B-\u0652\u0670\u0640]", "", text)
    text = re.sub(r"[إأآٱ]", "ا", text)
    text = re.sub(r"ى\b", "ي", text)
    text = re.sub(r"ة\b", "ه", text)
    text = re.sub(r"(.)\1{2,}", r"\1\1", text)
    text = re.sub(r"([\U00010000-\U0010ffff])", r" \1 ", text)
    text = re.sub(r"([!؟?])", r" \1 ", text)
    return re.sub(r"\s+", " ", text).strip()


class SentimentModel:
    def __init__(self, model_dir: Path = MODEL_DIR):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}

    @torch.no_grad()
    def predict(self, texts: list[str]) -> list[dict]:
        enc = self.tokenizer(
            [normalize_arabic(t) for t in texts],
            truncation=True, max_length=MAX_LENGTH, padding=True, return_tensors="pt",
        ).to(self.device)
        probs = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu()
        results = []
        for p in probs:
            idx = int(p.argmax())
            results.append({
                "label": self.id2label[idx],
                "confidence": float(p[idx]),
                "probabilities": {self.id2label[i]: float(v) for i, v in enumerate(p)},
            })
        return results
