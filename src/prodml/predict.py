"""OOP Predictor implementation for Arabic sentiment classification."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel, Field
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from prodml.config import ModelConfig, get_settings
from prodml.features import prepare_features
from prodml.timing import timed

logger = logging.getLogger(__name__)


class PredictionResult(BaseModel):
    """Prediction result output schema."""

    label: str = Field(..., description="Predicted sentiment class (Negative, Neutral, Positive)")
    confidence: float = Field(..., description="Probability of top predicted class")
    probabilities: dict[str, float] = Field(..., description="Full class probability distribution")
    latency_ms: float | None = Field(default=None, description="Inference latency in milliseconds")


class SentimentPredictor:
    """Object-Oriented Predictor encapsulating tokenizer, model, and inference lifecycle."""

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        config: ModelConfig | None = None,
    ) -> None:
        self.config = config or get_settings().model
        self.device = torch.device(self.config.device)
        self.tokenizer = tokenizer
        self.model = model.to(self.device).eval()
        self.id2label: dict[int, str] = {
            int(k): v
            for k, v in (
                getattr(self.model.config, "id2label", None) or self.config.id2label
            ).items()
        }
        logger.info("Initialized SentimentPredictor on device: %s", self.device)

    @classmethod
    def load(
        cls,
        model_dir: Path | str | None = None,
        device: str | None = None,
    ) -> SentimentPredictor:
        """Instantiate and load weights from disk into memory.

        Args:
            model_dir: Directory containing model weights and tokenizer config.
            device: Target execution device ('cpu' or 'cuda').
        """
        settings = get_settings()
        cfg = settings.model.model_copy()
        if model_dir is not None:
            cfg.model_dir = Path(model_dir)
        if device is not None:
            cfg.device = device

        target_dir = cfg.model_dir
        if not target_dir.exists():
            raise FileNotFoundError(f"Model directory does not exist: {target_dir}")

        logger.info("Loading model weights and tokenizer from %s", target_dir)
        tokenizer = AutoTokenizer.from_pretrained(target_dir)
        model = AutoModelForSequenceClassification.from_pretrained(target_dir)

        return cls(model=model, tokenizer=tokenizer, config=cfg)

    def is_loaded(self) -> bool:
        """Check if model and tokenizer are resident in memory."""
        return self.model is not None and self.tokenizer is not None

    @timed
    def predict_one(self, text: str) -> PredictionResult:
        """Generate prediction for a single text input string."""
        results = self.predict_batch([text])
        if not results:
            raise ValueError("Inference returned empty result set")
        return results[0]

    @timed
    @torch.no_grad()
    def predict_batch(self, texts: list[str]) -> list[PredictionResult]:
        """Generate sentiment predictions for a batch of input texts."""
        if not texts:
            return []

        enc = prepare_features(texts, self.tokenizer, max_length=self.config.max_length)
        inputs = {k: v.to(self.device) for k, v in enc.items()}

        outputs = self.model(**inputs)
        logits = outputs.logits.float()
        probs = F.softmax(logits, dim=-1).cpu().numpy()

        results: list[PredictionResult] = []
        for p in probs:
            idx = int(np.argmax(p))
            results.append(
                PredictionResult(
                    label=self.id2label[idx],
                    confidence=float(p[idx]),
                    probabilities={self.id2label[i]: float(val) for i, val in enumerate(p)},
                )
            )

        return results

    @torch.no_grad()
    def predict_logits_native(self, texts: list[str]) -> np.ndarray:
        """Extract raw logits from native PyTorch model for parity comparisons."""
        enc = prepare_features(texts, self.tokenizer, max_length=self.config.max_length)
        inputs = {k: v.to(self.device) for k, v in enc.items()}
        outputs = self.model(**inputs)
        return outputs.logits.detach().cpu().numpy()

    def export_onnx(
        self,
        output_path: Path | str,
        opset_version: int = 17,
    ) -> Path:
        """Export native PyTorch model to ONNX format with dynamic batching."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        dummy_text = ["نموذج تجريبي للتحقق"]
        enc = prepare_features(dummy_text, self.tokenizer, max_length=self.config.max_length)
        inputs = {k: v.to(self.device) for k, v in enc.items()}

        input_names = ["input_ids", "attention_mask"]
        dynamic_axes = {
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size"},
        }

        # Include token_type_ids if required by architecture
        args: tuple[torch.Tensor, ...]
        if "token_type_ids" in inputs:
            input_names.append("token_type_ids")
            dynamic_axes["token_type_ids"] = {0: "batch_size", 1: "sequence_length"}
            args = (inputs["input_ids"], inputs["attention_mask"], inputs["token_type_ids"])
        else:
            args = (inputs["input_ids"], inputs["attention_mask"])

        logger.info("Exporting ONNX model to %s (opset %d)", out, opset_version)
        self.model.eval()
        torch.onnx.export(
            self.model,
            args,
            str(out),
            input_names=input_names,
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset_version,
            do_constant_folding=True,
        )
        return out

    def predict_logits_onnx(
        self,
        texts: list[str],
        onnx_path: Path | str | None = None,
    ) -> np.ndarray:
        """Compute logits using ONNX Runtime for parity verification."""
        import onnxruntime as ort

        target_path = Path(onnx_path or self.config.onnx_path)
        if not target_path.exists():
            raise FileNotFoundError(f"ONNX model file not found at: {target_path}")

        session = ort.InferenceSession(str(target_path), providers=["CPUExecutionProvider"])
        enc = prepare_features(texts, self.tokenizer, max_length=self.config.max_length)

        ort_inputs = {
            "input_ids": enc["input_ids"].cpu().numpy(),
            "attention_mask": enc["attention_mask"].cpu().numpy(),
        }
        if "token_type_ids" in enc:
            ort_inputs["token_type_ids"] = enc["token_type_ids"].cpu().numpy()

        onnx_outputs = session.run(["logits"], ort_inputs)
        return onnx_outputs[0]
