"""Pytest configuration and shared fixtures for ProdML test suite."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from transformers import AutoTokenizer, BertConfig, BertForSequenceClassification

from prodml.api import create_app
from prodml.config import ModelConfig
from prodml.predict import SentimentPredictor


@pytest.fixture(scope="session")
def sample_arabic_texts() -> list[str]:
    """Diverse Arabic sample phrases for testing."""
    return [
        "المنتج رااااائع جداً وأنصح بالشراء 😍!",
        "تجربة سيئة جداً وخامة رديئة للغاية 😡",
        "المنتج عادي لا بأس به بالنسبة للسعر",
    ]


@pytest.fixture(scope="session")
def model_tokenizer() -> AutoTokenizer:
    """Pretrained AraBERT tokenizer from local artifacts."""
    model_dir = Path(__file__).resolve().parent.parent / "outputs" / "final_model"
    return AutoTokenizer.from_pretrained(str(model_dir))


@pytest.fixture(scope="session")
def fast_predictor(model_tokenizer: AutoTokenizer) -> SentimentPredictor:
    """Lightweight Bert model coupled with real tokenizer for fast unit testing."""
    config = BertConfig(
        vocab_size=len(model_tokenizer),
        hidden_size=32,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=64,
        num_labels=3,
        id2label={0: "Negative", 1: "Neutral", 2: "Positive"},
        label2id={"Negative": 0, "Neutral": 1, "Positive": 2},
    )
    torch.manual_seed(42)
    small_model = BertForSequenceClassification(config).eval()

    model_cfg = ModelConfig(
        max_length=64,
        device="cpu",
    )
    return SentimentPredictor(model=small_model, tokenizer=model_tokenizer, config=model_cfg)


@pytest.fixture
def client(fast_predictor: SentimentPredictor) -> Generator[TestClient]:
    """Test client with mock predictor initialized into app state."""
    test_app = create_app()
    test_app.state.predictor = fast_predictor
    test_app.state.artifact_hash = "mock_sha256_hash_1234567890abcdef"
    with TestClient(test_app) as test_client:
        yield test_client
