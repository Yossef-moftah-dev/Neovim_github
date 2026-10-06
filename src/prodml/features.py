"""Feature extraction and text preprocessing pipelines."""

from __future__ import annotations

import logging
import re
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Precompiled regex patterns for performance
TASHKEEL_TATWEEL_PATTERN = re.compile(r"[\u064B-\u0652\u0670\u0640]")
ALIF_VARIANTS_PATTERN = re.compile(r"[إأآٱ]")
YAA_PATTERN = re.compile(r"ى\b")
HAA_PATTERN = re.compile(r"ة\b")
ELONGATION_PATTERN = re.compile(r"(.)\1{2,}")
EMOJI_PATTERN = re.compile(r"([\U00010000-\U0010ffff])")
PUNCTUATION_PATTERN = re.compile(r"([!؟?])")
MULTIPLE_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text by removing diacritics, unifying alifs, and formatting emojis.

    Matches preprocessing logic from the exploration phase to guarantee parity.
    """
    if not isinstance(text, str):
        return ""

    # Strip diacritics (tashkeel) and tatweel
    normalized = TASHKEEL_TATWEEL_PATTERN.sub("", text)
    # Unify alif variants
    normalized = ALIF_VARIANTS_PATTERN.sub("ا", normalized)
    # Normalize final alif maqsura to yaa
    normalized = YAA_PATTERN.sub("ي", normalized)
    # Normalize final taa marbuta to haa
    normalized = HAA_PATTERN.sub("ه", normalized)
    # Collapse character elongation (e.g., رررررائع -> ررائع)
    normalized = ELONGATION_PATTERN.sub(r"\1\1", normalized)
    # Isolate emojis and Arabic punctuation with spaces
    normalized = EMOJI_PATTERN.sub(r" \1 ", normalized)
    normalized = PUNCTUATION_PATTERN.sub(r" \1 ", normalized)
    # Collapse multiple whitespaces and strip boundaries
    return MULTIPLE_WHITESPACE_PATTERN.sub(" ", normalized).strip()


def prepare_features(
    texts: list[str],
    tokenizer: Any,
    max_length: int = 128,
) -> dict[str, torch.Tensor]:
    """Preprocess and tokenize a batch of raw text strings.

    Args:
        texts: Raw input strings.
        tokenizer: HuggingFace AutoTokenizer instance.
        max_length: Maximum sequence token length.

    Returns:
        Tokenized tensors ready for model inference.
    """
    cleaned_texts = [normalize_arabic(t) for t in texts]
    encoded: dict[str, torch.Tensor] = tokenizer(
        cleaned_texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    return encoded
