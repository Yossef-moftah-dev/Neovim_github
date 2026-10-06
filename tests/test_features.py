"""Tests for feature preprocessing and text normalization."""

from __future__ import annotations

from transformers import AutoTokenizer

from prodml.features import normalize_arabic, prepare_features


def test_normalize_arabic_tashkeel() -> None:
    text = "شُكْراً جَزِيلاً لَكُمْ"
    expected = "شكرا جزيلا لكم"
    assert normalize_arabic(text) == expected


def test_normalize_arabic_alif_variants() -> None:
    text = "إبراهيم أحمد آمنة ٱستماع"
    expected = "ابراهيم احمد امنه استماع"
    assert normalize_arabic(text) == expected


def test_normalize_arabic_elongation() -> None:
    text = "رررررائع جددددا"
    expected = "ررائع جددا"
    assert normalize_arabic(text) == expected


def test_normalize_arabic_emojis_and_punctuation() -> None:
    text = "ممتاز😍!كيف الحال؟"
    normalized = normalize_arabic(text)
    assert "😍" in normalized
    assert "!" in normalized
    assert "؟" in normalized
    # Checks that punctuation and emoji are separated by spaces
    assert "ممتاز 😍 ! كيف الحال ؟" == normalized


def test_normalize_arabic_empty_and_non_string() -> None:
    assert normalize_arabic("") == ""
    assert normalize_arabic("   ") == ""
    assert normalize_arabic(None) == ""  # type: ignore[arg-type]


def test_prepare_features(model_tokenizer: AutoTokenizer) -> None:
    batch = ["المنتج رائع", "خدمة سيئة"]
    enc = prepare_features(batch, model_tokenizer, max_length=32)
    assert "input_ids" in enc
    assert "attention_mask" in enc
    assert enc["input_ids"].shape[0] == 2
