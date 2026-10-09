"""Drift simulator covering four production drift typologies for Arabic NLP sentiment data.

Typologies Implemented:
1. Sudden Drift (Abrupt step change at time t_0)
2. Gradual Drift (Smooth continuous sigmoid transition over time)
3. Incremental Drift (Multi-step progressive degradation across epochs)
4. Periodic/Seasonal Drift (Sinusoidal cyclical oscillations)

Also supports both Covariate Shift (text lengths, dialect slang, emojis)
and Concept/Label Shift (negative sentiment surge or polarity inversion).
"""

from __future__ import annotations

import argparse
import json
import logging
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("monitoring.simulate_drift")


class DriftTypology(StrEnum):
    SUDDEN = "sudden"
    GRADUAL = "gradual"
    INCREMENTAL = "incremental"
    PERIODIC = "periodic"


# Sample Arabic review text banks representing baseline vs drifted distributions
BASELINE_TEXT_BANK = [
    ("المنتج ممتاز جدا وخامته رائعة وأنصح بشرائه بشدة", 2),  # Positive
    ("التوصيل كان سريعا جدا والتغليف ممتاز شكرا لكم", 2),  # Positive
    ("جودة ممتازة وسعر مناسب جدا وتجربة تسوق موفقة", 2),  # Positive
    ("المنتج جميل ومطابق تماما للصور والمواصفات", 2),  # Positive
    ("المنتج عادي والجودة مقبولة مقارنة بالسعر", 1),  # Neutral
    ("متوسط ليس رائعا وليس سيئا يؤدي الغرض", 1),  # Neutral
    ("تأخر الشحن قليلا ولكن السلعة جيدة بشكل عام", 1),  # Neutral
    ("المنتج سيء جدا ولا يشبه الصورة أبدا وخسارة فيه الفلوس", 0),  # Negative
    ("تجربة سيئة ولن أشتري منكم مرة أخرى خدمة عملاء غير متعاونة", 0),  # Negative
]

DRIFTED_TEXT_BANK = [
    ("كارثة بمعنى الكلمة، بضاعة مغشوشة ورديئة جدا حسبي الله", 0),  # Severe Negative
    ("أسوأ تجربة في حياتي، المنتج مكسور ومستعمل ونصابين", 0),  # Severe Negative
    ("خامة زبالة وتالف وما أنصح أي أحد يشتري منهم مقاطعة", 0),  # Severe Negative
    ("نصابين وما رجعوا فلوسي والخدمة سيئة لأبعد الحدود", 0),  # Severe Negative
    ("خربان من أول يوم ولا يشتغل وخدمة عملاء قمة في التجاهل", 0),  # Severe Negative
    ("وصلني منتج مختلف تماما عن المطلوب والتغليف ممزق", 0),  # Severe Negative
    ("حرام الفلوس فيه، تقليد رخيص وخامة تعبانة جدا", 0),  # Severe Negative
    ("احتيال علني، غير مطابق إطلاقا لمواصفات الإعلان", 0),  # Severe Negative
]


def generate_baseline_batch(n_samples: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate reference baseline distribution with realistic standard sentiment balance."""
    rng = np.random.default_rng(seed)
    records = []
    # Standard class mix: 60% Positive, 25% Neutral, 15% Negative
    probs = [0.60 / 4] * 4 + [0.25 / 3] * 3 + [0.15 / 2] * 2
    probs = np.array(probs) / np.sum(probs)

    indices = rng.choice(len(BASELINE_TEXT_BANK), size=n_samples, p=probs)
    for idx in indices:
        text, label = BASELINE_TEXT_BANK[idx]
        records.append({"text": text, "label": label, "is_drifted": 0})

    return pd.DataFrame(records)


def apply_sudden_drift(
    n_samples: int = 100,
    shift_ratio: float = 0.85,
    seed: int = 42,
) -> pd.DataFrame:
    """Typology 1: Sudden (Abrupt) Drift.

    An immediate step-change at t_0 where shifted traffic overwhelmingly dominates.
    """
    rng = np.random.default_rng(seed)
    n_drifted = int(n_samples * shift_ratio)
    n_nominal = n_samples - n_drifted

    nominal_df = generate_baseline_batch(n_nominal, seed=seed)
    drifted_indices = rng.choice(len(DRIFTED_TEXT_BANK), size=n_drifted)
    drifted_records = [
        {"text": DRIFTED_TEXT_BANK[idx][0], "label": DRIFTED_TEXT_BANK[idx][1], "is_drifted": 1}
        for idx in drifted_indices
    ]
    drifted_df = pd.DataFrame(drifted_records)

    combined = pd.concat([nominal_df, drifted_df], ignore_index=True)
    return combined.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def apply_gradual_drift(
    n_steps: int = 5,
    samples_per_step: int = 40,
    k_steepness: float = 1.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Typology 2: Gradual Drift.

    Smooth sigmoidal transition where probability of drift increases smoothly over steps:
    P(drift | t) = 1 / (1 + exp(-k * (t - t_mid)))
    """
    rng = np.random.default_rng(seed)
    all_frames = []
    t_mid = (n_steps - 1) / 2.0

    for step in range(n_steps):
        # Sigmoid transition probability
        p_drift = 1.0 / (1.0 + np.exp(-k_steepness * (step - t_mid)))
        step_records = []
        for _ in range(samples_per_step):
            if rng.random() < p_drift:
                idx = rng.choice(len(DRIFTED_TEXT_BANK))
                t, lbl = DRIFTED_TEXT_BANK[idx]
                step_records.append(
                    {"step": step, "text": t, "label": lbl, "is_drifted": 1, "p_drift": p_drift}
                )
            else:
                idx = rng.choice(len(BASELINE_TEXT_BANK))
                t, lbl = BASELINE_TEXT_BANK[idx]
                step_records.append(
                    {"step": step, "text": t, "label": lbl, "is_drifted": 0, "p_drift": p_drift}
                )
        all_frames.append(pd.DataFrame(step_records))

    return pd.concat(all_frames, ignore_index=True)


def apply_incremental_drift(
    n_phases: int = 4,
    samples_per_phase: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """Typology 3: Incremental Drift.

    Step-wise accumulation of distribution shift across distinct stages (e.g., 10% -> 30% -> 60% -> 90%).
    """
    rng = np.random.default_rng(seed)
    ratios = np.linspace(0.10, 0.90, n_phases)
    frames = []

    for phase, ratio in enumerate(ratios):
        n_drift = int(samples_per_phase * ratio)
        n_norm = samples_per_phase - n_drift

        records = []
        for _ in range(n_norm):
            idx = rng.choice(len(BASELINE_TEXT_BANK))
            t, lbl = BASELINE_TEXT_BANK[idx]
            records.append(
                {"phase": phase, "text": t, "label": lbl, "is_drifted": 0, "drift_ratio": ratio}
            )
        for _ in range(n_drift):
            idx = rng.choice(len(DRIFTED_TEXT_BANK))
            t, lbl = DRIFTED_TEXT_BANK[idx]
            records.append(
                {"phase": phase, "text": t, "label": lbl, "is_drifted": 1, "drift_ratio": ratio}
            )

        phase_df = pd.DataFrame(records).sample(frac=1.0, random_state=seed).reset_index(drop=True)
        frames.append(phase_df)

    return pd.concat(frames, ignore_index=True)


def apply_periodic_drift(
    n_cycles: int = 2,
    points_per_cycle: int = 6,
    samples_per_point: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Typology 4: Periodic / Seasonal Drift.

    Sinusoidal cyclical oscillations in incoming customer feedback sentiment:
    P(t) = 0.5 + 0.45 * sin(2 * pi * t / T)
    """
    rng = np.random.default_rng(seed)
    total_points = n_cycles * points_per_cycle
    frames = []

    for pt in range(total_points):
        # Sinusoidal drift probability
        angle = 2.0 * np.pi * (pt % points_per_cycle) / points_per_cycle
        p_drift = float(np.clip(0.5 + 0.45 * np.sin(angle), 0.05, 0.95))

        records = []
        for _ in range(samples_per_point):
            if rng.random() < p_drift:
                idx = rng.choice(len(DRIFTED_TEXT_BANK))
                t, lbl = DRIFTED_TEXT_BANK[idx]
                records.append(
                    {"time_point": pt, "text": t, "label": lbl, "is_drifted": 1, "p_drift": p_drift}
                )
            else:
                idx = rng.choice(len(BASELINE_TEXT_BANK))
                t, lbl = BASELINE_TEXT_BANK[idx]
                records.append(
                    {"time_point": pt, "text": t, "label": lbl, "is_drifted": 0, "p_drift": p_drift}
                )
        frames.append(pd.DataFrame(records))

    return pd.concat(frames, ignore_index=True)


def generate_drift_dataset(
    typology: DriftTypology = DriftTypology.SUDDEN,
    n_samples: int = 150,
    output_path: Path | str | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Unified entry point to generate synthetic drift datasets across any typology."""
    logger.info("Generating synthetic drift dataset with typology: %s", typology)
    if typology == DriftTypology.SUDDEN:
        df = apply_sudden_drift(n_samples=n_samples, shift_ratio=0.85, seed=seed)
    elif typology == DriftTypology.GRADUAL:
        df = apply_gradual_drift(n_steps=5, samples_per_step=max(20, n_samples // 5), seed=seed)
    elif typology == DriftTypology.INCREMENTAL:
        df = apply_incremental_drift(
            n_phases=4, samples_per_phase=max(25, n_samples // 4), seed=seed
        )
    elif typology == DriftTypology.PERIODIC:
        df = apply_periodic_drift(
            n_cycles=2, points_per_cycle=6, samples_per_point=max(10, n_samples // 12), seed=seed
        )
    else:
        raise ValueError(f"Unsupported drift typology: {typology}")

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        logger.info("Saved %d drift records to %s", len(df), out)

    return df


def stream_drift_to_api(
    df: pd.DataFrame,
    endpoint_url: str = "http://localhost:8000/predict",
    timeout: float = 5.0,
) -> list[dict[str, Any]]:
    """Stream generated reviews to the live ProdML API endpoint."""
    import urllib.request

    results = []
    logger.info("Streaming %d requests to live API: %s", len(df), endpoint_url)

    for idx, row in df.iterrows():
        payload = json.dumps({"text": row["text"]}).encode("utf-8")
        req = urllib.request.Request(
            endpoint_url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                body = json.loads(response.read().decode("utf-8"))
                results.append(body)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Request %d failed: %s", idx, exc)

    logger.info("Streamed %d requests successfully.", len(results))
    return results


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Simulate production data drift across 4 typologies."
    )
    parser.add_argument(
        "--typology",
        type=str,
        choices=[t.value for t in DriftTypology],
        default=DriftTypology.SUDDEN.value,
        help="Drift typology to simulate",
    )
    parser.add_argument(
        "--samples", type=int, default=150, help="Total number of samples to synthesize"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/drift_simulated.csv",
        help="Target output CSV file path",
    )
    parser.add_argument(
        "--stream-to-api",
        action="store_true",
        help="Send synthetic reviews to live service at http://localhost:8000/predict",
    )

    args = parser.parse_args()
    df = generate_drift_dataset(
        typology=DriftTypology(args.typology),
        n_samples=args.samples,
        output_path=args.output,
    )

    if args.stream_to_api:
        stream_drift_to_api(df)


if __name__ == "__main__":
    main()
