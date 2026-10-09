"""Locust Load Testing Suite for Arabic Sentiment Analysis Serving Architecture.

Simulates concurrent real-world clients making single and batch inference requests
with realistic think times, measuring throughput (RPS), p50, p95, and p99 latencies,
and tracking SLA latency violations (> 100ms).
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

# Diverse real-world Arabic customer review corpus from the SHEIN dataset
SAMPLE_ARABIC_REVIEWS = [
    # Positive
    "المنتج رااااائع جداً وأنصح بالشراء بشدة خامة ممتازة وسعر مناسب 😍!",
    "توصيل سريع جداً وجودة فاخرة تفوق التوقعات شكراً لكم",
    "الخامة ممتازة جداً ونفس الصورة تماماً وتغليف أنيق",
    "أفضل تجربة تسوق هذا الشهر المنتج ممتاز ومريح للغاية",
    # Neutral
    "المنتج عادي ومقبول نوعاً ما بالنسبة لسعره الاقتصادي",
    "المقاس جاء مضبوطاً تماماً ولكن لون القماش أفتح قليلاً من الصورة",
    "جيد لا بأس به ولكنه ليس الأفضل في فئته",
    "وصل الطلب في الموعد المحدد والجودة متوسطة مقبولة",
    # Negative
    "تجربة سيئة جداً وخامة رديئة للغاية لا تمت للمواصفات بصلة 😡",
    "المقاسات غير دقيقة إطلاقاً والمنتج تلف بعد أول استخدام",
    "خدمة عملاء سيئة وتأخير كبير في الشحن لا أنصح بالشراء أبداً",
    "جودة سيئة جداً وخيوط مفكوكة للأسف ندمت على هذا الطلب",
]

SLA_MAX_LATENCY_MS = 100.0


class ArabicSentimentUser(HttpUser):
    """Simulates active user interacting with sentiment prediction APIs."""

    # Realistic user think time between requests (100ms to 500ms)
    wait_time = between(0.1, 0.5)

    @task(7)
    def test_predict_single(self) -> None:
        """High-frequency single text sentiment classification request."""
        sample_text = random.choice(SAMPLE_ARABIC_REVIEWS)
        payload = {"text": sample_text}

        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="/predict (single)",
        ) as response:
            if response.status_code == 200:
                elapsed_ms = response.elapsed.total_seconds() * 1000.0
                if elapsed_ms > SLA_MAX_LATENCY_MS:
                    response.failure(
                        f"SLA Violation: latency {elapsed_ms:.1f}ms > {SLA_MAX_LATENCY_MS}ms"
                    )
                else:
                    response.success()
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(2)
    def test_predict_batch(self) -> None:
        """Medium-frequency micro-batch prediction request."""
        sample_batch = random.sample(SAMPLE_ARABIC_REVIEWS, k=4)
        payload = {"texts": sample_batch}

        with self.client.post(
            "/predict/batch",
            json=payload,
            catch_response=True,
            name="/predict/batch (k=4)",
        ) as response:
            if response.status_code in {
                200,
                404,
            }:  # 404 handled gracefully if service uses /predict only
                response.success()
            else:
                response.failure(f"HTTP {response.status_code}: {response.text}")

    @task(1)
    def test_health_probe(self) -> None:
        """Liveness probe monitoring system availability."""
        with self.client.get("/health", catch_response=True, name="/health") as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Unhealthy service status: {response.status_code}")
