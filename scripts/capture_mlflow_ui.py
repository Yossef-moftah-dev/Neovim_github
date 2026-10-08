"""Automated Playwright script to capture MLflow experiment comparison table."""

from __future__ import annotations

import logging
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_IMAGE = PROJECT_ROOT / "reports" / "mlflow_comparison.png"
logger = logging.getLogger("capture_mlflow_ui")


def main() -> None:
    OUTPUT_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path="/usr/bin/chromium",
            headless=True,
            args=["--no-sandbox", "--disable-gpu"],
        )
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()

        print("Navigating to MLflow UI...")
        page.goto(
            "http://localhost:5000/#/experiments/2",
            wait_until="domcontentloaded",
            timeout=15000,
        )
        time.sleep(3)

        # Close any popups/modals if present (e.g. 'Got it' or 'Detect Issues')
        try:
            got_it_btn = page.locator("button:has-text('Got it')")
            if got_it_btn.is_visible():
                got_it_btn.click()
                time.sleep(1)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Modal close skipped: %s", exc)

        # Switch to 'Model training' if visible
        try:
            model_training_tab = page.locator("text='Model training'")
            if model_training_tab.is_visible():
                model_training_tab.click()
                time.sleep(3)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Tab switch skipped: %s", exc)

        time.sleep(2)
        # Close the right-hand MLflow Assistant panel
        try:
            close_buttons = page.locator("button svg, div[role='complementary'] button").all()
            for btn in close_buttons:
                try:
                    btn.click(timeout=500)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Button click skipped: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Drawer close skipped: %s", exc)

        time.sleep(2)
        print(f"Saving screenshot to {OUTPUT_IMAGE}...")
        page.screenshot(path=str(OUTPUT_IMAGE), full_page=False)
        browser.close()
        print("Screenshot captured successfully!")


if __name__ == "__main__":
    main()
