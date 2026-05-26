"""
take_screenshots.py
-------------------
Headless Playwright script that captures three README screenshots:
  1. swagger_ui.png       — FastAPI /docs (must have uvicorn running on :8000)
  2. mlflow_comparison.png — MLflow run-comparison table (starts mlflow ui on :5000)
  3. drift_report.png     — Evidently HTML drift report (file:// URL)

Output: docs/shap_samples/{swagger_ui,mlflow_comparison,drift_report}.png
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT  = ROOT / "docs" / "shap_samples"
OUT.mkdir(parents=True, exist_ok=True)

# ── helpers ───────────────────────────────────────────────────────────────────

def wait_http(url: str, timeout: int = 30) -> bool:
    import urllib.request, urllib.error
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=2)
            return True
        except Exception:
            time.sleep(1)
    return False


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    from playwright.sync_api import sync_playwright

    # ── 1. Start MLflow UI if not already running ─────────────────────────────
    mlflow_proc = None
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:5000", timeout=2)
        print("MLflow UI already running on :5000")
    except Exception:
        print("Starting MLflow UI on :5000 …")
        mlflow_proc = subprocess.Popen(
            [sys.executable, "-m", "mlflow", "ui",
             "--backend-store-uri", f"file:{ROOT / 'mlruns'}",
             "--port", "5000"],
            cwd=ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if not wait_http("http://localhost:5000", timeout=30):
            print("  WARNING: MLflow UI did not respond in 30 s — skipping that screenshot")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1400, "height": 900})

        # ── Screenshot 1: Swagger UI ──────────────────────────────────────────
        page = ctx.new_page()
        print("Navigating to Swagger UI …")
        page.goto("http://localhost:8000/docs", wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(1500)   # let animations settle

        # Expand first three endpoints so the page isn't just a collapsed list
        ops = page.locator(".opblock-summary").all()
        for op in ops[:6]:
            try:
                op.click()
                page.wait_for_timeout(300)
            except Exception:
                pass

        page.screenshot(path=str(OUT / "swagger_ui.png"), full_page=False)
        print("  [OK] Saved swagger_ui.png")
        page.close()

        # ── Screenshot 2: MLflow comparison ──────────────────────────────────
        page = ctx.new_page()
        print("Navigating to MLflow UI …")
        try:
            page.goto("http://localhost:5000", wait_until="networkidle", timeout=15000)
            page.wait_for_timeout(2000)

            # Navigate into the customer-intelligence experiment
            exp_link = page.locator("text=customer-intelligence").first
            exp_link.click()
            page.wait_for_timeout(2000)

            # Select all visible runs via checkboxes
            checkboxes = page.locator("input[type='checkbox']").all()
            for cb in checkboxes[:4]:          # first 4 — baseline + improved + stump
                try:
                    if not cb.is_checked():
                        cb.check()
                    page.wait_for_timeout(200)
                except Exception:
                    pass

            # Click the Compare button
            compare_btn = page.locator("button:has-text('Compare')").first
            compare_btn.click()
            page.wait_for_timeout(3000)

            page.screenshot(path=str(OUT / "mlflow_comparison.png"), full_page=False)
            print("  [OK] Saved mlflow_comparison.png")
        except Exception as exc:
            print(f"  WARNING: MLflow screenshot failed ({exc}) — saving whatever is on screen")
            page.screenshot(path=str(OUT / "mlflow_comparison.png"), full_page=False)
        page.close()

        # ── Screenshot 3: Evidently drift report ─────────────────────────────
        page = ctx.new_page()
        drift_html = ROOT / "monitoring" / "reports" / "ml_drift_report.html"
        print("Opening drift report …")
        page.goto(f"file:///{drift_html.as_posix()}", wait_until="networkidle", timeout=15000)
        page.wait_for_timeout(2000)

        # Scroll to the first drift chart (usually the age distribution)
        page.evaluate("window.scrollBy(0, 400)")
        page.wait_for_timeout(800)

        page.screenshot(path=str(OUT / "drift_report.png"), full_page=False)
        print("  [OK] Saved drift_report.png")
        page.close()

        browser.close()

    # ── cleanup ───────────────────────────────────────────────────────────────
    if mlflow_proc:
        mlflow_proc.terminate()
        print("MLflow UI stopped.")


    print("\nAll screenshots saved to docs/shap_samples/")


if __name__ == "__main__":
    main()
