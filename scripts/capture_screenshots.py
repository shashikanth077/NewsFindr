"""Capture README screenshots of the running NewsFindr app with Playwright.

Real end-to-end screenshots need a Groq key and internet access (live DuckDuckGo search):

    pip install -r requirements.txt playwright && playwright install chromium
    export GROQ_API_KEY=...            # never commit this
    python scripts/capture_screenshots.py                  # everything
    python scripts/capture_screenshots.py --only home,invalid_email   # no key/network needed

Images are written to docs/screenshots/. Search results are live, so they differ on each run.
"""
import argparse
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import TimeoutError as PWTimeout
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
RESULT_PATTERN = re.compile(r"personalised recommendation|No recent articles|Every DuckDuckGo|not registered|did not")

# name -> (demo customer button, request override)
E2E = {
    "john_generative_ai": "John",
    "david_llm": "David",
    "sarah_healthcare": "Sarah",
}


def wait_for_server(url: str, timeout: int = 60) -> None:
    end = time.time() + timeout
    while time.time() < end:
        try:
            urllib.request.urlopen(url, timeout=2)
            return
        except Exception:
            time.sleep(1)
    raise RuntimeError(f"Streamlit did not start at {url}")


def shot(page, out: Path, name: str) -> None:
    page.wait_for_timeout(800)
    page.screenshot(path=str(out / f"{name}.png"), full_page=True)
    print("saved", out / f"{name}.png")


def capture_end_to_end(page, out: Path, label: str, customer: str, key: str) -> bool:
    # Streamlit scrolls inside its main pane, so use a tall viewport to capture the whole result.
    page.set_viewport_size({"width": 1440, "height": 2200})
    page.goto(page.url.split("?")[0])
    page.get_by_text("Try a demo customer").wait_for()
    page.get_by_role("button", name=re.compile(customer)).first.click()
    page.wait_for_timeout(800)
    if key:
        box = page.get_by_role("textbox", name="Groq API key")
        box.fill(key)
        box.press("Enter")  # commit the value, then let Streamlit finish its rerun before submitting
        page.wait_for_timeout(1500)
    page.get_by_role("button", name="Find my news").click()
    try:
        page.get_by_text(RESULT_PATTERN).first.wait_for(timeout=300_000)
    except PWTimeout:
        print(f"!! {label}: no result within 5 minutes; skipping")
        return False
    page.wait_for_timeout(1500)
    shot(page, out, f"{label}_1_recommendations")
    for idx, (tab, suffix) in enumerate([("How it worked", "2_pipeline"), ("SQL agent trace", "3_agent_trace")], 1):
        page.get_by_role("tab", name=re.compile(tab)).click()
        shot(page, out, f"{label}_{suffix}")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="Use an already running app instead of starting one")
    ap.add_argument("--port", default="8599")
    ap.add_argument("--out", default=str(ROOT / "docs" / "screenshots"))
    ap.add_argument("--only", help="Comma list: home,invalid_email," + ",".join(E2E))
    args = ap.parse_args()
    wanted = set(args.only.split(",")) if args.only else {"home", "invalid_email", *E2E}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    key = os.environ.get("GROQ_API_KEY", "")

    server = None
    url = args.url or f"http://localhost:{args.port}"
    if not args.url:
        server = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", "app.py", "--server.headless", "true",
             "--server.port", args.port, "--browser.gatherUsageStats", "false"],
            cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_server(url)
        with sync_playwright() as p:
            exe = os.environ.get("PLAYWRIGHT_CHROMIUM_EXECUTABLE")
            browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 900})
            page.goto(url)
            page.get_by_text("Find the news that matters to you").wait_for(timeout=30_000)
            if "home" in wanted:
                shot(page, out, "01_home")
            if "invalid_email" in wanted:
                page.get_by_label("Customer email").fill("not-an-email")
                page.get_by_role("button", name="Find my news").click()
                page.get_by_text("valid email address").wait_for(timeout=15_000)
                shot(page, out, "02_invalid_email")
            for i, (label, customer) in enumerate(E2E.items(), start=3):
                if label in wanted:
                    if not key:
                        print(f"!! skipping {label}: set GROQ_API_KEY for end-to-end screenshots")
                        continue
                    capture_end_to_end(page, out, f"{i:02d}_{label}", customer, key)
            browser.close()
    finally:
        if server:
            server.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
