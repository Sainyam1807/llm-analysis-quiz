import asyncio
import sys

import base64
from bs4 import BeautifulSoup

import json
import os
import re
import subprocess
import tempfile
import time
from json import JSONDecodeError
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Request
from fastapi.responses import JSONResponse

from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
from urllib.parse import urljoin


# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

QUIZ_SECRET = os.getenv("QUIZ_SECRET")
AIPIPE_TOKEN = os.getenv("AIPIPE_TOKEN")


if not QUIZ_SECRET:
    raise RuntimeError("QUIZ_SECRET is not set in .env")
if not AIPIPE_TOKEN:
    raise RuntimeError("AIPIPE_TOKEN is not set in .env")


# AIPIPE endpoint (not in .env)
AIPIPE_URL = "https://aipipe.org/openrouter/v1/chat/completions"

# Hard limit (3 minutes)
QUIZ_TIME_LIMIT_SECONDS = 180

app = FastAPI(title="LLM Analysis Quiz Server")


# -----------------------------
# Helpers: quiz parsing
# -----------------------------
def extract_urls(text: str) -> List[str]:
    """Find all URLs in the quiz text."""
    pattern = r"https?://[^\s\"'<)]+"
    return re.findall(pattern, text)


def find_submit_url(text: str) -> Optional[str]:
    urls = extract_urls(text)

    # FIRST: absolute URLs containing "submit"
    for u in urls:
        if "submit" in u.lower():
            return u

    # SECOND: detect relative /submit manually
    rel = re.search(r"\s(/submit)\b", text)
    if rel:
        return rel.group(1)

    return None

def extract_urls_from_html(html: str, base_url: str) -> List[str]:
    """Extract absolute URLs from HTML <a href> links."""
    soup = BeautifulSoup(html, "html.parser")
    urls = []
    for tag in soup.find_all("a", href=True):
        full = urljoin(base_url, tag["href"])
        urls.append(full)
    return urls

def parse_quiz_instructions(page_text: str, html: str, base_url: str) -> Dict[str, Any]:
    print("\n========================")
    print("[DEBUG] Parsing Page Text")
    print("========================\n")

    print("[DEBUG] FULL TEXT (first 1000 chars):\n")
    print(page_text[:1000])
    print("\n-------------------------------------\n")

    # URLs from visible text
    text_urls = extract_urls(page_text)

    print("[DEBUG] URLs FOUND (text only):")
    for u in text_urls:
        print("  →", u)

    # Detect submit URL
    submit_url = find_submit_url(page_text)

    # Check HTML for <a href="...">
    soup = BeautifulSoup(html, "html.parser")
    html_urls = []
    for tag in soup.find_all("a", href=True):
        full = urljoin(base_url, tag["href"])
        html_urls.append(full)
        if "submit" in tag["href"].lower():
            submit_url = full

    print("\n[DEBUG] SUBMIT URL DETECTED:", submit_url)

    # Combined URLs
    all_urls = list(set(text_urls + html_urls))

    # Detect data URLs from combined list
    data_urls = [
        u for u in all_urls
        if any(ext in u.lower() for ext in [".csv", ".xlsx", ".xls", ".json", ".pdf"])
    ]

    print("\n[DEBUG] DATA URLs:", data_urls)
    print("\n-------------------------------------\n")

    return {
        "question_text": page_text,
        "submit_url": submit_url,
        "data_urls": data_urls,
        "all_urls": all_urls,
    }



# -----------------------------
# Helpers: LLM call to AIPIPE
# -----------------------------
async def call_aipipe_llm(prompt: str) -> str:
    """Call AIPIPE chat completions API."""
    headers = {
        "Authorization": f"Bearer {AIPIPE_TOKEN}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "model": "openai/gpt-4.1-mini",
        "max_tokens": 2000,
        "temperature": 0.2,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert Python developer. "
                    "You ONLY output valid Python code, nothing else. "
                    "No explanations. No markdown. "
                    "The script runs in Python 3.11 with internet access."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    }

    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(AIPIPE_URL, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        raise RuntimeError(f"Unexpected LLM response format: {data}")


# -----------------------------
# Sanitize Generated Code
# -----------------------------
def sanitize_generated_code(code: str) -> str:
    """Remove unsafe patterns from LLM code."""
    banned = [
        "os.system",
        "subprocess.Popen",
        "subprocess.call",
        "subprocess.run(",
        "import shutil",
        "shutil.rmtree",
    ]

    for b in banned:
        if b in code:
            code = code.replace(b, f"# BANNED:{b}")

    return code

# -----------------------------
# Run Generated Python Script
# -----------------------------
def run_generated_script(code: str, task_input: Dict[str, Any],
                         timeout_seconds: int = 120) -> Dict[str, Any]:

    temp_dir = tempfile.mkdtemp(prefix="llm_quiz_")
    script_path = os.path.join(temp_dir, "solver.py")
    task_path = os.path.join(temp_dir, "task.json")

    with open(task_path, "w", encoding="utf-8") as f:
        json.dump(task_input, f)

    safe_code = sanitize_generated_code(code)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(safe_code)

    # -------------------------------------------------------------
    # UPDATED: use same interpreter as FastAPI (guarantees packages)
    # -------------------------------------------------------------
    result = subprocess.run(
        [sys.executable, script_path],    # <<< UPDATED
        cwd=temp_dir,
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Generated script failed.\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("Generated script produced no output.")

    # Pick only lines that LOOK like JSON
    json_candidates = [
        line for line in lines
        if line.startswith("{") or line.startswith("[")
   ]
    
    if not json_candidates:
        raise RuntimeError("Generated script produced no valid JSON output.")

    last_line = json_candidates[-1]

    try:
        return json.loads(last_line)
    except Exception:
        raise RuntimeError(f"Generated script output is not valid JSON: {last_line}")



# -----------------------------
# Browserless-based Page Fetch
# -----------------------------
async def fetch_page_content(url: str):
    """
    Asynchronous wrapper that runs Playwright sync version inside a thread.
    This avoids Windows / Python 3.12 subprocess incompatibilities.
    """
    print("\n========================")
    print(f"[DEBUG] Fetching URL via Playwright (sync wrapper): {url}")
    print("========================\n")

    # Runs fetch_page_sync() in a thread so Playwright can launch normally
    html, text = await asyncio.to_thread(fetch_page_sync, url)

    return html, text

def fetch_page_sync(url: str):
    """
    All Playwright browser logic is executed here, inside a background thread.
    This prevents subprocess issues on Windows Python 3.12.
    """
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True,args=[
        "--no-sandbox",
        "--disable-setuid-sandbox",
        "--disable-gpu",
        "--disable-dev-shm-usage",
        "--disable-web-security",
        "--allow-running-insecure-content",
        ])
        page = browser.new_page()

        # Load page fully
        page.goto(url, timeout=60000)
        page.wait_for_load_state("networkidle")

        # Fully rendered HTML (with JavaScript executed)
        html = page.content()

        print("\n[DEBUG] RAW RENDERED HTML (first 1000 chars):\n")
        print(html[:1000])
        print("\n-------------------------------------\n")

        # Extract full visible text
        try:
            text = page.inner_text("body")
        except Exception:
            text = ""

        print("\n[DEBUG] RENDERED TEXT (first 1000 chars):\n")
        print(text[:1000])
        print("\n-------------------------------------\n")

        # ---------------------------------------------------
        # Optional: detect Base64-encoded atob(`...`) content
        # ---------------------------------------------------
        # ------------------- PATCHED BASE64 DETECTOR ---------------------
        # Detect atob("..."), atob('...'), or atob(`...`)
        base64_pattern = r'[`\'"]([A-Za-z0-9+/=]{20,})[`\'"]'
        match = re.search(base64_pattern, html)
        
        if match:
            encoded = match.group(1).strip()
            print("\n[DEBUG] Base64 detected! Trying decode...\n")
            try:
                decoded_bytes = base64.b64decode(encoded)
                decoded_html = decoded_bytes.decode("utf-8", errors="ignore")
                print(decoded_html[:1000])
                print("\n-------------------------------------\n")
                soup = BeautifulSoup(decoded_html, "html.parser")
                decoded_text = soup.get_text(" ", strip=True)
                browser.close()
                return decoded_html, decoded_text
            except Exception as e:
                print("[DEBUG] Base64 decode failed:", e)
# ---------------------------------------------------------------


        # No atob → return normal rendered page
        browser.close()
        return html, text

# -----------------------------
# LLM: Generate Solver Script
# -----------------------------
async def generate_solver_code_for_quiz(
    quiz_url: str, question_text: str, data_urls: List[str], submit_url: str
) -> str:

    prompt = f"""
You must write a complete Python script that reads task.json and prints exactly ONE line of output.

### STRICT OUTPUT FORMAT RULES ###
- The script MUST print exactly ONE line.
- That ONE line MUST be valid JSON.
- The JSON MUST be exactly of the form: {{"answer": <value>}}
- <value> may be a number, string, boolean, list, dict, or any valid JSON serializable value.
- DO NOT print Python dicts (use json.dumps).
- DO NOT print extra logs, prints, text, or comments. Only one JSON line.
- You must NOT output any Python comments (no lines starting with #).

### SCRIPT REQUIREMENTS ###
- Read task.json (no CLI args).
- Use quiz_url, question_text, and data_urls to determine what to compute.
- Download any CSV/JSON/XLSX/PDF files if needed.
- For PDFs, you may use pdfplumber or PyPDF2 or tabula-py.
- You may use: requests, BeautifulSoup, pandas, numpy, matplotlib, networkx.
- Convert NumPy types (e.g., int64, float32) into native Python types before json.dumps.
  Use: value = int(value) or float(value)
- If you generate plots, base64 encode them into the answer.
- DO NOT submit any HTTP requests to the submit_url.
- DO NOT print anything except the required JSON line.
- If the HTML page contains <script> blocks, inline JavaScript, 
  or Base64-encoded strings (e.g., inside atob(`...`)), you MUST extract 
  and decode them to find hidden secret codes or text.
- If the page text already contains a decoded value (e.g., "secret_code: 4242"), you MUST extract the numeric/string value directly from question_text.
- Search for patterns like "<key>: <value>" and extract <value>.


### CONTEXT ###
- quiz_url: {quiz_url}
- data_urls: {data_urls}
- question_text (first 2000 chars):
{question_text[:2000]}
- DO NOT POST to submit_url: {submit_url}

Now output ONLY valid Python code. No markdown. No explanations.
"""

    code = await call_aipipe_llm(prompt)
    return code



# -----------------------------------------------------
# Solve Single Quiz (with retry logic)
# -----------------------------------------------------
async def solve_single_quiz(quiz_url: str, email: str, secret: str, start_time: float):
    from urllib.parse import urlparse

    # Track first-quiz start time
    if time.time() - start_time > QUIZ_TIME_LIMIT_SECONDS:
        print("Time limit reached before starting.")
        return None

    print(f"[QUIZ] Solving: {quiz_url}")
    
    html, text = await fetch_page_content(quiz_url)
    
    # NEW: extract URLs from real HTML
    urls_html = extract_urls_from_html(html, quiz_url)
    
    parsed = parse_quiz_instructions(text, html, quiz_url)
    
    # Merge into parsed data
    parsed["all_urls"] = list(set(parsed["all_urls"] + urls_html))
    
    parsed["data_urls"] = [
        u for u in parsed["all_urls"]
        if any(ext in u.lower() for ext in [".csv", ".xlsx", ".xls", ".json", ".pdf"])
    ]

    submit_url = parsed.get("submit_url")
    question_text = parsed["question_text"]
    data_urls = parsed["data_urls"]

    print("\n[DEBUG] Parsed submit_url BEFORE fix:", submit_url)

    # Fix relative submit URL
    if submit_url and submit_url.startswith("/"):
        p = urlparse(quiz_url)
        submit_url = f"{p.scheme}://{p.netloc}{submit_url}"
        print("[DEBUG] Reconstructed FULL submit URL:", submit_url)
    elif not submit_url:
        print("[QUIZ] No submit URL found. Stopping.")
        return None

    # -----------------------------------------
    # RETRY LOOP (retry same quiz until correct or timeout)
    # -----------------------------------------
    attempt = 1

    while time.time() - start_time < QUIZ_TIME_LIMIT_SECONDS:
        print(f"\n[QUIZ] Attempt #{attempt} for {quiz_url}")

        # Generate solver code
        solver_code = await generate_solver_code_for_quiz(
            quiz_url, question_text, data_urls, submit_url
        )

        # Prepare input for solver script
        task_input = {
            "quiz_url": quiz_url,
            "question_text": question_text,
            "data_urls": data_urls,
            "submit_url": submit_url,
            "email": email,
            "secret": secret,
            "url_for_payload": quiz_url,
        }

        # Run solver script
        try:
            answer_json = run_generated_script(
                solver_code,
                task_input,
                timeout_seconds=max(20, QUIZ_TIME_LIMIT_SECONDS - int(time.time() - start_time)),
            )
        except Exception as e:
            print("[QUIZ] Script error:", e)
            attempt += 1
            continue  # retry

        if "answer" not in answer_json:
            print("[QUIZ] Missing 'answer'. Retrying...")
            attempt += 1
            continue

        answer_value = answer_json["answer"]

        # Submit answer to server
        submission_payload = {
            "email": email,
            "secret": secret,
            "url": quiz_url,
            "answer": answer_value,
        }

        print(f"[QUIZ] Submitting → {submit_url} : {submission_payload}")

        async with httpx.AsyncClient(timeout=60) as client:
            try:
                try:
                     # first attempt
                    resp = await client.post(submit_url, json=submission_payload)
                except TypeError:
                    print("[QUIZ] Serialization error — coercing answer to safe type")

                    # -------------------------------
                    # HARDENING ADDED (PATCH 1)
                    # -------------------------------
                    ans = answer_value

            # Try to coerce numpy/pandas types → Python native
                    try:
                        if hasattr(ans, "item"):
                            ans = ans.item()
                    except Exception:
                        pass

            # fallback heuristics
                    if isinstance(ans, (list, tuple)):
                        ans = [x.item() if hasattr(x, "item") else x for x in ans]
                    elif isinstance(ans, dict):
                        safe_dict = {}
                        for k, v in ans.items():
                            if hasattr(v, "item"):
                                safe_dict[k] = v.item()
                            else:
                                safe_dict[k] = v
                        ans = safe_dict
                    else:
                # final fallback to str
                        try:
                            ans = int(ans)
                        except Exception:
                            try:
                                ans = float(ans)
                            except Exception:
                                ans = str(ans)

                    submission_payload["answer"] = ans
                    resp = await client.post(submit_url, json=submission_payload)

            except Exception as e:
                print("[QUIZ] Submit error:", e)
                attempt += 1
                continue

        if resp.status_code != 200:
            print("[QUIZ] Submit endpoint error:", resp.status_code)
            attempt += 1
            continue

        try:
            resp_json = resp.json()
        except Exception:
            print("[QUIZ] Submit returned non-JSON.")
            attempt += 1
            continue

        correct = resp_json.get("correct", False)
        next_url = resp_json.get("url")

        print(f"[QUIZ] correct={correct} next_url={next_url}")

        # -----------------------------------------
        # DECISION LOGIC
        # -----------------------------------------

        if correct:
            # Successful → follow next_url
            return True, next_url

        else:
            # WRONG ANSWER
            if next_url:
                # You have option to skip OR retry
                print("[QUIZ] Wrong answer — next_url provided. Deciding...")

                # For now: always retry once, then skip.
                if attempt < 2:
                    print("[QUIZ] RETRYING same quiz (attempt < 2)")
                    attempt += 1
                    continue
                else:
                    print("[QUIZ] SKIPPING to next_url")
                    return False, next_url

            else:
                # No next_url → must retry until timeout
                print("[QUIZ] Wrong AND no next_url → MUST retry.")
                attempt += 1
                continue

    print("[QUIZ] Time limit reached. Stopping attempts.")
    return None


# -----------------------------------------------------
# Solve Quiz Chain
# -----------------------------------------------------
async def solve_quiz_chain(initial_data: Dict[str, Any], start_time: float):
    email = initial_data.get("email")
    secret = initial_data.get("secret")
    url = initial_data.get("url")

    if not (email and secret and url):
        print("[CHAIN] Missing fields.")
        return

    current_url = url

    while current_url and time.time() - start_time < QUIZ_TIME_LIMIT_SECONDS:
        result = await solve_single_quiz(current_url, email, secret, start_time)
        if result is None:
            print("[CHAIN] Stopping chain (error or timeout).")
            break

        correct, next_url = result

        if not next_url:
            print("[CHAIN] Finished — no further quizzes.")
            break

        current_url = next_url

    print("[CHAIN] Done.")



# -----------------------------
# FastAPI Endpoint
# -----------------------------
@app.post("/quiz")
async def quiz_endpoint(request: Request, background_tasks: BackgroundTasks):
    try:
        data = await request.json()
    except JSONDecodeError:
        return JSONResponse(status_code=400, content={"error": "Invalid JSON"})

    if data.get("secret") != QUIZ_SECRET:
        return JSONResponse(status_code=403, content={"error": "Invalid secret"})

    if "email" not in data or "url" not in data:
        return JSONResponse(status_code=400, content={"error": "Missing email/url"})

    start_time = time.time()

    background_tasks.add_task(solve_quiz_chain, data, start_time)

    return JSONResponse(
        status_code=200,
        content={"status": "accepted", "message": "Quiz solving started in background"},
    )
