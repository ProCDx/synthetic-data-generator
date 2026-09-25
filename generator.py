"""
generator.py
------------
Everything that is NOT user interface lives here:
  * the API clients for Gemini and Ollama
  * the prompts (system prompt + diversity hints)
  * the batch generation loop (round-robin across models)
  * defensive JSON parsing
  * cleaning the final table and saving it to CSV
  * validating rows against the schema (flags problems, never drops rows)

app.py only calls check_providers() and generate_dataset().
"""

import json
import math
import os
import random
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import openai
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Read variables from the .env file into os.environ.
# override=True means values in .env win over anything already set in the shell.
load_dotenv(override=True)

# ---------------------------------------------------------------------------
# Models and clients
# ---------------------------------------------------------------------------

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
OLLAMA_BASE_URL = "http://localhost:11434/v1"

# model name -> provider. Change models here and the whole app follows.
MODELS = {
    "gemini-3.1-flash-lite": "gemini",  # Google, free tier
    "gemma4:31b-cloud": "ollama",  # Ollama cloud model (runs remotely, proxied by local Ollama)
    "llama3.2": "ollama",          # Ollama local, 3B
    "qwen2.5:3b": "ollama",        # Ollama local, 3B
}

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# The openai library refuses to build a client with api_key=None,
# so we only create the Gemini client when the key actually exists.
# max_retries=1: the library retries 429/5xx errors itself with a back-off;
# one retry is enough and keeps the app from hanging on a used-up quota.
gemini_client = (
    OpenAI(base_url=GEMINI_BASE_URL, api_key=GOOGLE_API_KEY, timeout=120, max_retries=1)
    if GOOGLE_API_KEY
    else None
)

# Ollama ignores the API key, but the openai library requires *some* string.
# Local 3B models on a CPU can be slow, so we allow a generous timeout.
ollama_client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=300, max_retries=0)

# Where CSV files are written (a folder next to this file).
OUTPUT_DIR = Path(__file__).parent / "outputs"

# If a model fails this many batches in a row, stop sending it work.
MAX_CONSECUTIVE_FAILS = 3


def check_providers():
    """
    Check which providers are usable right now.

    Returns (status, messages):
      status   -> {"gemini": True/False, "ollama": True/False}
      messages -> list of human-readable warnings to show in the UI
    """
    status = {}
    messages = []

    # Gemini: we can only check that the key exists. A wrong key will show up
    # later as an "Auth" error in the generation log.
    if gemini_client is None:
        status["gemini"] = False
        messages.append(
            "GOOGLE_API_KEY is missing, so Gemini will be skipped. "
            "Add it to your .env file and restart the app."
        )
    else:
        status["gemini"] = True

    # Ollama: ask for the list of installed models. This is fast and tells us
    # both "is the server running?" and "which models are pulled?".
    try:
        # with_options() makes a copy of the client with a short timeout,
        # so a dead server doesn't freeze the UI for 5 minutes.
        installed = {m.id for m in ollama_client.with_options(timeout=3).models.list().data}
        status["ollama"] = True
        for name, provider in MODELS.items():
            # Ollama lists "llama3.2" as "llama3.2:latest", so add the default tag before comparing.
            full_name = name if ":" in name else f"{name}:latest"
            if provider == "ollama" and full_name not in installed:
                messages.append(f"Ollama model '{name}' is not installed. Run: ollama pull {name}")
    except Exception:
        status["ollama"] = False
        messages.append(
            "Could not reach Ollama at http://localhost:11434, so Ollama models will be skipped. "
            "Start the Ollama app (or run `ollama serve`) and click 'Re-check providers'."
        )

    return status, messages


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a synthetic data generator. You create realistic but entirely fictional records.
Rules:
- Respond with ONLY a JSON object. No markdown, no code fences, no explanations.
- The JSON must have exactly this shape: {"rows": [ {...}, {...} ]}
- Every row is a JSON object that uses exactly the keys you are given, spelled exactly the same.
- Respect each column's type: numbers as JSON numbers, booleans as true/false, everything else as strings.
- Never copy real people's personal data."""

# One of these is injected at random into each batch, so different batches
# (and different models) are pushed toward different corners of the data.
# Each entry is (short label for the log, instruction for the model).
DIVERSITY_HINTS = [
    ("typical", "Focus on typical, everyday records - the most common cases you'd see in real life."),
    ("edge cases", "Include edge cases: unusual but valid values, rare categories, boundary situations."),
    ("regional variety", "Maximise regional and demographic variety: different states/cities, age groups, "
                         "genders, languages and income levels."),
    ("messy", "Make the records messy and realistic: informal wording, typos, abbreviations, "
              "inconsistent capitalisation - but keep valid JSON and correct types."),
    ("numeric extremes", "Push numeric columns toward their extremes: very small and very large values, "
                         "zeros, and round vs. odd numbers."),
    ("negative/problem", "Focus on negative or problematic situations: complaints, rejections, errors, "
                         "low scores, disputes."),
    ("premium/high-end", "Focus on high-end or premium cases: large amounts, senior roles, "
                         "expensive products, VIP customers."),
]


def normalize_key(key):
    """Turn ' Order ID ' / 'order-id' into 'order_id' so small models' key spellings still match."""
    return re.sub(r"[\s\-]+", "_", str(key).strip().lower())


def parse_schema(schema_text):
    """
    Turn "name:string, age:int" into [("name", "string"), ("age", "int")].
    Tip: because columns are split on commas, write enum options with '|',
    e.g. "priority:enum(low|medium|high)".
    """
    columns = []
    seen = set()
    for part in schema_text.split(","):
        # partition(":") splits at the FIRST colon only -> ("name", ":", "type")
        name, _, col_type = part.strip().partition(":")
        name = normalize_key(name)
        if name and name not in seen:
            seen.add(name)
            columns.append((name, col_type.strip() or "string"))
    return columns


def build_user_prompt(purpose, columns, n_rows, hint):
    column_lines = "\n".join(f"- {name}: {col_type}" for name, col_type in columns)
    # The random "variation seed" is a cheap trick: a slightly different prompt
    # each time makes models less likely to return the same rows twice.
    # {{ and }} in an f-string produce literal { and } characters.
    return f"""Dataset purpose: {purpose}

Each row must have exactly these keys (key: type):
{column_lines}

Generate exactly {n_rows} rows.
Diversity instruction for this batch: {hint}
Do not repeat values across rows - vary names, places, numbers and wording.
(Variation seed: {random.randint(1000, 9999)})

Return JSON only, in the shape {{"rows": [ ... ]}}"""


# ---------------------------------------------------------------------------
# Calling models and parsing their answers
# ---------------------------------------------------------------------------

def call_model(model_name, messages, temperature):
    """Send one chat request and return the raw text of the answer."""
    client = gemini_client if MODELS[model_name] == "gemini" else ollama_client
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=temperature,
        # "JSON mode": both Gemini and Ollama support this and it makes
        # small models far more likely to return valid JSON.
        response_format={"type": "json_object"},
    )
    return response.choices[0].message.content


def parse_rows(text):
    """
    Turn a model's text answer into a list of row dicts.
    Raises ValueError (json.JSONDecodeError is a kind of ValueError) if nothing usable is found.
    """
    if not text or not text.strip():
        raise ValueError("empty response")

    # Some models "think out loud" inside <think> tags - drop that part.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # If the answer is wrapped in ```json ... ``` fences, keep only the inside.
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    # If there is chatter around the JSON ("Sure! Here is..."), cut from the
    # first { or [ to the last } or ].
    starts = [i for i in (text.find("{"), text.find("[")) if i != -1]
    end = max(text.rfind("}"), text.rfind("]"))
    if not starts or end == -1:
        raise ValueError("no JSON found in response")
    data = json.loads(text[min(starts): end + 1])

    # Accept several shapes:
    #   {"rows": [...]}        <- what we asked for
    #   [...]                  <- a bare list
    #   {"data": [...]} etc.   <- right idea, wrong key name
    #   {...}                  <- a single row
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        if isinstance(data.get("rows"), list):
            rows = data["rows"]
        else:
            lists = [v for v in data.values() if isinstance(v, list)]
            rows = lists[0] if lists else [data]
    else:
        raise ValueError(f"unexpected JSON type: {type(data).__name__}")

    clean_rows = []
    for row in rows:
        if not isinstance(row, dict):  # skip strings/numbers that slipped into the list
            continue
        clean_row = {}
        for key, value in row.items():
            # Nested lists/dicts would break duplicate detection later
            # (pandas can't compare them), so store them as JSON text.
            if isinstance(value, (list, dict)):
                value = json.dumps(value, ensure_ascii=False)
            clean_row[normalize_key(key)] = value
        clean_rows.append(clean_row)

    if not clean_rows:
        raise ValueError("JSON contained no row objects")
    return clean_rows


def describe_error(exc):
    """Short, human-readable error type for the log."""
    detail = str(exc).replace("\n", " ")[:150]
    # Order matters: more specific error classes must be checked before their parents.
    if isinstance(exc, openai.RateLimitError):
        return f"RateLimit (429) - too many requests or quota used up: {detail}"
    if isinstance(exc, openai.AuthenticationError):
        return f"Auth (401) - bad GOOGLE_API_KEY, or run `ollama signin` for cloud models: {detail}"
    if isinstance(exc, openai.PermissionDeniedError):
        return f"PermissionDenied (403) - key/account not allowed to use this model: {detail}"
    if isinstance(exc, openai.NotFoundError):
        return f"NotFound (404) - model name wrong or not pulled: {detail}"
    if isinstance(exc, openai.APITimeoutError):
        return "Timeout - the model took too long (large batch on a slow machine?)"
    if isinstance(exc, openai.APIConnectionError):
        return "Connection - could not reach the server (is Ollama running? internet OK?)"
    if isinstance(exc, openai.APIStatusError):
        return f"APIError ({exc.status_code}): {detail}"
    if isinstance(exc, json.JSONDecodeError):
        return f"BadJSON - model returned invalid JSON: {detail}"
    if isinstance(exc, ValueError):
        return f"BadOutput - {detail}"
    return f"{type(exc).__name__}: {detail}"


# ---------------------------------------------------------------------------
# Cleaning, stats and saving
# ---------------------------------------------------------------------------

def clean_dataframe(rows, column_names, limit=None):
    """Build a tidy table: schema columns only, no empty rows, no duplicates, at most `limit` rows."""
    df = pd.DataFrame(rows)
    # reindex keeps exactly these columns, in this order. Extra keys a model
    # invented are dropped; missing keys become empty (NaN).
    df = df.reindex(columns=column_names + ["source_model"])
    # Treat blank strings like "" or "   " as missing values.
    df = df.map(lambda v: pd.NA if isinstance(v, str) and not v.strip() else v)
    # Drop rows where every schema column is empty (source_model doesn't count).
    df = df.dropna(how="all", subset=column_names)
    # Drop exact duplicates. We ignore source_model so the same record from
    # two different models still counts as a duplicate.
    df = df.drop_duplicates(subset=column_names)
    if limit is not None:
        df = df.head(limit)
    # Missing values force whole-number columns to become floats (1 -> 1.0).
    # convert_dtypes() switches to pandas' "nullable" types so 1 stays 1.
    return df.reset_index(drop=True).convert_dtypes()


# ---------------------------------------------------------------------------
# Validation: flag problems in an "issues" column, never drop rows
# ---------------------------------------------------------------------------

# Near-duplicate checks only make sense on real free text (like "message"),
# not on short columns like "city" where repeats are normal. So the longest
# text column must average at least this many characters to be checked.
NEAR_DUP_MIN_AVG_LENGTH = 30


def parse_rules(columns):
    """Read the checkable rules out of the schema types, e.g. enum(a|b), int(1-5), date(YYYY-MM-DD)."""
    rules = {}
    for name, col_type in columns:
        t = col_type.strip().lower()  # lowercase so "Enum(...)" or "DATE(yyyy-mm-dd)" also work
        # fullmatch = the WHOLE type string must match the pattern, not just part of it.
        enum_match = re.fullmatch(r"enum\((.*)\)", t)
        # int(300-900) or float(1.0-5.0): group 1 = int/float, groups 2 and 3 = the two limits.
        # "-?" allows negative limits, "(?:\.\d+)?" allows an optional decimal part.
        range_match = re.fullmatch(r"(int|float)\(\s*(-?\d+(?:\.\d+)?)\s*-\s*(-?\d+(?:\.\d+)?)\s*\)", t)
        if enum_match:
            options = {o.strip() for o in enum_match.group(1).split("|") if o.strip()}
            rules[name] = {"kind": "enum", "options": options}
        elif range_match:
            rules[name] = {
                "kind": "range",
                "is_int": range_match.group(1) == "int",
                "low": float(range_match.group(2)),
                "high": float(range_match.group(3)),
            }
        elif t == "date(yyyy-mm-dd)":
            rules[name] = {"kind": "date"}
    return rules


def check_value(column, value, rule):
    """Return a short problem description if `value` breaks `rule`, otherwise None."""
    if rule["kind"] == "enum":
        # Compared case-insensitively, so "High" is accepted for enum(low|medium|high).
        if str(value).strip().lower() not in rule["options"]:
            return f"{column}: '{value}' not an allowed value"

    elif rule["kind"] == "range":
        # In Python True/False count as 1/0, so reject booleans explicitly.
        if isinstance(value, bool):
            return f"{column}: '{value}' is not a number"
        try:
            number = float(value)  # also accepts numeric strings like "750"
        except (TypeError, ValueError):
            return f"{column}: '{value}' is not a number"
        if not math.isfinite(number):  # float("nan") and float("inf") don't raise, so check separately
            return f"{column}: '{value}' is not a number"
        if rule["is_int"] and not number.is_integer():
            return f"{column}: {value} is not a whole number"
        if not rule["low"] <= number <= rule["high"]:
            # :g prints 300.0 as "300" and 1.5 as "1.5"
            return f"{column}: {value} outside {rule['low']:g}-{rule['high']:g}"

    elif rule["kind"] == "date":
        text = str(value).strip()
        # The regex checks the exact shape (strptime alone would accept "2024-1-5");
        # strptime then checks it is a real calendar date (rejects "2024-02-30").
        try:
            if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
                raise ValueError
            datetime.strptime(text, "%Y-%m-%d")
        except ValueError:
            return f"{column}: '{value}' is not a valid YYYY-MM-DD date"

    return None


def normalize_text(text):
    """Lowercase, remove punctuation and squeeze spaces, so "Late delivery!!" == "late  delivery"."""
    text = re.sub(r"[^\w\s]", "", text.lower())  # [^\w\s] = anything that is not a letter/digit/_ or space
    return " ".join(text.split())                # split() + join collapses runs of whitespace


def pick_text_column(df, column_names):
    """Return the non-ID column with the longest average text, or None if nothing is long enough."""
    best_column, best_length = None, 0.0
    for column in column_names[1:]:  # [1:] skips the first column, which is the ID
        texts = [v for v in df[column] if isinstance(v, str)]
        if not texts:
            continue
        average_length = sum(len(t) for t in texts) / len(texts)
        if average_length > best_length:
            best_column, best_length = column, average_length
    return best_column if best_length >= NEAR_DUP_MIN_AVG_LENGTH else None


def validate_dataframe(df, columns):
    """Return a copy of `df` with an "issues" column: '; '-separated problems, or '' if the row is valid."""
    column_names = [name for name, _ in columns]
    # One empty list of problems per row. We fill them in, then join them at the end.
    issues = [[] for _ in range(len(df))]

    # Checks 1-3: enum values, numeric ranges and date formats, cell by cell.
    for column, rule in parse_rules(columns).items():
        for i, value in enumerate(df[column]):
            if pd.isna(value):  # missing values are not checked here
                continue
            problem = check_value(column, value, rule)
            if problem:
                issues[i].append(problem)

    # Check 4: the first schema column is the ID, and IDs must be unique across the
    # whole dataset. Counter counts how often each ID appears.
    if column_names:
        id_column = column_names[0]
        ids = [None if pd.isna(v) else str(v).strip() for v in df[id_column]]
        id_counts = Counter(i for i in ids if i)
        for i, row_id in enumerate(ids):
            if row_id and id_counts[row_id] > 1:
                issues[i].append(f"duplicate {id_column} '{row_id}'")

    # Check 5: near-duplicates, meaning the same free text once case and punctuation are ignored.
    text_column = pick_text_column(df, column_names)
    if text_column:
        normalized = [normalize_text(v) if isinstance(v, str) else "" for v in df[text_column]]
        text_counts = Counter(t for t in normalized if t)
        for i, text in enumerate(normalized):
            if text and text_counts[text] > 1:
                issues[i].append(f"near-duplicate {text_column}")

    df = df.copy()  # don't modify the caller's table
    df["issues"] = ["; ".join(problems) for problems in issues]
    return df


def stats_table(stats, df):
    """Per-model comparison table shown in the UI (df must already have the "issues" column)."""
    # value_counts() gives {model_name: number_of_rows} for the cleaned data.
    contributed = df["source_model"].value_counts() if not df.empty else {}
    # Same count, but only for rows with no issues.
    valid = df.loc[df["issues"] == "", "source_model"].value_counts() if not df.empty else {}
    records = []
    for model, s in stats.items():
        attempts = s["ok"] + s["failed"]
        n_contributed = int(contributed.get(model, 0))
        n_valid = int(valid.get(model, 0))
        records.append({
            "model": model,
            "rows contributed (final)": n_contributed,
            "valid rows": n_valid,
            "validity rate": f"{n_valid / n_contributed:.0%}" if n_contributed else "n/a",
            "rows returned (raw)": s["rows_returned"],
            "successful batches": s["ok"],
            "failed batches": s["failed"],
            "failure rate": f"{s['failed'] / attempts:.0%}" if attempts else "n/a",
            # Time spent in successful batches divided by the rows they returned.
            "sec per row": f"{s['seconds'] / s['rows_returned']:.2f}" if s["rows_returned"] else "n/a",
        })
    return pd.DataFrame(records)


def save_csv(df):
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / f"synthetic_data_{datetime.now():%Y%m%d_%H%M%S}.csv"
    # "utf-8-sig" adds a marker so Excel shows characters like ₹ or Hindi names correctly.
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


# ---------------------------------------------------------------------------
# The main loop
# ---------------------------------------------------------------------------

def generate_dataset(purpose, schema_text, total_rows, rows_per_call, temperature, models):
    """
    Generate the dataset in batches, rotating through `models`.

    This is a *generator*: it `yield`s (log_text, preview_df, csv_path, stats_df)
    after every batch so the Gradio UI can update live. csv_path is None until the end.
    """
    total_rows = int(total_rows)          # sliders can hand us floats like 50.0
    rows_per_call = int(rows_per_call)
    columns = parse_schema(schema_text)
    column_names = [name for name, _ in columns]

    log = [f"Target: {total_rows} rows | {rows_per_call} rows per call | temperature {temperature} | "
           f"models: {', '.join(models)}", ""]
    # "seconds" = total time spent in this model's successful batches (for "sec per row").
    stats = {m: {"rows_returned": 0, "ok": 0, "failed": 0, "seconds": 0.0} for m in models}
    consecutive_fails = {m: 0 for m in models}
    active = list(models)   # models still in the rotation
    all_rows = []

    # Safety cap: 3x the number of batches we'd need if everything worked.
    # This guarantees the loop ends even if every call fails or returns duplicates.
    max_attempts = math.ceil(total_rows / rows_per_call) * 3 + len(models)
    attempt = 0
    df = clean_dataframe(all_rows, column_names)

    while len(df) < total_rows and attempt < max_attempts and active:
        # Round-robin: attempt 0 -> first model, 1 -> second, ... then wrap around.
        model = active[attempt % len(active)]
        attempt += 1
        n_rows = min(rows_per_call, total_rows - len(df))
        hint_label, hint_text = random.choice(DIVERSITY_HINTS)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(purpose, columns, n_rows, hint_text)},
        ]

        started = time.time()
        try:
            rows = parse_rows(call_model(model, messages, temperature))
        except Exception as exc:  # anything at all: log it and move on, never crash
            stats[model]["failed"] += 1
            consecutive_fails[model] += 1
            log.append(f"[{attempt:>3}] FAIL {model} ({hint_label}): {describe_error(exc)}")
            if consecutive_fails[model] >= MAX_CONSECUTIVE_FAILS:
                active.remove(model)
                log.append(f"      -> {model} failed {MAX_CONSECUTIVE_FAILS} times in a row; "
                           f"removed from the rotation.")
        else:  # runs only when the try block raised no exception
            elapsed = time.time() - started
            for row in rows:
                row["source_model"] = model
            all_rows.extend(rows)
            stats[model]["ok"] += 1
            stats[model]["rows_returned"] += len(rows)
            stats[model]["seconds"] += elapsed
            consecutive_fails[model] = 0
            log.append(f"[{attempt:>3}] OK   {model} ({hint_label}): {len(rows)} rows "
                       f"in {elapsed:.1f}s")

        df = clean_dataframe(all_rows, column_names)
        # Validate the whole table so far; duplicate IDs can span batches and models.
        df = validate_dataframe(df, columns)
        yield "\n".join(log), df, None, stats_table(stats, df)

    # ---- Finish up ----
    df = clean_dataframe(all_rows, column_names, limit=total_rows)
    df = validate_dataframe(df, columns)  # validate AFTER trimming, so flags match the final CSV
    log.append("")
    if not active:
        log.append("Stopped: every selected model was removed after repeated failures.")
    elif len(df) < total_rows:
        log.append(f"Stopped after {attempt} attempts (safety cap is {max_attempts}).")
    log.append(f"Done: {len(df)} unique rows from {len(all_rows)} raw rows.")

    csv_path = save_csv(df) if not df.empty else None
    if csv_path:
        log.append(f"Saved to {csv_path}")
    yield "\n".join(log), df, csv_path, stats_table(stats, df)
