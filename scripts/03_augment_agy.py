"""
Script 03b — Augment dataset with Gemini 3.5 Flash via google-generativeai SDK.

Uses google-generativeai with Vertex AI backend (ADC from hustle config).
N_WORKERS threads call the API in parallel — no subprocess session conflicts.

Generates:
  - Grammar errors: participle agreement, subjunctive, negation, avoir COD
  - Diacritic errors: circumflex, cedilla, elision, acute/grave
  - Structural errors: merged words, split words, word order, double letter
  - Clean hard negatives: literary prose, oral dialogue, complex syntax

Appends to data/examples/all_examples.jsonl

Requires:
  GOOGLE_CLOUD_PROJECT and Application Default Credentials configured.
"""

import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
from pathlib import Path
from threading import Lock


import yaml
from tqdm import tqdm

ROOT          = Path(__file__).parent.parent
cfg           = yaml.safe_load((ROOT / "config.yaml").read_text())
OUT_PATH      = ROOT / cfg["data"]["examples_out"]
AGY_CFG       = cfg.get("agy", {})
GEMINI_CFG    = cfg.get("gemini", {})
AUGMENT_CFG    = cfg.get("augment", {})
N_CASES       = AGY_CFG.get("n_hard_cases", cfg["ollama"]["n_hard_cases"])
N_WORKERS     = GEMINI_CFG.get("workers", AGY_CFG.get("workers", 6))
SYSTEM_PROMPT = cfg["system_prompt"].strip()
MAX_API_ATTEMPTS = int(AUGMENT_CFG.get("max_api_attempts", 2))
BASE_BACKOFF_S = float(AUGMENT_CFG.get("base_backoff_s", 1.0))
MAX_BACKOFF_S = float(AUGMENT_CFG.get("max_backoff_s", 8.0))
REQUEST_TIMEOUT_MS = int(AUGMENT_CFG.get("request_timeout_ms", 45_000))
RATE_LIMIT_COOLDOWN_S = float(AUGMENT_CFG.get("rate_limit_cooldown_s", 120.0))

VERTEX_PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR_GOOGLE_CLOUD_PROJECT")
VERTEX_LOCATION = "global"
MODEL_POOL_CFG  = GEMINI_CFG.get("model_pool", [
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    "gemini-2.5-pro",
    "gemini-3.1-flash-lite-preview",
])
MODEL_WEIGHTS   = GEMINI_CFG.get("model_weights", {})
MODEL_POOL      = [
    model.strip()
    for model in MODEL_POOL_CFG
    if str(model).strip()
]
if isinstance(MODEL_WEIGHTS, dict) and MODEL_WEIGHTS:
    weighted_pool: list[str] = []
    for model in MODEL_POOL:
        weight = int(MODEL_WEIGHTS.get(model, 1))
        weighted_pool.extend([model] * max(1, weight))
    MODEL_POOL = weighted_pool or MODEL_POOL

_client = None
STATS = Counter()
STAT_LOCK = Lock()
MODEL_STATE_LOCK = Lock()
MODEL_COOLDOWNS: dict[str, float] = {}


def bump(name: str, amount: int = 1) -> None:
    with STAT_LOCK:
        STATS[name] += amount


def mark_model_cooldown(model_name: str, seconds: float = RATE_LIMIT_COOLDOWN_S) -> None:
    until = time.monotonic() + max(1.0, seconds)
    with MODEL_STATE_LOCK:
        MODEL_COOLDOWNS[model_name] = max(MODEL_COOLDOWNS.get(model_name, 0.0), until)


def pick_model(task_index: int) -> tuple[str, float]:
    if not MODEL_POOL:
        return "gemini-3-flash-preview", 0.0

    with MODEL_STATE_LOCK:
        now = time.monotonic()
        start = task_index % len(MODEL_POOL)
        for offset in range(len(MODEL_POOL)):
            model = MODEL_POOL[(start + offset) % len(MODEL_POOL)]
            until = MODEL_COOLDOWNS.get(model, 0.0)
            if until <= now:
                return model, 0.0

        model, until = min(MODEL_COOLDOWNS.items(), key=lambda item: item[1])
        return model, max(0.0, until - now)

def init_gemini():
    global _client
    from google import genai as _genai
    from google.genai import types as _genai_types
    _client = _genai.Client(
        vertexai=True,
        project=VERTEX_PROJECT,
        location=VERTEX_LOCATION,
        http_options=_genai_types.HttpOptions(timeout=REQUEST_TIMEOUT_MS),
    )


def gemini_generate(prompt: str, model_name: str) -> str:
    transient_markers = (
        "429",
        "rate limit",
        "resource exhausted",
        "temporarily unavailable",
        "deadline exceeded",
        "timeout",
        "timed out",
        "503",
        "500",
        "internal error",
        "service unavailable",
    )
    for attempt in range(1, MAX_API_ATTEMPTS + 1):
        try:
            r = _client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            text = r.text.strip() if r.text else ""
            if not text:
                bump("api_empty_response")
            return text
        except Exception as exc:
            msg = str(exc).lower()
            retryable = any(marker in msg for marker in transient_markers)
            bump("api_errors")
            if any(marker in msg for marker in ("429", "rate limit", "resource exhausted", "quota exceeded")):
                bump("api_rate_limited")
                mark_model_cooldown(model_name)
                return ""
            if not retryable or attempt >= MAX_API_ATTEMPTS:
                bump("api_errors_final")
                return ""
            sleep_s = min(MAX_BACKOFF_S, BASE_BACKOFF_S * (2 ** (attempt - 1)))
            time.sleep(sleep_s)
    return ""


def check_gemini() -> bool:
    probe_model = MODEL_POOL[0] if MODEL_POOL else "gemini-3-flash-preview"
    return "OK" in gemini_generate("Reply with only the word: OK", probe_model)


# ── JSON parsing ───────────────────────────────────────────────────────────────

def parse_json(raw: str) -> dict | None:
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        start, end = raw.find("{"), raw.rfind("}") + 1
        if start >= 0 and end > start:
            try:
                return json.loads(raw[start:end])
            except json.JSONDecodeError:
                pass
    return None


# ── Error templates ────────────────────────────────────────────────────────────
# Each prompt requests a single JSON: {correct, wrong, explanation}

ERROR_TEMPLATES = [
    (
        "complex_pronominal",
        'Génère UNE phrase française avec un verbe pronominal au passé composé.\n'
        'Retourne UNIQUEMENT ce JSON, rien d\'autre:\n'
        '{"correct": "...", "wrong": "... (erreur d\'accord du participe pronominal)", "explanation": "Accord du participe passé avec un verbe pronominal : le participe s\'accorde avec le sujet quand le pronom réfléchi est COD."}\n'
        'Exemple: {"correct": "Ils se sont regardés longuement.", "wrong": "Ils se sont regardé longuement.", "explanation": "..."}'
    ),
    (
        "subjunctive_trigger",
        'Génère UNE phrase française avec "bien que", "pour que", "afin que" ou "avant que" + subjonctif.\n'
        'Retourne UNIQUEMENT ce JSON, rien d\'autre:\n'
        '{"correct": "... (subjonctif correct)", "wrong": "... (indicatif à la place du subjonctif)", "explanation": "La conjonction employée exige le subjonctif dans la subordonnée."}'
    ),
    (
        "negative_agreement",
        'Génère UNE phrase française au passé avec "ne...pas" suivi d\'un nom.\n'
        'Retourne UNIQUEMENT ce JSON, rien d\'autre:\n'
        '{"correct": "... (\'pas de/d\'\')", "wrong": "... (\'pas un/une\' — erreur)", "explanation": "Après \'ne...pas\', l\'article indéfini un/une devient la particule de/d\'."}'
    ),
    (
        "avoir_cod_agreement",
        'Génère UNE phrase où l\'auxiliaire avoir précède un COD qui s\'est déplacé avant le verbe (accord du participe passé avec avoir).\n'
        'Retourne UNIQUEMENT ce JSON, rien d\'autre:\n'
        '{"correct": "... (participe accordé)", "wrong": "... (participe non accordé — erreur)", "explanation": "Le participe passé avec avoir s\'accorde avec le COD antéposé."}'
    ),
    (
        "circumflex_error",
        'Génère UNE phrase littéraire française avec un mot portant un accent circonflexe (â, ê, î, ô, û).\n'
        'Retourne UNIQUEMENT ce JSON, rien d\'autre:\n'
        '{"correct": "...", "wrong": "... (accent circonflexe manquant)", "explanation": "Accent circonflexe manquant ou incorrect."}\n'
        'Exemple: {"correct": "La forêt était sombre.", "wrong": "La foret était sombre.", "explanation": "..."}'
    ),
    (
        "cedilla_error",
        'Génère UNE phrase française utilisant un mot avec cédille (ç).\n'
        'Retourne UNIQUEMENT ce JSON, rien d\'autre:\n'
        '{"correct": "...", "wrong": "... (ç remplacé par c)", "explanation": "La cédille est requise devant a, o, u pour le son [s]."}\n'
        'Exemple: {"correct": "Il commença à parler.", "wrong": "Il commenca à parler.", "explanation": "..."}'
    ),
    (
        "elision_error",
        'Génère UNE phrase française avec une élision obligatoire (l\', d\', j\', n\', etc.).\n'
        'Retourne UNIQUEMENT ce JSON, rien d\'autre:\n'
        '{"correct": "... (élision correcte)", "wrong": "... (élision manquante, ex: \'de ami\' au lieu de \'d\'ami\')", "explanation": "L\'élision est obligatoire devant voyelle ou h muet."}'
    ),
    (
        "double_letter",
        'Génère UNE phrase française avec un mot contenant une double consonne (ll, pp, mm, rr, ss, tt, ff, cc).\n'
        'Retourne UNIQUEMENT ce JSON, rien d\'autre:\n'
        '{"correct": "...", "wrong": "... (consonne simple à la place de la double)", "explanation": "Erreur de doublement de consonne."}\n'
        'Exemple: {"correct": "Elle appelle son ami.", "wrong": "Elle apelle son ami.", "explanation": "..."}'
    ),
    (
        "merged_words",
        'Génère UNE courte phrase française (6-10 mots).\n'
        'Retourne UNIQUEMENT ce JSON, rien d\'autre:\n'
        '{"correct": "...", "wrong": "... (deux mots collés sans espace)", "explanation": "Espace manquant entre deux mots."}\n'
        'Exemple: {"correct": "Il fait beau aujourd\'hui.", "wrong": "Il faitbeau aujourd\'hui.", "explanation": "..."}'
    ),
    (
        "word_order",
        'Génère UNE phrase française simple (sujet + verbe + adverbe ou complément court).\n'
        'Retourne UNIQUEMENT ce JSON, rien d\'autre:\n'
        '{"correct": "...", "wrong": "... (ordre de mots incorrect, ex: adverbe mal placé)", "explanation": "Erreur d\'ordre des mots dans la phrase."}\n'
        'Exemple: {"correct": "Il courait vite.", "wrong": "Il vite courait.", "explanation": "..."}'
    ),
]

CLEAN_TEMPLATES = [
    "Génère UNE phrase de prose littéraire française correcte, style XIXe siècle, 50-150 caractères. Vocabulaire riche. Donne UNIQUEMENT la phrase.",
    "Génère UNE réplique de dialogue en français parlé familier, linguistiquement correcte. Commence par —. Donne UNIQUEMENT la réplique.",
    "Génère UNE phrase fragmentaire littéraire française acceptable en contexte (style Flaubert). Exemple: 'Silence. Puis le vent.' Donne UNIQUEMENT la phrase.",
    "Génère UNE phrase française avec un subjonctif passé correctement employé. Donne UNIQUEMENT la phrase.",
    "Génère UNE phrase française avec 'bien que' + subjonctif, correcte. Donne UNIQUEMENT la phrase.",
    "Génère UNE phrase française avec une proposition relative complexe (dont, auquel, lequel) bien construite. Donne UNIQUEMENT la phrase.",
    "Génère UNE phrase française avec un gérondif ou participe présent correctement employé. Donne UNIQUEMENT la phrase.",
]


# ── Example builders ───────────────────────────────────────────────────────────

def diff_span(correct: str, wrong: str) -> tuple[int, int, str, str] | None:
    """Return (start, end_in_wrong, original, correction) or None if undiffable."""
    if correct == wrong or len(correct) < 10 or len(wrong) < 10:
        return None
    min_len = min(len(correct), len(wrong))
    start = 0
    for i in range(min_len):
        if correct[i] != wrong[i]:
            start = i
            break
    else:
        start = min_len  # one string is a prefix of the other
    end_c, end_w = len(correct) - 1, len(wrong) - 1
    while end_c >= start and end_w >= start and correct[end_c] == wrong[end_w]:
        end_c -= 1
        end_w -= 1
    original   = wrong[start:end_w + 1]
    correction = correct[start:end_c + 1]
    if not original or not correction:
        return None
    return start, end_w + 1, original, correction


def make_error_example(etype: str, prompt: str, model_name: str) -> dict | None:
    raw = gemini_generate(prompt, model_name)
    if not raw:
        bump("error_example_failed")
        return None
    data = parse_json(raw)
    if not data:
        bump("parse_failures")
        return None
    correct     = data.get("correct", "").strip().strip('"')
    wrong       = data.get("wrong",   "").strip().strip('"')
    explanation = data.get("explanation", "Erreur grammaticale détectée.")
    span = diff_span(correct, wrong)
    if not span:
        bump("diff_failures")
        return None
    start, end, original, correction = span
    bump("accepted_examples")
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Texte: {wrong}"},
            {"role": "assistant", "content": json.dumps({
                "errors": [{
                    "original":    original,
                    "correction":  correction,
                    "type":        etype,
                    "explanation": explanation,
                }]
            }, ensure_ascii=False)},
        ]
    }


def make_clean_example(prompt: str, model_name: str) -> dict | None:
    sentence = gemini_generate(prompt, model_name)
    if not sentence or len(sentence) < 15:
        bump("clean_example_failed")
        return None
    bump("accepted_examples")
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Texte: {sentence.strip()}"},
            {"role": "assistant", "content": '{"errors": []}'},
        ]
    }


def run_task(task_index: int, task: tuple) -> dict | None:
    kind, etype, prompt = task
    model_name, wait_s = pick_model(task_index)
    if wait_s > 0:
        bump("model_cooldown_waits")
        time.sleep(wait_s)
    if kind == "error":
        return make_error_example(etype, prompt, model_name)
    return make_clean_example(prompt, model_name)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    init_gemini()
    probe_model = MODEL_POOL[0] if MODEL_POOL else "gemini-3-flash-preview"
    print(f"Checking Gemini API ({probe_model}, project={VERTEX_PROJECT})...")
    if not check_gemini():
        print("ERROR: Gemini API not responding. Set GOOGLE_CLOUD_PROJECT and Application Default Credentials.")
        sys.exit(1)

    print(f"✓ Gemini ready  |  workers={N_WORKERS}  |  target={N_CASES} examples")
    print(f"  model pool: {', '.join(MODEL_POOL)}")

    per_template = max(1, N_CASES // (len(ERROR_TEMPLATES) + len(CLEAN_TEMPLATES)))

    tasks: list[tuple] = []
    for etype, prompt in ERROR_TEMPLATES:
        tasks.extend(("error", etype, prompt) for _ in range(per_template))
    for prompt in CLEAN_TEMPLATES:
        tasks.extend(("clean", None, prompt) for _ in range(per_template))
    random.shuffle(tasks)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    accepted_count = 0
    with OUT_PATH.open("a", encoding="utf-8") as f:
        with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {pool.submit(run_task, i, t): t for i, t in enumerate(tasks)}
            with tqdm(total=len(tasks), desc="Generating", unit="ex") as bar:
                for future in as_completed(futures):
                    ex = future.result()
                    if ex:
                        accepted_count += 1
                        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
                        f.flush()
                    bar.update(1)
                    bar.set_postfix(ok=accepted_count, rate=f"{accepted_count/max(bar.n,1):.0%}")

    print(f"\n✓ Done. Added {accepted_count}/{len(tasks)} examples → {OUT_PATH}")
    print(f"  Yield rate: {accepted_count/len(tasks):.0%}  (expect 70-85%)")
    print(
        "  Stats: "
        f"accepted={STATS['accepted_examples']} "
        f"api_errors={STATS['api_errors']} "
        f"rate_limited={STATS['api_rate_limited']} "
        f"api_empty={STATS['api_empty_response']} "
        f"parse_failures={STATS['parse_failures']} "
        f"diff_failures={STATS['diff_failures']} "
        f"cooldown_waits={STATS['model_cooldown_waits']}"
    )


if __name__ == "__main__":
    main()
