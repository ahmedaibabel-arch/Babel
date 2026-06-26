"""
Script 03 — Augment dataset with Ollama-generated hard cases.

Asks local Mistral (via Ollama) to generate:
  - Complex agreement sentences (past participle with avoir COD)
  - Subjunctive trigger errors
  - Register-appropriate clean literary prose (hard negatives)
  - Dialogue with orality markers (hard negatives)

Appends results to data/examples/all_examples.jsonl

Runtime: ~4-8 hours for 5,000 examples (runs overnight).
Requires: Ollama running with Mistral loaded.
"""

import json
import random
import sys
import time
from pathlib import Path
import requests
import yaml
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
cfg  = yaml.safe_load((ROOT / "config.yaml").read_text())

OUT_PATH      = ROOT / cfg["data"]["examples_out"]
OLLAMA_MODEL  = cfg["ollama"]["model"]
OLLAMA_URL    = cfg["ollama"]["endpoint"]
N_CASES       = cfg["ollama"]["n_hard_cases"]
SYSTEM_PROMPT = cfg["system_prompt"].strip()


def ollama_generate(prompt: str, max_tokens: int = 400) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": max_tokens}
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=60)
        r.raise_for_status()
        return r.json().get("response", "").strip()
    except Exception as e:
        return ""


def check_ollama() -> bool:
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


# ── Prompt templates ──────────────────────────────────────────────────────────

ERROR_TEMPLATES = [
    # Complex participle error
    (
        "Génère UNE phrase française correcte avec un verbe pronominal au passé composé "
        "suivi d'un participe passé accordé. Exemples : 'Elle s'est rendu compte trop tard.' "
        "'Ils se sont regardés longuement.' Donne uniquement la phrase, rien d'autre.",
        "complex_pronominal",
        "Puis génère la même phrase avec UNE SEULE erreur d'accord du participe passé pronominal. "
        "Donne uniquement la phrase erronée, rien d'autre."
    ),
    # Subjunctive trigger
    (
        "Génère UNE phrase française correcte utilisant une conjonction qui exige le subjonctif "
        "(bien que, pour que, afin que, avant que, à moins que). "
        "Exemple : 'Bien qu'il fût tard, elle continua.' Donne uniquement la phrase.",
        "subjunctive_trigger",
        "Puis génère la même phrase avec l'indicatif à la place du subjonctif (erreur). "
        "Donne uniquement la phrase erronée."
    ),
    # Negative agreement
    (
        "Génère UNE phrase française correcte au passé avec une négation 'ne...pas de' "
        "suivie d'un nom. Exemple : 'Il n'y avait pas de lumière dans la pièce.' "
        "Donne uniquement la phrase.",
        "negative_agreement",
        "Puis génère la même phrase avec 'un/une' au lieu de 'de/d'' après la négation (erreur). "
        "Donne uniquement la phrase erronée."
    ),
]

CLEAN_TEMPLATES = [
    "Génère UNE phrase de prose littéraire française correcte, style XIXe siècle, "
    "entre 50 et 150 caractères. Vocabulaire riche. Donne uniquement la phrase.",

    "Génère UNE réplique de dialogue en français parlé familier, correcte du point de vue "
    "linguistique mais avec un registre oral. Commence par un tiret (—). "
    "Donne uniquement la réplique.",

    "Génère UNE courte phrase fragmentaire littéraire française intentionnelle "
    "(style Flaubert, Zola, Maupassant), grammaticalement acceptable en contexte littéraire. "
    "Exemple : 'Silence. Puis le vent.' Donne uniquement la phrase.",

    "Génère UNE phrase française avec un subjonctif passé correctement employé. "
    "Donne uniquement la phrase.",

    "Génère UNE phrase française avec 'bien que' + subjonctif, correcte. "
    "Donne uniquement la phrase.",
]


def build_error_example_from_pair(correct: str, wrong: str,
                                   etype: str, explanation: str) -> dict | None:
    # Find the difference
    correct = correct.strip().rstrip('.')
    wrong   = wrong.strip()
    if correct == wrong or len(correct) < 15 or len(wrong) < 15:
        return None

    # Simple diff: find first char that differs
    min_len = min(len(correct), len(wrong))
    start   = 0
    for i in range(min_len):
        if correct[i] != wrong[i]:
            start = i
            break
    end_c = len(correct) - 1
    end_w = len(wrong)   - 1
    while end_c >= start and end_w >= start and correct[end_c] == wrong[end_w]:
        end_c -= 1
        end_w -= 1

    original_in_wrong   = wrong[start:end_w + 1]
    correction_from_correct = correct[start:end_c + 1]

    if not original_in_wrong or not correction_from_correct:
        return None

    assistant_json = {
        "errors": [{
            "start":       start,
            "end":         end_w + 1,
            "original":    original_in_wrong,
            "correction":  correction_from_correct,
            "type":        etype,
            "confidence":  0.88,
            "explanation": explanation
        }]
    }
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Texte: {wrong}"},
            {"role": "assistant", "content": json.dumps(assistant_json, ensure_ascii=False)}
        ]
    }


def build_clean_example(sentence: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Texte: {sentence.strip()}"},
            {"role": "assistant", "content": '{"errors": []}'}
        ]
    }


def main():
    if not check_ollama():
        print("WARNING: Ollama not reachable at localhost:11434.")
        print("Skipping augmentation. Run Ollama with Mistral and re-run this script.")
        print("(This step is optional — training can proceed without it.)")
        sys.exit(0)

    print(f"Ollama reachable. Model: {OLLAMA_MODEL}")
    print(f"Generating {N_CASES} augmented examples...")

    examples = []
    per_template = N_CASES // (len(ERROR_TEMPLATES) + len(CLEAN_TEMPLATES))

    # Error examples from Ollama
    for correct_prompt, etype, wrong_prompt_suffix in ERROR_TEMPLATES:
        explanation_map = {
            "complex_pronominal": "Accord du participe passé avec un verbe pronominal : le participe s'accorde avec le sujet quand le pronom réfléchi est COD.",
            "subjunctive_trigger": "La conjonction employée exige le subjonctif dans la proposition subordonnée.",
            "negative_agreement": "Après 'ne...pas', l'article indéfini 'un/une' devient la particule 'de/d''.",
        }
        expl = explanation_map.get(etype, "Erreur grammaticale détectée.")

        for _ in tqdm(range(per_template), desc=f"  {etype}"):
            correct = ollama_generate(correct_prompt, 150)
            if not correct or len(correct) < 20:
                time.sleep(0.3)
                continue
            wrong_prompt = f"Phrase correcte: {correct}\n{wrong_prompt_suffix}"
            wrong = ollama_generate(wrong_prompt, 150)
            if not wrong or wrong == correct or len(wrong) < 20:
                time.sleep(0.3)
                continue
            ex = build_error_example_from_pair(correct, wrong, etype, expl)
            if ex:
                examples.append(ex)
            time.sleep(0.1)

    # Clean hard negatives from Ollama
    for template in CLEAN_TEMPLATES:
        for _ in tqdm(range(per_template), desc=f"  clean"):
            sentence = ollama_generate(template, 150)
            if sentence and len(sentence) > 15:
                examples.append(build_clean_example(sentence))
            time.sleep(0.1)

    random.shuffle(examples)

    # Append to existing examples file
    with OUT_PATH.open("a", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n✓ Done. Added {len(examples)} Ollama-generated examples → {OUT_PATH}")


if __name__ == "__main__":
    main()
