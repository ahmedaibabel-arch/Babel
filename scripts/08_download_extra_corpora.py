"""
Script 08 v2 — Download additional French corpora for Babel training.

  Source 1: UD French GSD — grammar-annotated, news/reviews/blogs register
  Source 2: French Wikipedia — 500 random articles, encyclopedic formal register
  Source 3: Wikisource non-fiction — philosophy, speeches, essays via raw wikitext

All sentences appended to data/sentences/clean_sentences.jsonl.
Re-run: make corrupt → make dataset → make train
"""

import json
import re
import subprocess
import time
from pathlib import Path

import requests
from tqdm import tqdm
import yaml

ROOT     = Path(__file__).parent.parent
cfg      = yaml.safe_load((ROOT / "config.yaml").read_text())
OUT_PATH = ROOT / cfg["data"]["sentences_out"]
MIN_LEN  = cfg["data"]["min_sentence_len"]
MAX_LEN  = cfg["data"]["max_sentence_len"]

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

WIKI_HEADERS   = {"User-Agent": "Babel-Training/1.0 (personal linguistics research)"}
WIKI_API       = "https://fr.wikipedia.org/w/api.php"
WIKISOURCE_API = "https://fr.wikisource.org/w/api.php"

_ALPHA = re.compile(r'[a-zA-ZÀ-ÿ]')
_SPLIT = re.compile(r'(?<=[.!?…])\s+(?=[A-ZÁÀÂÄÉÈÊËÎÏÔÙÛÜŒ«"])')
_MARKUP = re.compile(
    r'\[\[[^\]]+\|([^\]]+)\]\]|\[\[([^\]]+)\]\]'
    r'|\{\{[^}]+\}\}|<[^>]+>'
    r'|\[https?://\S+[^\]]*\]'
    r"|'''?|==+[^=]+==+|\|[^\n]+|^\s*[*#:;]\s*",
    re.MULTILINE
)


def is_good(s: str) -> bool:
    s = s.strip()
    if len(s) < MIN_LEN or len(s) > MAX_LEN:
        return False
    alpha = len(_ALPHA.findall(s))
    if alpha < 20 or alpha / max(len(s), 1) < 0.55:
        return False
    if any(x in s for x in ['http', '{{', '[[', '|', '==']):
        return False
    return True


def extract_sentences(text: str, source: str) -> list[dict]:
    text = _MARKUP.sub(' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return [{"source": source, "sentence": p.strip()}
            for p in _SPLIT.split(text) if is_good(p.strip())]


def append_sentences(rows: list[dict], label: str) -> int:
    written = 0
    with OUT_PATH.open("a", encoding="utf-8") as f:
        for row in rows:
            if is_good(row["sentence"]):
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
    print(f"  ✓ {label}: {written:,} sentences added")
    return written


# ── Source 1: UD French GSD ───────────────────────────────────────────────────

def download_ud_gsd() -> list[dict]:
    tmp = Path("/tmp/UD_French_GSD")
    if not tmp.exists():
        print("  Cloning UD French GSD...")
        r = subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/UniversalDependencies/UD_French-GSD.git", str(tmp)],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            print(f"  ERROR: {r.stderr[:200]}")
            return []
    else:
        print("  UD French GSD already cloned — skipping download.")

    sentences = []
    for f in tmp.glob("*.conllu"):
        for line in f.open(encoding="utf-8"):
            if line.startswith("# text = "):
                text = line[len("# text = "):].strip()
                if text:
                    sentences.append({"source": "ud_french_gsd", "sentence": text})
    print(f"  Found {len(sentences):,} UD GSD sentences")
    return sentences


# ── Source 2: French Wikipedia (random articles) ──────────────────────────────

def _random_titles(n: int) -> list[str]:
    try:
        r = requests.get(WIKI_API, params={
            "action": "query", "list": "random",
            "rnnamespace": 0, "rnlimit": min(n, 500), "format": "json"
        }, headers=WIKI_HEADERS, timeout=15)
        return [p["title"] for p in r.json()["query"]["random"]]
    except Exception as e:
        print(f"  WARNING: {e}")
        return []


def _article_text(title: str) -> str:
    try:
        r = requests.get(WIKI_API, params={
            "action": "query", "titles": title,
            "prop": "extracts", "explaintext": True,
            "exsectionformat": "plain", "format": "json"
        }, headers=WIKI_HEADERS, timeout=20)
        pages = r.json()["query"]["pages"]
        return next(iter(pages.values())).get("extract", "")
    except Exception:
        return ""


def download_wikipedia(n_articles: int = 500) -> list[dict]:
    all_sentences: list[dict] = []
    fetched = 0
    pbar = tqdm(total=n_articles, desc="  Wikipedia articles")
    while fetched < n_articles:
        titles = _random_titles(min(50, n_articles - fetched))
        if not titles:
            break
        for title in titles:
            text = _article_text(title)
            if len(text) >= 400:
                all_sentences.extend(extract_sentences(text, "wikipedia_fr"))
            fetched += 1
            pbar.update(1)
            time.sleep(0.2)
        time.sleep(0.5)
    pbar.close()
    print(f"  Found {len(all_sentences):,} Wikipedia sentences from {fetched} articles")
    return all_sentences


# ── Source 3: Wikisource non-fiction via raw wikitext ─────────────────────────

# Verified page titles on fr.wikisource.org
WIKISOURCE_PAGES = [
    "Déclaration des droits de l'homme et du citoyen de 1789",
    "Du Contrat social",
    "Discours de la méthode",
    "De l'esprit des lois",
    "Pensées (Pascal)",
    "Lettres philosophiques",
    "Discours préliminaire de l'Encyclopédie",
    "Qu'est-ce qu'une nation ?",
    "Introduction à la médecine expérimentale",
    "Discours sur l'origine et les fondements de l'inégalité parmi les hommes",
    "Préface de Cromwell",
    "De l'Amour (Stendhal)",
    "Qu'est-ce que le Tiers-État ?",
    "Discours sur la misère (Victor Hugo)",
    "Discours de réception de Victor Hugo à l'Académie française",
    "Lettre à d'Alembert sur les spectacles",
    "Correspondance de Voltaire",
    "Préface de la Comédie humaine",
    "Émile ou De l'éducation",
    "De la démocratie en Amérique",
]


def _wikisource_text(title: str) -> str:
    try:
        r = requests.get(WIKISOURCE_API, params={
            "action": "query", "titles": title,
            "prop": "revisions", "rvprop": "content",
            "rvslots": "main", "format": "json"
        }, timeout=20)
        pages = r.json()["query"]["pages"]
        page  = next(iter(pages.values()))
        revs  = page.get("revisions", [])
        if not revs:
            return ""
        return revs[0].get("slots", {}).get("main", {}).get("*", "")
    except Exception:
        return ""


def download_wikisource() -> list[dict]:
    all_sentences: list[dict] = []
    for title in tqdm(WIKISOURCE_PAGES, desc="  Wikisource pages"):
        text = _wikisource_text(title)
        if len(text) >= 300:
            all_sentences.extend(extract_sentences(text, "wikisource_nonfiction"))
        time.sleep(0.4)
    print(f"  Found {len(all_sentences):,} Wikisource sentences")
    return all_sentences


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    existing = sum(1 for _ in OUT_PATH.open(encoding="utf-8")) if OUT_PATH.exists() else 0
    print(f"Existing sentences in pool: {existing:,}\n")

    total = 0

    print("Source 1: UD French GSD (grammar-annotated, diverse register)")
    total += append_sentences(download_ud_gsd(), "UD French GSD")
    print()

    print("Source 2: French Wikipedia (500 random articles, encyclopedic)")
    total += append_sentences(download_wikipedia(500), "French Wikipedia")
    print()

    print("Source 3: Wikisource non-fiction (philosophy, speeches, essays)")
    total += append_sentences(download_wikisource(), "Wikisource non-fiction")
    print()

    print("═══════════════════════════════════")
    print(f"  Added:     {total:,} new sentences")
    print(f"  New total: {existing + total:,} sentences")
    print(f"  File:      {OUT_PATH}")
    print()
    print("  Next: make corrupt → make dataset → make train")
    print("═══════════════════════════════════")


if __name__ == "__main__":
    main()
