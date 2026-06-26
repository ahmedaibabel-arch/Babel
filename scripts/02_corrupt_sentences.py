"""
Script 02 — Generate training examples via deterministic corruption.

For each clean sentence:
  - 65% chance: introduce exactly one error (with known ground truth)
  - 35% chance: keep clean (model must output {"errors": []})

Error types:
  Homophones        à/a, ou/où, et/est, son/sont, etc.
  Accent acute/grave  décidé→decide, était→etait
  Accent circumflex  trône→trone, être→etre, pêche→peche
  Cedilla            aperçut→apercut, façon→facon
  Accent position    cérémonie→cerémonie (wrong vowel accented)
  Merged words       "certes le" → "certesle"
  Split word         "important" → "im portant"
  Word order swap    "il courait vite" → "il vite courait"
  Double letter      "radieux" → "raddieux"
  Typography         missing space before ?!;:
  Apostrophe         curly → straight
  Guillemets         « » → " "
  Missing elision    "d'enfants" → "de enfants"

Ground truth is always exact — we introduced the error.
No AI needed.

Runtime: ~15-25 min for 500K sentences.
"""

import json
import random
import re
import sys
from pathlib import Path
import yaml
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
cfg  = yaml.safe_load((ROOT / "config.yaml").read_text())

SENTENCES_PATH = ROOT / cfg["data"]["sentences_out"]
OUT_PATH       = ROOT / cfg["data"]["examples_out"]
TARGET         = cfg["data"]["target_examples"]
CLEAN_RATIO    = cfg["data"]["clean_ratio"]
SEED           = cfg["corruption"]["seed"]
SYSTEM_PROMPT  = cfg["system_prompt"].strip()

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
random.seed(SEED)

taxonomy = yaml.safe_load((ROOT / "rules/error_taxonomy.yaml").read_text())

# ── Homophones ─────────────────────────────────────────────────────────────────

HOMOPHONES: list[tuple[str, str, str, str]] = [
    (r'\bà\b',    'a',    'homophone_à_a',
     "« à » (préposition) confondu avec « a » (verbe avoir, 3e sg.)."),
    (r'\ba\b',    'à',    'homophone_a_à',
     "« a » (verbe avoir, 3e sg.) confondu avec « à » (préposition)."),
    (r'\boù\b',   'ou',   'homophone_où_ou',
     "« où » (adverbe/pronom de lieu) confondu avec « ou » (conjonction)."),
    (r'\bou\b',   'où',   'homophone_ou_où',
     "« ou » (conjonction) confondu avec « où » (adverbe de lieu)."),
    (r'\bet\b',   'est',  'homophone_et_est',
     "« et » (conjonction) confondu avec « est » (verbe être, 3e sg.)."),
    (r'\best\b',  'et',   'homophone_est_et',
     "« est » (verbe être) confondu avec « et » (conjonction)."),
    (r'\bson\b',  'sont', 'homophone_son_sont',
     "« son » (adjectif possessif) confondu avec « sont » (verbe être, 3e pl.)."),
    (r'\bsont\b', 'son',  'homophone_sont_son',
     "« sont » (verbe être, 3e pl.) confondu avec « son » (possessif)."),
    (r'\bces\b',  'ses',  'homophone_ces_ses',
     "« ces » (démonstratif pluriel) confondu avec « ses » (possessif pluriel)."),
    (r'\bses\b',  'ces',  'homophone_ses_ces',
     "« ses » (possessif pluriel) confondu avec « ces » (démonstratif pluriel)."),
    (r'\bce\b',   'se',   'homophone_ce_se',
     "« ce » (démonstratif) confondu avec « se » (pronom réfléchi)."),
    (r'\bse\b',   'ce',   'homophone_se_ce',
     "« se » (pronom réfléchi) confondu avec « ce » (démonstratif)."),
    (r'\bleur\b', 'leurs','homophone_leur_leurs',
     "« leur » (possessif invariable) confondu avec « leurs » (possessif pluriel)."),
    (r'\bleurs\b','leur', 'homophone_leurs_leur',
     "« leurs » (possessif pluriel) confondu avec « leur »."),
    (r'\bpeu\b',  'peut', 'homophone_peu_peut',
     "« peu » (adverbe de quantité) confondu avec « peut » (verbe pouvoir, 3e sg.)."),
    (r'\bpeut\b', 'peu',  'homophone_peut_peu',
     "« peut » (verbe pouvoir) confondu avec « peu » (adverbe de quantité)."),
    (r"\bc'est\b","s'est",'homophone_cest_sest',
     "« c'est » (présentatif) confondu avec « s'est » (verbe pronominal)."),
    (r"\bs'est\b","c'est",'homophone_sest_cest',
     "« s'est » (verbe pronominal) confondu avec « c'est » (présentatif)."),
    (r'\bla\b',   'là',   'homophone_la_là',
     "« la » (article/pronom) confondu avec « là » (adverbe de lieu)."),
    (r'\blà\b',   'la',   'homophone_là_la',
     "« là » (adverbe de lieu) confondu avec « la » (article/pronom)."),
]
_HOMO_RE = [(re.compile(pat), repl, tid, expl) for pat, repl, tid, expl in HOMOPHONES]

# ── Acute/grave accent removal ────────────────────────────────────────────────

ACCENT_PAIRS_ACUTE: list[tuple[str, str]] = [
    row for row in taxonomy["accents"]["common_words"]
]
_ACCENT_MAP: list[tuple[re.Pattern, str, str, str]] = []
for correct, wrong in ACCENT_PAIRS_ACUTE:
    pat  = re.compile(r'\b' + re.escape(correct) + r'\b', re.IGNORECASE)
    expl = f"Accent manquant : « {wrong} » devrait s'écrire « {correct} »."
    _ACCENT_MAP.append((pat, wrong, 'accent_manquant', expl))

# ── Circumflex accent removal ──────────────────────────────────────────────────

CIRCUMFLEX_PAIRS: list[tuple[str, str]] = [
    ("trône", "trone"), ("être", "etre"), ("bientôt", "bientot"),
    ("pêche", "peche"), ("pêcheur", "pecheur"), ("pêcheurs", "pecheurs"),
    ("fête", "fete"), ("forêt", "foret"), ("île", "ile"), ("âme", "ame"),
    ("château", "chateau"), ("fenêtre", "fenetre"), ("tête", "tete"),
    ("arrêter", "arreter"), ("arrêta", "arreta"), ("arrête", "arrete"),
    ("paraître", "paraitre"), ("connaître", "connaitre"), ("naître", "naitre"),
    ("côté", "cote"), ("côte", "cote"), ("rôle", "role"), ("dôme", "dome"),
    ("crème", "creme"), ("même", "meme"), ("suprême", "supreme"),
    ("extrême", "extreme"), ("problème", "probleme"), ("système", "systeme"),
    ("poème", "poeme"), ("thème", "theme"), ("schème", "scheme"),
    ("fantôme", "fantome"), ("diplôme", "diplome"), ("symptôme", "symptome"),
    ("hôpital", "hopital"), ("hôtel", "hotel"), ("hôte", "hote"),
    ("côte", "cote"), ("côté", "cote"), ("flûte", "flute"),
    ("août", "aout"), ("goût", "gout"), ("coût", "cout"),
]
_CIRCUM_MAP: list[tuple[re.Pattern, str, str, str]] = []
for correct, wrong in CIRCUMFLEX_PAIRS:
    pat  = re.compile(r'\b' + re.escape(correct) + r'\b', re.IGNORECASE)
    expl = f"Accent circonflexe manquant : « {wrong} » devrait s'écrire « {correct} »."
    _CIRCUM_MAP.append((pat, wrong, 'accent_circonflexe_manquant', expl))

# ── Cedilla removal ────────────────────────────────────────────────────────────

CEDILLA_PAIRS: list[tuple[str, str]] = [
    ("façon", "facon"), ("garçon", "garcon"), ("français", "francais"),
    ("française", "francaise"), ("leçon", "lecon"), ("reçu", "recu"),
    ("aperçut", "apercut"), ("aperçu", "apercu"), ("reçoit", "recoit"),
    ("reçu", "recu"), ("commençait", "commencait"), ("commença", "commenca"),
    ("plaçait", "placait"), ("plaça", "placa"), ("lançait", "lancait"),
    ("ça", "ca"), ("çà", "ca"),
]
_CED_MAP: list[tuple[re.Pattern, str, str, str]] = []
for correct, wrong in CEDILLA_PAIRS:
    pat  = re.compile(r'\b' + re.escape(correct) + r'\b', re.IGNORECASE)
    expl = f"Cédille manquante : « {wrong} » devrait s'écrire « {correct} »."
    _CED_MAP.append((pat, wrong, 'cedille_manquante', expl))

# ── Typography ─────────────────────────────────────────────────────────────────

_TYPO_RULES: list[tuple[re.Pattern, str, str, str]] = [
    (re.compile(r' (\?)'), '?', 'typographie_espace_interrogation',
     "L'espace avant le point d'interrogation est manquante (typographie française)."),
    (re.compile(r' (!)'), '!', 'typographie_espace_exclamation',
     "L'espace avant le point d'exclamation est manquante (typographie française)."),
    (re.compile(r' (;)'), ';', 'typographie_espace_point_virgule',
     "L'espace avant le point-virgule est manquante (typographie française)."),
    (re.compile(r' (:)'), ':', 'typographie_espace_deux_points',
     "L'espace avant les deux-points est manquante (typographie française)."),
]

_CURLY_APOS_RE  = re.compile(r"([a-záàâäéèêëîïôùûüçœæ])'([a-záàâäéèêëîïôùûüçœæ])", re.IGNORECASE)
_GUILLEMET_O_RE = re.compile(r'«[   ]*')
_GUILLEMET_C_RE = re.compile(r'[   ]*»')

# ── Elision removal ────────────────────────────────────────────────────────────

_ELISION_RE = re.compile(
    r"\b(d')([aeiouàâäéèêëîïôùûüœæh]\w+)\b", re.IGNORECASE
)

# ── Merged words ───────────────────────────────────────────────────────────────

_WORD_RE = re.compile(r'\b([a-záàâäéèêëîïôùûüçœæ]{3,})\s+([a-záàâäéèêëîïôùûüçœæ]{3,})\b',
                      re.IGNORECASE)

# ── Split word ─────────────────────────────────────────────────────────────────

_LONG_WORD_RE = re.compile(r'\b([a-záàâäéèêëîïôùûüçœæ]{7,})\b', re.IGNORECASE)

# ── Word order swap ────────────────────────────────────────────────────────────

# Only swap clearly swappable pairs: adverb/adjective around verb/noun
_SWAP_RE = re.compile(
    r'\b([a-záàâäéèêëîïôùûüçœæ]{3,})\s+([a-záàâäéèêëîïôùûüçœæ]{3,})\s+([a-záàâäéèêëîïôùûüçœæ]{3,})\b',
    re.IGNORECASE
)

# ── Double letter ──────────────────────────────────────────────────────────────

_DOUBLE_RE = re.compile(r'\b([a-záàâäéèêëîïôùûüçœæ]{5,})\b', re.IGNORECASE)
DOUBLED_LETTERS = list('bcdflmnprstv')


# ── Corruption functions ───────────────────────────────────────────────────────

def corrupt_homophone(sentence: str):
    rules = list(_HOMO_RE)
    random.shuffle(rules)
    for pat, repl, tid, expl in rules:
        m = pat.search(sentence)
        if not m:
            continue
        corrupted = sentence[:m.start()] + repl + sentence[m.end():]
        start = m.start()
        end   = m.start() + len(repl)
        return corrupted, m.group(0), repl, start, end, tid, expl
    return None


def corrupt_accent(sentence: str):
    rules = list(_ACCENT_MAP)
    random.shuffle(rules)
    for pat, wrong, tid, expl in rules:
        m = pat.search(sentence)
        if not m:
            continue
        original    = m.group(0)
        replacement = wrong.capitalize() if original[0].isupper() else wrong
        corrupted   = sentence[:m.start()] + replacement + sentence[m.end():]
        return corrupted, original, replacement, m.start(), m.start() + len(replacement), tid, expl
    return None


def corrupt_circumflex(sentence: str):
    rules = list(_CIRCUM_MAP)
    random.shuffle(rules)
    for pat, wrong, tid, expl in rules:
        m = pat.search(sentence)
        if not m:
            continue
        original    = m.group(0)
        replacement = wrong.capitalize() if original[0].isupper() else wrong
        corrupted   = sentence[:m.start()] + replacement + sentence[m.end():]
        return corrupted, original, replacement, m.start(), m.start() + len(replacement), tid, expl
    return None


def corrupt_cedilla(sentence: str):
    rules = list(_CED_MAP)
    random.shuffle(rules)
    for pat, wrong, tid, expl in rules:
        m = pat.search(sentence)
        if not m:
            continue
        original    = m.group(0)
        replacement = wrong.capitalize() if original[0].isupper() else wrong
        corrupted   = sentence[:m.start()] + replacement + sentence[m.end():]
        return corrupted, original, replacement, m.start(), m.start() + len(replacement), tid, expl
    return None


def corrupt_typography(sentence: str):
    rules = list(_TYPO_RULES)
    random.shuffle(rules)
    for pat, repl, tid, expl in rules:
        m = pat.search(sentence)
        if not m:
            continue
        corrupted = sentence[:m.start()] + repl + sentence[m.end():]
        return corrupted, repl, ' ' + repl, m.start(), m.start() + len(repl), tid, expl
    return None


def corrupt_apostrophe(sentence: str):
    m = _CURLY_APOS_RE.search(sentence)
    if not m:
        return None
    original    = m.group(0)
    replacement = original.replace('’', "'").replace('ʼ', "'")
    if original == replacement:
        return None
    corrupted = sentence[:m.start()] + replacement + sentence[m.end():]
    return corrupted, replacement, original, m.start(), m.start() + len(replacement), \
           'typographie_apostrophe', "L'apostrophe droite (') devrait être l'apostrophe typographique (')."


def corrupt_guillemets(sentence: str):
    if '«' not in sentence and '»' not in sentence:
        return None
    corrupted = _GUILLEMET_O_RE.sub('"', sentence)
    corrupted = _GUILLEMET_C_RE.sub('"', corrupted)
    if corrupted == sentence:
        return None
    for i, (a, b) in enumerate(zip(sentence, corrupted)):
        if a != b:
            j = i
            while j < min(len(sentence), len(corrupted)) and sentence[j] != corrupted[j]:
                j += 1
            expl = "Les guillemets français « » sont remplacés par des guillemets anglais."
            return corrupted, sentence[i:j] or '«', corrupted[i:j] or '"', i, j, \
                   'typographie_guillemets', expl
    return None


def corrupt_merged_words(sentence: str):
    """Remove the space between two common words: "certes le" → "certesle"."""
    matches = list(_WORD_RE.finditer(sentence))
    if not matches:
        return None
    m = random.choice(matches)
    w1, w2    = m.group(1), m.group(2)
    merged    = w1 + w2
    corrupted = sentence[:m.start()] + merged + sentence[m.end():]
    start     = m.start()
    end       = m.start() + len(merged)
    expl      = f"Deux mots sont collés sans espace : « {merged} » devrait s'écrire « {w1} {w2} »."
    return corrupted, merged, f"{w1} {w2}", start, end, 'mots_fusionnes', expl


def corrupt_split_word(sentence: str):
    """Insert a space inside a long word: "important" → "im portant"."""
    matches = list(_LONG_WORD_RE.finditer(sentence))
    if not matches:
        return None
    m    = random.choice(matches)
    word = m.group(0)
    # Split after 2–4 chars (avoid splitting at position 0 or end)
    split_at = random.randint(2, min(4, len(word) - 3))
    w1, w2   = word[:split_at], word[split_at:]
    split    = w1 + ' ' + w2
    corrupted = sentence[:m.start()] + split + sentence[m.end():]
    start     = m.start()
    end       = m.start() + len(split)
    expl      = f"Un espace intempestif coupe le mot : « {w1} {w2} » devrait s'écrire « {word} »."
    return corrupted, split, word, start, end, 'mot_coupe', expl


def corrupt_word_order(sentence: str):
    """Swap two adjacent content words (min 4 chars each)."""
    # Find 3-word windows: W1 W2 W3 → W1 W3 W2 (swap last two)
    matches = list(_SWAP_RE.finditer(sentence))
    if not matches:
        return None
    m  = random.choice(matches)
    w1 = m.group(1)
    w2 = m.group(2)
    w3 = m.group(3)
    # Skip if any word is very short (function word) or same
    if len(w2) < 4 or len(w3) < 4 or w2.lower() == w3.lower():
        return None
    # Swap w2 and w3
    original_span = m.group(0)
    swapped_span  = w1 + ' ' + w3 + ' ' + w2
    corrupted     = sentence[:m.start()] + swapped_span + sentence[m.end():]
    start         = m.start()
    end           = m.start() + len(swapped_span)
    expl          = f"Ordre des mots incorrect : « {w2} {w3} » → remettre dans l'ordre attendu."
    return corrupted, swapped_span, original_span, start, end, 'ordre_mots', expl


def corrupt_double_letter(sentence: str):
    """Double a consonant in a long word: "radieux" → "raddieux"."""
    matches = list(_DOUBLE_RE.finditer(sentence))
    if not matches:
        return None
    random.shuffle(matches)
    for m in matches:
        word = m.group(0)
        # Find a doublelable consonant in the middle of the word
        candidates = [i for i, c in enumerate(word[1:-1], 1)
                      if c in DOUBLED_LETTERS]
        if not candidates:
            continue
        i         = random.choice(candidates)
        corrupted_word = word[:i] + word[i] + word[i:]  # double the letter
        corrupted = sentence[:m.start()] + corrupted_word + sentence[m.end():]
        start     = m.start()
        end       = m.start() + len(corrupted_word)
        expl      = f"Doublement de lettre incorrect : « {corrupted_word} » devrait s'écrire « {word} »."
        return corrupted, corrupted_word, word, start, end, 'doublon_lettre', expl
    return None


def corrupt_elision(sentence: str):
    """Remove elision: "d'enfants" → "de enfants"."""
    matches = list(_ELISION_RE.finditer(sentence))
    if not matches:
        return None
    m         = random.choice(matches)
    elided    = m.group(0)          # "d'enfants"
    prefix    = 'd’' if '’' in elided else "d'"
    rest      = m.group(2)          # "enfants"
    expanded  = 'de ' + rest
    corrupted = sentence[:m.start()] + expanded + sentence[m.end():]
    start     = m.start()
    end       = m.start() + len(expanded)
    expl      = f"Élision manquante : « de {rest} » devrait s'écrire « d'{rest} »."
    return corrupted, expanded, elided, start, end, 'elision_manquante', expl


CORRUPTION_FUNCS = [
    corrupt_homophone,
    corrupt_accent,
    corrupt_circumflex,
    corrupt_cedilla,
    corrupt_merged_words,
    corrupt_split_word,
    corrupt_typography,
    corrupt_apostrophe,
    corrupt_guillemets,
    corrupt_word_order,
    corrupt_double_letter,
    corrupt_elision,
]

FUNC_WEIGHTS = [
    0.22,  # homophone      — most common
    0.18,  # accent acute   — very common
    0.10,  # circumflex     — common
    0.06,  # cedilla        — moderately common
    0.10,  # merged words   — common (fast typing)
    0.06,  # split word     — less common
    0.08,  # typography     — common
    0.06,  # apostrophe     — common
    0.04,  # guillemets     — occasional
    0.04,  # word order     — occasional
    0.03,  # double letter  — occasional
    0.03,  # elision        — occasional
]


# ── Example builders ──────────────────────────────────────────────────────────

def build_error_example(corrupted: str, original: str, correction: str,
                        start: int, end: int, error_type: str, explanation: str) -> dict:
    wrong_text = corrupted[start:end]
    correct_text = resolve_correction(wrong_text, original, correction)
    if not wrong_text or not correct_text or wrong_text == correct_text:
        raise ValueError(f"Invalid corruption target for {error_type}: {wrong_text!r} -> {correct_text!r}")

    assistant_json = {
        "errors": [{
            "original":    wrong_text,
            "correction":  correct_text,
            "type":        error_type,
            "explanation": explanation
        }]
    }
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Texte: {corrupted}"},
            {"role": "assistant", "content": json.dumps(assistant_json, ensure_ascii=False)}
        ]
    }


def resolve_correction(wrong_text: str, first: str, second: str) -> str:
    """Corruption functions historically returned mixed tuple orders.

    One of `first`/`second` is normally the corrupted span and the other is the
    correction. Derive the correction from the actual corrupted sentence span.
    """
    if first == wrong_text and second != wrong_text:
        return second
    if second == wrong_text and first != wrong_text:
        return first
    if first != wrong_text:
        return first
    return second


def build_clean_example(sentence: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"Texte: {sentence}"},
            {"role": "assistant", "content": '{"errors": []}'}
        ]
    }


# ── False-positive hard negatives ─────────────────────────────────────────────

FALSE_POSITIVES = taxonomy["false_positives"]["literary_fragments"]

DIALOGUE_ORALITY = [
    "— T'as vu ça ?",
    "— J'sais pas, moi.",
    "— C'est quoi ce truc ?",
    "— Faut pas s'inquiéter.",
    "— On s'en fout, non ?",
    "— Il peut pas venir.",
    "— Elle a rien dit.",
    "— On y va ?",
    "— C'est bon, laisse tomber.",
    "— Arrête, tu vois pas ?",
]

ARCHAIC_FORMS = [
    "Il ne savait s'il devait partir ou demeurer.",
    "Elle eût voulu lui dire adieu.",
    "Certes, il n'eût point hésité en d'autres temps.",
    "Ainsi fut-il convenu entre eux.",
    "C'était là, semblait-il, la seule issue.",
    "Il advint que le soir même, tout changea.",
    "Or il se fit que nul ne parla.",
]

INVENTED_WORDS = [
    "Le Maji Leuk Daour s'imposa dans leur dimension.",
    "Les Ufundis, forgerons de l'univers, forgèrent ces lames.",
    "Kaleikaumaka regardait la mer depuis la cabine.",
    "Aduna abritait sept types d'êtres.",
    "Bùɓakar tenait la barre avec assurance.",
    "Les Ifritts et les Ardhi s'étaient longtemps combattus.",
    "Kauma sourit et acquiesça d'un geste.",
]


def hard_negatives() -> list[dict]:
    rows = []
    for s in FALSE_POSITIVES + DIALOGUE_ORALITY + ARCHAIC_FORMS + INVENTED_WORDS:
        rows.append(build_clean_example(s))
    return rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not SENTENCES_PATH.exists():
        print(f"ERROR: {SENTENCES_PATH} not found. Run script 01 first.")
        sys.exit(1)

    sentences = []
    with SENTENCES_PATH.open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            sentences.append(row["sentence"])

    print(f"Loaded {len(sentences):,} clean sentences.")
    print(f"Target: {TARGET:,} training examples ({int(CLEAN_RATIO * 100)}% clean).")
    print(f"Corruption types: {len(CORRUPTION_FUNCS)}")

    n_clean = int(TARGET * CLEAN_RATIO)
    n_error = TARGET - n_clean

    random.shuffle(sentences)
    examples: list[dict] = []

    hn = hard_negatives()
    examples.extend(hn)
    print(f"Added {len(hn)} hard-negative false-positive examples.")

    error_pool  = sentences[:n_error * 3]
    error_count = 0
    type_counts: dict[str, int] = {}
    for s in tqdm(error_pool, desc="Corrupting"):
        if error_count >= n_error:
            break
        func   = random.choices(CORRUPTION_FUNCS, weights=FUNC_WEIGHTS, k=1)[0]
        result = func(s)
        if result is None:
            continue
        corrupted, original, correction, start, end, etype, expl = result
        if corrupted == s:
            continue
        examples.append(build_error_example(corrupted, original, correction, start, end, etype, expl))
        type_counts[etype] = type_counts.get(etype, 0) + 1
        error_count += 1

    clean_pool  = sentences[n_error * 3:n_error * 3 + n_clean * 2]
    clean_count = 0
    for s in tqdm(clean_pool, desc="Clean examples"):
        if clean_count >= n_clean:
            break
        examples.append(build_clean_example(s))
        clean_count += 1

    random.shuffle(examples)

    with OUT_PATH.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\n✓ Done.")
    print(f"  Error examples:  {error_count:,}")
    print(f"  Clean examples:  {clean_count:,}")
    print(f"  Hard negatives:  {len(hn)}")
    print(f"  Total:           {len(examples):,}")
    print(f"\n  Error type distribution:")
    for t, c in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {t:<35} {c:>7,}")
    print(f"\n  Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
