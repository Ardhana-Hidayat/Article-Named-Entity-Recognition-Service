"""
Step 2: Annotation Pipeline, Evaluation Examples, and Data Split (70/15/15)
- Weak-supervision pre-annotation using spaCy
- Exact character span alignment (start, end)
- Deterministic 70/15/15 split by original_id (SEED=42)
- Saves JSONL splits and evaluation_examples.json
"""
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pandas as pd
import spacy

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_OUT = PROJECT_ROOT / "data"
SEED = 42

SUPPORTED_LABELS = {"PERSON": "PERSON", "ORG": "ORG", "GPE": "GPE", "LOC": "GPE", "DATE": "DATE", "TIME": "DATE", "EVENT": "EVENT"}


def get_data_dir() -> Path:
    if "NER_DATA_DIR" in os.environ and os.path.exists(os.environ["NER_DATA_DIR"]):
        return Path(os.environ["NER_DATA_DIR"])
    for cand in [PROJECT_ROOT / "use_cases" / "pnm_study_case" / "sample", PROJECT_ROOT / "dataset"]:
        if cand.exists() and list(cand.glob("*.csv")):
            return cand
    return PROJECT_ROOT / "dataset"


def load_and_deduplicate(data_dir: Path) -> pd.DataFrame:
    """Load CSVs and perform strict article-level deduplication."""
    files = sorted(data_dir.glob("*.csv"))
    dfs = [pd.read_csv(f, sep=";", quotechar='"', encoding="utf-8", on_bad_lines="warn") for f in files]
    df = pd.concat(dfs, ignore_index=True)
    df = df.drop_duplicates(subset=["original_id"], keep="first")
    df["_hash"] = df.apply(lambda r: hashlib.md5((str(r["title"]).strip() + str(r["body"]).strip()).encode()).hexdigest(), axis=1)
    df = df.drop_duplicates(subset=["_hash"], keep="first").reset_index(drop=True)
    return df


def extract_spans(nlp, text: str) -> list[dict]:
    """Extract and validate exact character spans for supported labels."""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        mapped = SUPPORTED_LABELS.get(ent.label_)
        if mapped and text[ent.start_char:ent.end_char] == ent.text:
            entities.append({"text": ent.text, "label": mapped, "start": ent.start_char, "end": ent.end_char})
    return entities


def run_pipeline():
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    data_dir = get_data_dir()
    df = load_and_deduplicate(data_dir)
    print(f"Unique articles after deduplication: {len(df)}")

    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 200000

    # Annotate articles
    records = []
    for idx, row in df.iterrows():
        title = str(row["title"]).strip()
        body = str(row["body"]).strip()
        combined = f"{title}\n\n{body}"
        entities = extract_spans(nlp, combined)
        records.append({
            "doc_id": idx,
            "original_id": row["original_id"],
            "title": title,
            "body": body,
            "combined_text": combined,
            "language": row.get("language", "en"),
            "entities": entities,
        })

    # Deterministic 70/15/15 split
    np.random.seed(SEED)
    unique_ids = df["original_id"].unique()
    np.random.shuffle(unique_ids)

    n = len(unique_ids)
    train_ids = set(unique_ids[:int(n * 0.70)])
    val_ids = set(unique_ids[int(n * 0.70):int(n * 0.85)])

    for r in records:
        oid = r["original_id"]
        r["split"] = "train" if oid in train_ids else ("val" if oid in val_ids else "test")

    # Save JSONL splits
    splits = {"train": [], "val": [], "test": []}
    for r in records:
        splits[r["split"]].append(r)

    for name, items in splits.items():
        path = DATA_OUT / f"annotations_{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {name} split: {len(items)} docs, {sum(len(x['entities']) for x in items)} spans -> {path}")

    # Save metadata
    meta = {
        "seed": SEED,
        "total_unique_articles": len(records),
        "train_docs": len(splits["train"]),
        "val_docs": len(splits["val"]),
        "test_docs": len(splits["test"]),
        "train_spans": sum(len(x["entities"]) for x in splits["train"]),
        "val_spans": sum(len(x["entities"]) for x in splits["val"]),
        "test_spans": sum(len(x["entities"]) for x in splits["test"]),
    }
    (DATA_OUT / "split_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved split metadata -> {DATA_OUT / 'split_metadata.json'}")


if __name__ == "__main__":
    run_pipeline()
