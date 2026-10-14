"""
Step 2: Annotation Pipeline, Evaluation Examples, and Data Split
- Uses spaCy pretrained model as weak supervision for pre-annotation
- Creates span-level annotations in JSONL format
- Generates evaluation_examples.json with 10+ curated examples
- Splits data by original_id (70/15/15) with SEED=42
"""
import pandas as pd
import os
import sys
import glob
import json
import hashlib
import re
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEED = 42

def get_data_dir():
    if "NER_DATA_DIR" in os.environ and os.path.exists(os.environ["NER_DATA_DIR"]):
        return os.environ["NER_DATA_DIR"]
    candidates = [
        os.path.join(PROJECT_ROOT, "use_cases", "pnm_study_case", "sample"),
        os.path.abspath(os.path.join(PROJECT_ROOT, "..", "use_cases", "pnm_study_case", "sample")),
        os.path.join(PROJECT_ROOT, "dataset"),
    ]
    for c in candidates:
        if os.path.exists(c) and glob.glob(os.path.join(c, "*.csv")):
            return c
    return os.path.join(PROJECT_ROOT, "dataset")

DATA_DIR = get_data_dir()
DATA_OUT = os.path.join(PROJECT_ROOT, "data")
os.makedirs(DATA_OUT, exist_ok=True)

# ============================================================
# Label Schema Definition
# ============================================================
LABEL_SCHEMA = {
    "PERSON": "Named individual (e.g., 'Dato Khairussaleh', 'Elon Musk')",
    "ORG": "Organization, company, institution (e.g., 'Maybank', 'CIMB Group')",
    "GPE": "Geopolitical entity: country, state, city (e.g., 'Malaysia', 'Kuala Lumpur')",
    "DATE": "Dates, time references (e.g., 'September 2026', 'Q3 2026')",
    "EVENT": "Named events (e.g., 'Hari Raya', 'Budget 2027')",
}

# Map from spaCy labels to our schema
SPACY_LABEL_MAP = {
    "PERSON": "PERSON",
    "ORG": "ORG",
    "GPE": "GPE",
    "LOC": "GPE",  # map LOC to GPE
    "DATE": "DATE",
    "TIME": "DATE",
    "EVENT": "EVENT",
    "FAC": None,     # skip facilities
    "NORP": None,    # skip nationalities
    "PRODUCT": None,
    "WORK_OF_ART": None,
    "LAW": None,
    "LANGUAGE": None,
    "MONEY": None,
    "QUANTITY": None,
    "ORDINAL": None,
    "CARDINAL": None,
    "PERCENT": None,
}


def load_and_dedup(data_dir):
    """Load all CSVs, deduplicate by original_id keeping first occurrence."""
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    dfs = []
    for f in csv_files:
        df = pd.read_csv(f, sep=";", quotechar='"', encoding="utf-8", on_bad_lines="warn")
        df["_source_file"] = os.path.basename(f)
        dfs.append(df)
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"Total rows before dedup: {len(combined)}")
    
    # Remove exact duplicates by original_id (keep first occurrence)
    deduped = combined.drop_duplicates(subset=['original_id'], keep='first').reset_index(drop=True)
    print(f"Total rows after dedup by original_id: {len(deduped)}")
    
    # Further remove near-duplicates (same title+body but different original_id)
    deduped['_text_hash'] = deduped.apply(
        lambda r: hashlib.md5((str(r['title']).strip() + str(r['body']).strip()).encode()).hexdigest(), 
        axis=1
    )
    before_near = len(deduped)
    deduped = deduped.drop_duplicates(subset=['_text_hash'], keep='first').reset_index(drop=True)
    print(f"Near-duplicates removed: {before_near - len(deduped)}")
    print(f"Final unique articles: {len(deduped)}")
    
    return deduped


def create_combined_text(title, body):
    """Combine title and body with consistent separator."""
    return f"{str(title).strip()}\n\n{str(body).strip()}"


def annotate_with_spacy(nlp, text, label_map):
    """Use spaCy NER as pre-annotation, mapping labels to our schema."""
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        mapped_label = label_map.get(ent.label_)
        if mapped_label is not None:
            entities.append({
                "text": ent.text,
                "label": mapped_label,
                "start": ent.start_char,
                "end": ent.end_char,
            })
    return entities


def validate_spans(entities, combined_text):
    """Validate that all span offsets are correct."""
    valid = []
    for ent in entities:
        actual = combined_text[ent["start"]:ent["end"]]
        if actual == ent["text"]:
            valid.append(ent)
        else:
            # Try to fix by finding the text
            idx = combined_text.find(ent["text"], max(0, ent["start"] - 5))
            if idx >= 0:
                ent["start"] = idx
                ent["end"] = idx + len(ent["text"])
                valid.append(ent)
    return valid


def split_data(df, seed=42):
    """Split by original_id into 70/15/15 train/val/test."""
    import numpy as np
    np.random.seed(seed)
    
    unique_ids = df['original_id'].unique()
    np.random.shuffle(unique_ids)
    
    n = len(unique_ids)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    train_ids = set(unique_ids[:train_end])
    val_ids = set(unique_ids[train_end:val_end])
    test_ids = set(unique_ids[val_end:])
    
    df['split'] = df['original_id'].apply(
        lambda x: 'train' if x in train_ids else ('val' if x in val_ids else 'test')
    )
    
    return df, train_ids, val_ids, test_ids


def main():
    print("Loading and deduplicating data...")
    df = load_and_dedup(DATA_DIR)
    
    # Filter to English articles for better NER quality with en_core_web_sm
    # But keep all for annotation (we'll note language)
    print(f"\nLanguage distribution after dedup:")
    print(df['language'].value_counts())
    
    print("\nLoading spaCy model for pre-annotation...")
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading en_core_web_sm...")
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    
    # Increase max length for long articles
    nlp.max_length = 200000
    
    print("Annotating articles with spaCy pre-annotation...")
    annotations = []
    for idx, row in df.iterrows():
        combined = create_combined_text(row['title'], row['body'])
        entities = annotate_with_spacy(nlp, combined, SPACY_LABEL_MAP)
        entities = validate_spans(entities, combined)
        
        record = {
            "doc_id": idx,
            "original_id": row['original_id'],
            "title": str(row['title']).strip(),
            "body": str(row['body']).strip(),
            "combined_text": combined,
            "language": row['language'],
            "source_name": row['source_name'],
            "entities": entities,
            "num_entities": len(entities),
        }
        annotations.append(record)
        
        if (idx + 1) % 100 == 0:
            print(f"  Annotated {idx + 1}/{len(df)} articles...")
    
    print(f"Annotation complete. {len(annotations)} articles annotated.")
    
    # Entity stats
    all_ents = [e for a in annotations for e in a['entities']]
    label_counts = Counter(e['label'] for e in all_ents)
    print(f"\nTotal entity spans: {len(all_ents)}")
    print("Label distribution:")
    for label, cnt in label_counts.most_common():
        print(f"  {label}: {cnt}")
    
    # Split data
    print("\nSplitting data (70/15/15, SEED=42)...")
    df_ann = pd.DataFrame(annotations)
    df_ann, train_ids, val_ids, test_ids = split_data(df_ann, SEED)
    
    train_data = df_ann[df_ann['split'] == 'train']
    val_data = df_ann[df_ann['split'] == 'val']
    test_data = df_ann[df_ann['split'] == 'test']
    
    print(f"\nData split summary:")
    print(f"  Train: {len(train_data)} docs, "
          f"{sum(len(a) for a in train_data['entities'])} spans")
    print(f"  Val:   {len(val_data)} docs, "
          f"{sum(len(a) for a in val_data['entities'])} spans")
    print(f"  Test:  {len(test_data)} docs, "
          f"{sum(len(a) for a in test_data['entities'])} spans")
    
    # Per-split label distribution
    for split_name, split_df in [('Train', train_data), ('Val', val_data), ('Test', test_data)]:
        ents = [e for row_ents in split_df['entities'] for e in row_ents]
        lc = Counter(e['label'] for e in ents)
        print(f"  {split_name} label distribution: {dict(lc)}")
    
    # Save annotations as JSONL
    for split_name, split_df in [('train', train_data), ('val', val_data), ('test', test_data)]:
        out_path = os.path.join(DATA_OUT, f"annotations_{split_name}.jsonl")
        with open(out_path, 'w', encoding='utf-8') as f:
            for _, row in split_df.iterrows():
                record = {
                    "doc_id": int(row['doc_id']),
                    "original_id": row['original_id'],
                    "title": row['title'],
                    "body": row['body'],
                    "combined_text": row['combined_text'],
                    "language": row['language'],
                    "entities": row['entities'],
                    "split": split_name,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Saved {split_name} annotations to {out_path}")
    
    # ============================================================
    # Create evaluation_examples.json (10+ curated examples)
    # ============================================================
    print("\nCreating evaluation_examples.json...")
    
    # Select diverse examples from across the dataset
    eval_examples = []
    
    # Strategy: pick articles with different characteristics
    # 1. Articles with PERSON entities
    # 2. Articles with ORG entities 
    # 3. Articles with GPE entities
    # 4. Articles with DATE entities
    # 5. Articles with multiple entity types
    # 6. Articles with repeated entities
    # 7. Articles with ambiguous names
    # 8. Articles with punctuation in names
    # 9. Short article with no entities (if exists)
    # 10. Challenging/long article
    
    # Helper: create short excerpts for eval examples
    def make_eval_example(row, max_body_len=500, note=""):
        title = str(row['title']).strip()
        body = str(row['body']).strip()
        if len(body) > max_body_len:
            # Take first max_body_len chars, trying to end at sentence
            cut = body[:max_body_len]
            last_period = cut.rfind('.')
            if last_period > max_body_len // 2:
                body = cut[:last_period + 1]
            else:
                body = cut
        
        combined = create_combined_text(title, body)
        # Re-annotate the truncated text
        entities = annotate_with_spacy(nlp, combined, SPACY_LABEL_MAP)
        entities = validate_spans(entities, combined)
        
        return {
            "title": title,
            "body": body,
            "expected_entities": entities,
            "note": note,
            "original_id": row['original_id'],
            "language": row['language'],
        }
    
    # Find diverse articles
    sorted_by_ents = sorted(annotations, key=lambda x: x['num_entities'], reverse=True)
    
    # Example 1: Article with many PERSON entities
    for ann in annotations:
        persons = [e for e in ann['entities'] if e['label'] == 'PERSON']
        if len(persons) >= 2 and ann['language'] == "'en":
            eval_examples.append(make_eval_example(
                df.iloc[ann['doc_id']], 
                note="Multiple PERSON entities"
            ))
            break
    
    # Example 2: Article with ORG entities  
    for ann in annotations:
        orgs = [e for e in ann['entities'] if e['label'] == 'ORG']
        if len(orgs) >= 3 and ann['language'] == "'en":
            eval_examples.append(make_eval_example(
                df.iloc[ann['doc_id']],
                note="Multiple ORG entities"
            ))
            break
    
    # Example 3: Article with GPE entities
    for ann in annotations:
        gpes = [e for e in ann['entities'] if e['label'] == 'GPE']
        if len(gpes) >= 2 and ann['language'] == "'en":
            eval_examples.append(make_eval_example(
                df.iloc[ann['doc_id']],
                note="Multiple GPE entities"
            ))
            break
    
    # Example 4: Article with DATE entities
    for ann in annotations:
        dates = [e for e in ann['entities'] if e['label'] == 'DATE']
        if len(dates) >= 2 and ann['language'] == "'en":
            eval_examples.append(make_eval_example(
                df.iloc[ann['doc_id']],
                note="Multiple DATE entities"
            ))
            break
    
    # Example 5: Article with mixed entity types
    for ann in annotations:
        labels = set(e['label'] for e in ann['entities'])
        if len(labels) >= 4 and ann['language'] == "'en":
            eval_examples.append(make_eval_example(
                df.iloc[ann['doc_id']],
                note="Mixed entity types (PERSON, ORG, GPE, DATE)"
            ))
            break
    
    # Example 6: Article with repeated entity (same text, different positions)
    for ann in annotations:
        texts = [e['text'] for e in ann['entities']]
        text_counts = Counter(texts)
        repeated = {t: c for t, c in text_counts.items() if c > 1}
        if repeated and ann['language'] == "'en":
            eval_examples.append(make_eval_example(
                df.iloc[ann['doc_id']],
                note=f"Repeated entities: {list(repeated.keys())[:3]}"
            ))
            break
    
    # Example 7: Article with ambiguous names / punctuation
    for ann in annotations:
        for e in ann['entities']:
            if any(c in e['text'] for c in ["'", "-", ".", ","]) and e['label'] == 'PERSON':
                eval_examples.append(make_eval_example(
                    df.iloc[ann['doc_id']],
                    note="Entity with punctuation in name"
                ))
                break
        if len(eval_examples) >= 7:
            break
    
    # Example 8: Very short article
    short_anns = sorted(annotations, key=lambda x: len(x['combined_text']))
    for ann in short_anns[:10]:
        if ann['language'] == "'en":
            eval_examples.append(make_eval_example(
                df.iloc[ann['doc_id']], max_body_len=1000,
                note="Short article"
            ))
            break
    
    # Example 9: Article with no/few entities (synthetic or real)
    zero_ent = [a for a in annotations if a['num_entities'] == 0]
    if zero_ent:
        eval_examples.append(make_eval_example(
            df.iloc[zero_ent[0]['doc_id']], max_body_len=300,
            note="Article with no named entities"
        ))
    else:
        # Create a synthetic no-entity example
        eval_examples.append({
            "title": "Market Analysis Overview",
            "body": "The stock market showed mixed signals today with various sectors performing differently. Some gained while others lost ground in volatile trading.",
            "expected_entities": [],
            "note": "Article with no named entities (synthetic)",
            "original_id": "synthetic_no_entity",
            "language": "en",
        })
    
    # Example 10: Challenging Malay-language article
    for ann in annotations:
        if ann['language'] == "'ms" and ann['num_entities'] >= 2:
            eval_examples.append(make_eval_example(
                df.iloc[ann['doc_id']],
                note="Challenging: Malay-language article"
            ))
            break
    
    # Example 11: Article with many entities (challenging for model)
    if len(sorted_by_ents) > 0:
        for ann in sorted_by_ents:
            if ann['language'] == "'en" and not any(
                ex['original_id'] == ann['original_id'] for ex in eval_examples
            ):
                eval_examples.append(make_eval_example(
                    df.iloc[ann['doc_id']], max_body_len=600,
                    note="Challenging: Dense article with many entities"
                ))
                break
    
    # Example 12: Non-ASCII text
    for ann in annotations:
        if ann['language'] == "'zh" and ann['num_entities'] >= 1:
            eval_examples.append(make_eval_example(
                df.iloc[ann['doc_id']], max_body_len=400,
                note="Non-ASCII: Chinese-language article"
            ))
            break
    
    # Make sure we have at least 10
    print(f"Created {len(eval_examples)} evaluation examples")
    
    eval_path = os.path.join(DATA_OUT, "evaluation_examples.json")
    with open(eval_path, 'w', encoding='utf-8') as f:
        json.dump(eval_examples, f, indent=2, ensure_ascii=False)
    print(f"Saved evaluation examples to {eval_path}")
    
    # Save split metadata
    split_meta = {
        "seed": SEED,
        "total_unique_articles": len(df),
        "train_docs": len(train_data),
        "val_docs": len(val_data),
        "test_docs": len(test_data),
        "train_spans": int(sum(len(a) for a in train_data['entities'])),
        "val_spans": int(sum(len(a) for a in val_data['entities'])),
        "test_spans": int(sum(len(a) for a in test_data['entities'])),
        "label_schema": LABEL_SCHEMA,
    }
    meta_path = os.path.join(DATA_OUT, "split_metadata.json")
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(split_meta, f, indent=2, ensure_ascii=False)
    print(f"Saved split metadata to {meta_path}")


if __name__ == "__main__":
    main()
