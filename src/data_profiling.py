"""
Step 1: Data Profiling Script
Reads all CSV files from the dataset directory, profiles them, and outputs key metrics.
"""
import pandas as pd
import os
import glob
import json
import sys

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

def load_all_csvs(data_dir):
    """Load all CSV files and concatenate them."""
    csv_files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    print(f"Found {len(csv_files)} CSV files in {data_dir}\n")
    
    dfs = []
    for f in csv_files:
        fname = os.path.basename(f)
        try:
            df = pd.read_csv(f, sep=";", quotechar='"', encoding="utf-8", on_bad_lines="warn")
            df["_source_file"] = fname
            print(f"  {fname}: {df.shape[0]} rows, {df.shape[1]} cols")
            dfs.append(df)
        except Exception as e:
            print(f"  ERROR reading {fname}: {e}")
    
    combined = pd.concat(dfs, ignore_index=True)
    print(f"\nTotal rows loaded: {combined.shape[0]}")
    print(f"Total columns: {combined.shape[1]}")
    return combined


def profile_dataset(df):
    """Compute and print profiling metrics."""
    print("\n" + "="*60)
    print("DATA PROFILING REPORT")
    print("="*60)
    
    # 1. Basic stats
    print(f"\n1. BASIC STATISTICS")
    print(f"   Total rows: {len(df)}")
    print(f"   Columns: {list(df.columns)}")
    
    # 2. Key columns check
    key_cols = ['title', 'body', 'language', 'entities', 'original_id', 'source_name', 'date_published']
    print(f"\n2. KEY COLUMNS PRESENCE")
    for col in key_cols:
        present = col in df.columns
        print(f"   {col}: {'PRESENT' if present else 'MISSING'}")
    
    # 3. Duplicates
    print(f"\n3. DUPLICATE ANALYSIS")
    # Exact duplicates by original_id
    n_unique_ids = df['original_id'].nunique()
    n_total = len(df)
    n_dup_ids = n_total - n_unique_ids
    print(f"   Unique original_id values: {n_unique_ids}")
    print(f"   Duplicate rows (same original_id): {n_dup_ids}")
    
    # Near-duplicates (same title+body)
    df['_title_body'] = df['title'].fillna('') + '|||' + df['body'].fillna('')
    n_unique_tb = df['_title_body'].nunique()
    n_near_dup = n_total - n_unique_tb
    print(f"   Unique title+body combinations: {n_unique_tb}")
    print(f"   Near-duplicate rows (same title+body): {n_near_dup}")
    
    # Show most duplicated original_ids
    dup_counts = df['original_id'].value_counts()
    multi_dup = dup_counts[dup_counts > 1]
    if len(multi_dup) > 0:
        print(f"   original_ids appearing >1 time: {len(multi_dup)}")
        print(f"   Top 5 duplicated IDs:")
        for oid, cnt in multi_dup.head(5).items():
            print(f"     {oid[:40]}...: {cnt} times")
    
    # 4. Missing values
    print(f"\n4. MISSING VALUES (title & body)")
    title_null = df['title'].isna().sum()
    title_empty = (df['title'].fillna('').str.strip() == '').sum()
    body_null = df['body'].isna().sum()
    body_empty = (df['body'].fillna('').str.strip() == '').sum()
    print(f"   title - null: {title_null}, empty/whitespace: {title_empty}")
    print(f"   body  - null: {body_null}, empty/whitespace: {body_empty}")
    
    # 5. Language distribution
    print(f"\n5. LANGUAGE DISTRIBUTION")
    lang_dist = df['language'].value_counts(dropna=False)
    for lang, cnt in lang_dist.items():
        pct = cnt / len(df) * 100
        print(f"   {lang}: {cnt} ({pct:.1f}%)")
    
    # 6. Entities column analysis
    print(f"\n6. ENTITIES COLUMN ANALYSIS")
    ent_null = df['entities'].isna().sum()
    ent_notnull = len(df) - ent_null
    print(f"   Null/NaN: {ent_null}")
    print(f"   Non-null: {ent_notnull}")
    
    if ent_notnull > 0:
        # Check format of non-null entities
        ent_samples = df[df['entities'].notna()]['entities'].head(5)
        print(f"   Sample values (first 5 non-null):")
        for i, val in enumerate(ent_samples):
            val_str = str(val)[:200]
            print(f"     [{i}]: {val_str}")
        
        # Try to parse as JSON if string
        ent_str = df[df['entities'].notna()]['entities'].astype(str)
        json_parseable = 0
        has_spans = 0
        for val in ent_str:
            try:
                parsed = json.loads(val)
                json_parseable += 1
                if isinstance(parsed, list) and len(parsed) > 0:
                    first = parsed[0]
                    if isinstance(first, dict) and ('start' in first or 'offset' in first):
                        has_spans += 1
            except (json.JSONDecodeError, TypeError):
                pass
        print(f"   JSON-parseable entities: {json_parseable}")
        print(f"   Entities with span offsets: {has_spans}")
    
    # 7. Source name distribution
    print(f"\n7. SOURCE NAME DISTRIBUTION (top 10)")
    src_dist = df['source_name'].value_counts().head(10)
    for src, cnt in src_dist.items():
        print(f"   {src}: {cnt}")
    
    # 8. Date range
    print(f"\n8. DATE RANGE")
    dates = pd.to_datetime(df['date_published'], errors='coerce')
    print(f"   Earliest: {dates.min()}")
    print(f"   Latest: {dates.max()}")
    print(f"   Unparseable dates: {dates.isna().sum()}")
    
    # 9. Text length stats
    print(f"\n9. TEXT LENGTH STATISTICS")
    df['_title_len'] = df['title'].fillna('').str.len()
    df['_body_len'] = df['body'].fillna('').str.len()
    print(f"   Title: min={df['_title_len'].min()}, max={df['_title_len'].max()}, "
          f"mean={df['_title_len'].mean():.0f}, median={df['_title_len'].median():.0f}")
    print(f"   Body:  min={df['_body_len'].min()}, max={df['_body_len'].max()}, "
          f"mean={df['_body_len'].mean():.0f}, median={df['_body_len'].median():.0f}")
    
    # Cleanup temp columns
    df.drop(columns=['_title_body', '_title_len', '_body_len'], inplace=True, errors='ignore')
    
    return df


if __name__ == "__main__":
    df = load_all_csvs(DATA_DIR)
    df = profile_dataset(df)
    
    # Save profiling results summary as JSON for README
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                               "data", "profiling_results.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    results = {
        "total_rows": len(df),
        "unique_original_ids": int(df['original_id'].nunique()),
        "duplicate_by_id": int(len(df) - df['original_id'].nunique()),
        "title_null": int(df['title'].isna().sum()),
        "title_empty": int((df['title'].fillna('').str.strip() == '').sum()),
        "body_null": int(df['body'].isna().sum()),
        "body_empty": int((df['body'].fillna('').str.strip() == '').sum()),
        "language_distribution": df['language'].value_counts(dropna=False).to_dict(),
        "entities_null": int(df['entities'].isna().sum()),
        "entities_notnull": int(df['entities'].notna().sum()),
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\nProfiling results saved to {output_path}")
