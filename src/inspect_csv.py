"""Quick CSV inspection script to understand dataset structure."""
import pandas as pd
import os
import glob

DATA_DIR = r"d:\TUGAS SEMESTER 5\DATA SCIENCE\UTS-Article-Named-Entity-Recognition-Service\dataset"

csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
print(f"Found {len(csv_files)} CSV files\n")

for f in csv_files:
    fname = os.path.basename(f)
    try:
        df = pd.read_csv(f, sep=";", quotechar='"', encoding="utf-8")
        print(f"=== {fname} ===")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print()
    except Exception as e:
        print(f"  ERROR reading {fname}: {e}")
        print()

# Load the smallest file for detailed inspection
smallest = r"d:\TUGAS SEMESTER 5\DATA SCIENCE\UTS-Article-Named-Entity-Recognition-Service\dataset\Maybank_Malaysia_OnlineMedia_07-Sep-2026_07-Oct-2026_4sxH8Dz0hr.csv"
df = pd.read_csv(smallest, sep=";", quotechar='"', encoding="utf-8")
print("=== DETAILED INSPECTION (smallest file) ===")
print(f"Columns: {list(df.columns)}")
print(f"Shape: {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nFirst row sample:")
for col in df.columns:
    val = df[col].iloc[0] if len(df) > 0 else "N/A"
    val_str = str(val)[:200]
    print(f"  {col}: {val_str}")

print(f"\nNull counts:\n{df.isnull().sum()}")

# Check entities column format
if 'entities' in df.columns:
    print(f"\nEntities sample (first 3):")
    for i, val in enumerate(df['entities'].head(3)):
        print(f"  [{i}]: {str(val)[:300]}")
