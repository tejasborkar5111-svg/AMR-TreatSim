"""
data_loader.py

Loads literature_dataset.csv, validates it, and prints a summary so the
team can quickly check progress on data extraction (Steps 1-2 of the
project report).

Usage:
    python3 data_loader.py
"""

import csv
from collections import defaultdict

REQUIRED_COLUMNS = [
    "citation", "technology", "target_type", "target_name", "water_matrix",
    "removal_log", "removal_percent",
]

PLACEHOLDER_MARKER = "SAMPLE - replace with real paper"


def load_dataset(path: str = "literature_dataset.csv"):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    missing_cols = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
    if missing_cols:
        raise ValueError(f"Dataset is missing required columns: {missing_cols}")

    return rows


def summarize(rows):
    by_tech = defaultdict(int)
    placeholder_count = 0

    for row in rows:
        by_tech[row["technology"]] += 1
        if row["citation"].strip() == PLACEHOLDER_MARKER:
            placeholder_count += 1

    print(f"Total rows: {len(rows)}")
    print(f"Placeholder rows still needing a real citation: {placeholder_count}")
    print("\nRows per technology:")
    for tech, count in sorted(by_tech.items()):
        print(f"  {tech}: {count}")

    if placeholder_count == len(rows):
        print("\n>>> All rows are still placeholders. Start Step 1 (paper collection) "
              "and replace these with your team's extracted data.")
    elif placeholder_count > 0:
        print(f"\n>>> {placeholder_count} placeholder row(s) remain - "
              "replace with real literature data before building models on this.")
    else:
        print("\n>>> Dataset looks fully populated with real data. Good to proceed.")


if __name__ == "__main__":
    rows = load_dataset()
    summarize(rows)
