"""
create_twitter_eval.py
----------------------
Run this once to create data/twitter_eval.jsonl from data/labeled_data.csv.
This creates a balanced evaluation set (50 hateful, 50 safe) from the Twitter dataset.
This file is NEVER ingested into ChromaDB.

How to run:
    python create_twitter_eval.py
"""

import pandas as pd
import json

df = pd.read_csv('data/labeled_data.csv')

# Class 0 = hate speech (hateful), Class 2 = neither (safe)
hate = df[df['class'] == 0].sample(n=50, random_state=42)
safe = df[df['class'] == 2].sample(n=50, random_state=42)

eval_df = pd.concat([hate, safe]).sample(frac=1, random_state=42)

with open('data/twitter_eval.jsonl', 'w') as f:
    for _, row in eval_df.iterrows():
        entry = {
            'id'    : int(row['Unnamed: 0']),
            'text'  : str(row['tweet']),
            'label' : 1 if row['class'] == 0 else 0
        }
        f.write(json.dumps(entry) + '\n')

print(f"Created data/twitter_eval.jsonl")
print(f"Total entries : {len(eval_df)}")
print(f"Hateful (1)   : {len(hate)}")
print(f"Safe    (0)   : {len(safe)}")
