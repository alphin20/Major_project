
import pandas as pd

import json

df = pd.read_csv("data/medquad_1.csv")

data = []

for _, row in df.iterrows():
    data.append({
        "instruction": row["question"],
        "input": "",
        "output": row["answer"]
    })

with open("data/crafted_instruction_data_alpaca.json", "w") as f:
    json.dump(data, f, indent=2)

print("Converted", len(data), "samples")