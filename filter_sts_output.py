# This file reads predictions/sts-test-output.csv which contains two columns id and score
# and rounds up the score to the closest .2 and writes the result to predictions/sts-test-output-round.csv

import pandas as pd

# arguments
from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument("--input", type=str, default="predictions/sts-dev-output.csv")
args = parser.parse_args()

df = pd.read_csv(args.input, sep=" , ")
print(df.head())

# 1. First clip between 0 and 5
df["Predicted_Similiary"] = df["Predicted_Similiary"].clip(0, 5)

# 2. Then round to the closest .05
df["Predicted_Similiary"] =  (2 * df["Predicted_Similiary"]).round(1) / 2.

# 3. Save to file
df.to_csv(args.input, index=False)

# 4. Fix the header
with open(args.input, "r") as f:
    lines = f.readlines()
with open(args.input, "w") as f:
    f.write(lines[0].replace(",", "   "))
    for line in lines[1:]:
        f.write(line.replace(",", " , "))
