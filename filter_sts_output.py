# This file reads predictions/sts-test-output.csv which contains two columns id and score
# and rounds up the score to the closest .2 and writes the result to predictions/sts-test-output-round.csv

import pandas as pd

df = pd.read_csv("predictions/sts-test-output.csv", sep=" , ")
print(df.head())

# 1. First clip between 0 and 5
df["Predicted_Similiary"] = df["Predicted_Similiary"].clip(0, 5)

# 2. Then round to the closest .05
df["Predicted_Similiary"] =  (2 * df["Predicted_Similiary"]).round(1) / 2.

# 3. Save to file
df.to_csv("predictions/sts-test-output.csv", index=False)

# 4. Fix the header
with open("predictions/sts-test-output.csv", "r") as f:
    lines = f.readlines()
with open("predictions/sts-test-output.csv", "w") as f:
    f.write(lines[0].replace(",", "   "))
    for line in lines[1:]:
        f.write(line.replace(",", " , "))
