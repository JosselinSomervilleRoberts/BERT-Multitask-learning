# Opens plots/plot_results.csv in a pandas dataframe

import pandas as pd
import matplotlib.pyplot as plt

files = ["plots/finetune_pal_1.csv", "plots/finetune-combine-encourage-new-1.csv", "plots/finetune-combine-encourage-smart-1.csv"]
names = ["Pal Schedule", "Gradient compromise", "Gradient compromise + SMART"]

# Create a figure with two subplots
fig, axs = plt.subplots(1, 1, figsize=(6, 4))

# Plot the loss per input
axs.set_xlabel('Number of inputs')
axs.set_ylabel('Mean dev accuracy')
for file, name in zip(files, names):
    df = pd.read_csv(file)
    axs.plot(df["Step"][:15], df["Value"][:15], label=name)
axs.legend()

# # Plot the loss for wall time
# axs[1].set_xlabel('Time (s)')
# axs[1].set_ylabel('Mean dev accuracy')
# for file, name in zip(files, names):
#     df = pd.read_csv(file)
#     axs[1].plot(df["Wall time"] - df["Wall time"][0], df["Value"], label=name)
# axs[1].legend()

# plt.subplots_adjust(hspace=0.3, top=0.95, bottom=0.075)
plt.show()