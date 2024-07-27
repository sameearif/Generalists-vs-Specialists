import krippendorff
import pandas as pd
import numpy as np

class KrippendorffAlpha():
    def __init__(self, task, to_eval):
        self.task = task
        self.to_eval = to_eval

    def calculate_alpha(self):
        df_h1 = pd.read_excel("human-evaluation/human/annotator-1.xlsx", self.task)
        df_h2 = pd.read_excel("human-evaluation/human/annotator-2.xlsx", self.task)
        a1 = df_h1[["Rank M1", "Rank M2", "Rank M3", "Rank M4"]].to_numpy()
        a2 = df_h2[["Rank M1", "Rank M2", "Rank M3", "Rank M4"]].to_numpy()
        x = [np.array(a1).flatten(), np.array(a2).flatten()]

        df_gpt = None
        df_llama = None

        if self.to_eval == "gpt" or self.to_eval == "all":
            df_gpt = pd.read_csv(f"human-evaluation/gpt/ranks/{self.task}.csv")
            a3 = df_gpt[["M1", "M2", "M3", "M4"]].to_numpy()
            x.append(np.array(a3).flatten())

        if self.to_eval == "llama" or self.to_eval == "all":
            df_llama = pd.read_csv(f"human-evaluation/llama/ranks/{self.task}.csv")
            a4 = df_llama[["M1", "M2", "M3", "M4"]].to_numpy()
            x.append(np.array(a4).flatten())

        if self.to_eval == "claude" or self.to_eval == "all":
            df_llama = pd.read_csv(f"human-evaluation/claude/ranks/{self.task}.csv")
            a4 = df_llama[["M1", "M2", "M3", "M4"]].to_numpy()
            x.append(np.array(a4).flatten())

        return krippendorff.alpha(reliability_data=x, level_of_measurement='ordinal')

k = KrippendorffAlpha("ai-assistant", "claude")
print(k.calculate_alpha())
