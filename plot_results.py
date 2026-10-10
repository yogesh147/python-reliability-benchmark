from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

results = Path("results")
summary = pd.read_csv(results / "model_summary.csv")

sns.set_theme(style="whitegrid")

chart = sns.catplot(
    data=summary,
    x="mean_pass_rate",
    y="model",
    hue="category",
    kind="bar",
    height=5,
    aspect=1.5,
)

chart.set_axis_labels("Mean test pass rate", "AI model")
chart.set(xlim=(0, 1))
chart.fig.suptitle(
    "Python Reliability Benchmark",
    y=1.03,
)
chart.savefig(
    results / "category_comparison.png",
    dpi=200,
    bbox_inches="tight",
    )
plt.close("all")

print("Saved results/category_comparison.png")