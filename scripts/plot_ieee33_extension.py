from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = Path(os.environ.get("PEDF_FIGURE_OUTPUT", ROOT / "artifacts/ieee33dc_figures"))
PEDF8 = Path(os.environ.get("PEDF8_ORACLE_PATH", ROOT / "artifacts/analysis/campaign_20260802_internal_revision/REV-B4/results.csv"))
PEDF33 = Path(os.environ.get("IEEE33_RESULTS_PATH", ROOT / "artifacts/ieee33dc_main/daily_metrics.csv"))
COLORS = {"eight": "#466A8F", "thirty_three": "#C06C4E", "b1": "#8A929B", "p": "#176B6B", "red": "#A23E48"}
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8.2,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.6,
    "ytick.labelsize": 7.6,
    "legend.frameon": False,
    "axes.linewidth": 0.8,
    "pdf.fonttype": 42,
})


def scatter_box(ax, groups, labels, colors):
    box = ax.boxplot(groups, patch_artist=True, widths=0.55, showfliers=False, medianprops={"color": "#263746", "linewidth": 1.5})
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.22)
        patch.set_edgecolor(color)
    rng = np.random.default_rng(20260802)
    for index, (values, color) in enumerate(zip(groups, colors), 1):
        ax.scatter(rng.normal(index, 0.045, len(values)), values, s=13, color=color, alpha=0.68, edgecolor="none")
    ax.set_xticks(range(1, len(labels) + 1), labels)


def panel(ax, title):
    ax.set_title(title, loc="left", fontsize=9.6, pad=5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.yaxis.grid(True, color="#E8EBEE", linewidth=0.7, linestyle="--")
    ax.set_axisbelow(True)


def metrics(path: Path, topology: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    if topology == "eight":
        data = data.rename(columns={"delta": "cost_budget"})
        names = {"B0": "B0_ECON", "B1": "B1_BULK_ACI", "B3": "B3_NODAL_ACI_MEM", "P": "P_RESPONSE_MCI", "B4": "B4_DIRECT_SOURCE"}
        data["method"] = data["method"].map(names)
    return data[data["method"].notna()].copy()


def derived(data: pd.DataFrame, topology: str) -> pd.DataFrame:
    pivot = data.pivot(index="date", columns=["cost_budget", "method"], values="source_emissions_kg")
    rows = []
    for date in pivot.index:
        b0_2 = pivot.loc[date, (0.02, "B0_ECON")]
        b4_2 = pivot.loc[date, (0.02, "B4_DIRECT_SOURCE")]
        b1 = pivot.loc[date, (0.02, "B1_BULK_ACI")] if (0.02, "B1_BULK_ACI") in pivot else np.nan
        rows.append({
            "topology": topology,
            "date": date,
            "p_vs_b3_2pct": 100 * (pivot.loc[date, (0.02, "B3_NODAL_ACI_MEM")] - pivot.loc[date, (0.02, "P_RESPONSE_MCI")]) / b0_2,
            "b1_regret_2pct": 100 * (b1 - b4_2) / b0_2,
            "p_regret_2pct": 100 * (pivot.loc[date, (0.02, "P_RESPONSE_MCI")] - b4_2) / b0_2,
            "p_change_10pct": 100 * (pivot.loc[date, (0.10, "P_RESPONSE_MCI")] - pivot.loc[date, (0.10, "B0_ECON")]) / pivot.loc[date, (0.10, "B0_ECON")],
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    eight = derived(metrics(PEDF8, "eight"), "8 node")
    thirty_three = derived(metrics(PEDF33, "thirty_three"), "33 node")
    combined = pd.concat([eight, thirty_three], ignore_index=True)
    combined.to_csv(OUT / "Figure_6_source.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.0))

    ax = axes[0, 0]
    x = np.arange(1, len(eight) + 1)
    ax.plot(x, eight["p_vs_b3_2pct"], marker="o", ms=3, lw=1.2, color=COLORS["eight"], label="8 node")
    ax.plot(x, thirty_three["p_vs_b3_2pct"], marker="s", ms=3, lw=1.2, color=COLORS["thirty_three"], label="33 node")
    ax.axhline(0, color="#666D74", lw=0.8)
    ax.set_xlabel("Chronological day")
    ax.set_ylabel("P correction over B3 (pp)")
    ax.legend(loc="upper right", fontsize=7.3)
    panel(ax, "a  Local correction changes with topology")

    ax = axes[0, 1]
    groups = [eight["p_vs_b3_2pct"].to_numpy(), thirty_three["p_vs_b3_2pct"].to_numpy()]
    scatter_box(ax, groups, ["8 node", "33 node"], [COLORS["eight"], COLORS["thirty_three"]])
    ax.axhline(0, color="#666D74", lw=0.8)
    ax.set_ylabel("P correction over B3 (pp)")
    panel(ax, "b  Positive medians retain negative days")

    ax = axes[1, 0]
    groups = [eight["b1_regret_2pct"].dropna().to_numpy(), eight["p_regret_2pct"].to_numpy(), thirty_three["b1_regret_2pct"].dropna().to_numpy(), thirty_three["p_regret_2pct"].to_numpy()]
    scatter_box(ax, groups, ["8 B1", "8 P", "33 B1", "33 P"], [COLORS["b1"], COLORS["p"], COLORS["b1"], COLORS["p"]])
    ax.axhline(0, color=COLORS["red"], lw=0.9, ls="--")
    ax.set_ylabel("Regret to B4 at 2% (pp)")
    panel(ax, "c  B4 exposes residual surrogate regret")

    ax = axes[1, 1]
    groups = [eight["p_change_10pct"].to_numpy(), thirty_three["p_change_10pct"].to_numpy()]
    scatter_box(ax, groups, ["8 node", "33 node"], [COLORS["eight"], COLORS["thirty_three"]])
    ax.axhline(0, color=COLORS["red"], lw=0.9, ls="--")
    ax.set_ylabel("P emission change from B0 at 10% (%)")
    panel(ax, "d  Large actions retain daily rebounds")

    fig.suptitle("Topology transfer narrows the interpretation of a fixed point marginal signal", x=0.07, y=0.995, ha="left", fontsize=10.8, fontweight="bold", color="#263746")
    fig.subplots_adjust(left=0.12, right=0.98, bottom=0.11, top=0.89, hspace=0.46, wspace=0.43)
    fig.savefig(OUT / "Figure_6.png", dpi=600, facecolor="white")
    fig.savefig(OUT / "Figure_6.tiff", dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUT / "Figure_6.pdf", facecolor="white")
    with Image.open(OUT / "Figure_6.png") as image:
        image.convert("RGB").save(OUT / "Figure_6.png", dpi=(600, 600))
    with Image.open(OUT / "Figure_6.tiff") as image:
        image.convert("RGB").save(OUT / "Figure_6.tiff", dpi=(600, 600), compression="tiff_lzw")
    diagnostics = combined.groupby("topology").agg(
        days=("date", "count"),
        median_p_vs_b3_2pct=("p_vs_b3_2pct", "median"),
        positive_p_vs_b3_days=("p_vs_b3_2pct", lambda values: int((values > 0).sum())),
        median_b1_regret_2pct=("b1_regret_2pct", "median"),
        median_p_regret_2pct=("p_regret_2pct", "median"),
        median_p_change_10pct=("p_change_10pct", "median"),
        p_rebound_days_10pct=("p_change_10pct", lambda values: int((values > 0).sum())),
    ).reset_index().to_dict("records")
    (OUT / "Figure_6_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    plt.close(fig)


if __name__ == "__main__":
    main()
