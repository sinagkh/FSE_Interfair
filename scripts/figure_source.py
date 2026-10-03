"""Generate the paper figures from supplied measurements."""
from pathlib import Path
import csv
import json
import math
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"

INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8f8d86"
GRID = "#e4e3df"
UP = "#eb6834"      # significant increase in the rule residual (warm pole)
DOWN = "#2a78d6"    # significant decrease (cool pole)
WHITE_C = "#2a78d6"
BLACK_C = "#eb6834"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": INK2,
    "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

METHOD_NAME = {"ltdd": "LTDD", "cot_phi": "CoT-Phi", "dralign": "DRAlign", "neufair": "NeuFair",
               "reweighing": "Reweighing", "hifi_shared": "HIFI"}
MARKER = {"LTDD": "o", "CoT-Phi": "s", "DRAlign": "^", "NeuFair": "D", "Reweighing": "v", "HIFI": "P"}
SETTING = {"hmda_oh": "HMDA-OH", "hmda_md": "HMDA-MD", "hmda_va": "HMDA-VA", "hmda_pa": "HMDA-PA",
           "credit_broad": "Credit", "acs_income": "ACSIncome"}


METHOD_NAME.update({"fairsmote": "Fair-SMOTE"})
MARKER = {"LTDD": "o", "CoT-Phi": "s", "DRAlign": "^", "Fair-SMOTE": "D", "Reweighing": "v", "HIFI": "P"}


def logit(p):
    return math.log(p / (1 - p))


def fig_example():
    ex = json.loads((OUT / "response_example.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.05), sharey=True)
    xs = [0, 1]
    for ax, key, title in ((axes[0], "erm_p", "Before repair (ERM)"), (axes[1], "is_p", "After repair (InterFair)")):
        w0, w1, b0, b1 = [100 * v for v in ex[key]]
        ew = logit(ex[key][1]) - logit(ex[key][0])
        eb = logit(ex[key][3]) - logit(ex[key][2])
        ax.plot(xs, [w0, w1], color=WHITE_C, lw=2, marker="o", ms=5, zorder=3)
        ax.plot(xs, [b0, b1], color=BLACK_C, lw=2, marker="s", ms=5, zorder=3)
        up_first = b0 > w0  # after repair the Black-coded line sits slightly higher
        for x, w, b in ((0, w0, b0), (1, w1, b1)):
            ha, dx = ("right", -6) if x == 0 else ("left", 6)
            close = abs(w - b) < 1.0
            ax.annotate(f"{w:.2f}", (x, w), xytext=(dx, -5 if (close and up_first) else (5 if close else 0)),
                        textcoords="offset points", ha=ha, va="center", fontsize=6.8, color=INK)
            ax.annotate(f"{b:.2f}", (x, b), xytext=(dx, 5 if (close and up_first) else (-5 if close else 0)),
                        textcoords="offset points", ha=ha, va="center", fontsize=6.8, color=INK)
        ax.text(0.02, 0.98, "log-odds change\nWhite-coded " + f"{ew:+.3f}".replace("-", "\u2212") + "\nBlack-coded " + f"{eb:+.3f}".replace("-", "\u2212"),
                transform=ax.transAxes, ha="left", va="top", fontsize=6.6, color=INK2, linespacing=1.25)
        ax.set_title(f"{title}   $|R|={abs(eb - ew):.3f}$", fontsize=7.6, color=INK, loc="left")
        ax.set_xticks(xs)
        ax.set_xticklabels(["income \\$24k\nDTI 45%", "income \\$34k\nDTI code 33"], fontsize=6.8)
        ax.set_xlim(-0.45, 1.45)
        ax.grid(True, axis="y", color=GRID, lw=0.5, zorder=0)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("Approval probability (%)")
    axes[0].set_ylim(82.5, 92.5)
    handles = [Line2D([], [], color=WHITE_C, lw=2, marker="o", ms=4, label="White-coded"),
               Line2D([], [], color=BLACK_C, lw=2, marker="s", ms=4, label="Black-coded")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT / "example.pdf")
    fig.savefig(OUT / "example.png", dpi=200)
    plt.close(fig)
    return {k: ex[k] for k in ("erm_residual", "is_residual")}


def fig_maintenance():
    path = ROOT / "results/studies/requirements/maintenance_per_seed.csv"
    rows = [r for r in csv.DictReader(open(path)) if r["task"] == "hmda_oh" and r["architecture"] == "mlp"]
    arms = ["before", "update", "task_control", "rerepair"]
    for a in arms:
        assert sorted(int(r["seed"]) for r in rows if r["arm"] == a) == list(range(1000, 1010)), a
    specs = [("L_R", 1, "Interaction residual $\\bar R$", ".3f"),
             ("decision_disagreement", 100, "Decision disagreement $D$ (%)", ".2f"),
             ("direction_adverse_005", 100, "Wrong direction $W$ (%)", ".2f")]
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 1.95))
    summary = {}
    for ax, (metric, scale, title, form) in zip(axes, specs):
        mean, sd = [], []
        for a in arms:
            v = [float(r[metric]) * scale for r in rows if r["arm"] == a]
            mean.append(st.mean(v)); sd.append(st.stdev(v))
        summary[metric] = dict(zip(arms, [round(m, 4) for m in mean]))
        ax.plot([0, 1], mean[:2], color=MUTED, lw=2, zorder=2)
        ax.errorbar([0, 1], mean[:2], yerr=sd[:2], fmt="o", color=MUTED, markerfacecolor="white", ms=4.5,
                    capsize=2.2, lw=1, zorder=3)
        for idx, col, mk in ((2, UP, "s"), (3, DOWN, "D")):
            ax.plot([1, 2], [mean[1], mean[idx]], color=col, lw=2, zorder=2)
            ax.errorbar(2, mean[idx], yerr=sd[idx], fmt=mk, color=col, markerfacecolor="white", ms=4.5,
                        capsize=2.2, lw=1, zorder=3)
        for idx, xx in ((0, 0), (1, 1), (2, 2), (3, 2)):
            ax.annotate(format(mean[idx], form), (xx, mean[idx] + sd[idx]), xytext=(-9 if idx == 3 else 0, 3),
                        textcoords="offset points", ha="center", va="bottom", fontsize=6.6, color=INK)
        ax.set_ylim(0, max(m + d for m, d in zip(mean, sd)) * 1.3)
        ax.set_xlim(-0.3, 2.4)
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["repaired", "after task-\nonly update", "follow-up"], fontsize=6.6)
        ax.set_title(title, fontsize=7.4, loc="left", color=INK)
        ax.grid(True, axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    handles = [Line2D([], [], color=UP, marker="s", markerfacecolor="white", lw=2, ms=4, label="task-only follow-up"),
               Line2D([], [], color=DOWN, marker="D", markerfacecolor="white", lw=2, ms=4, label="InterFair follow-up")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(OUT / "maintenance.pdf"); fig.savefig(OUT / "maintenance.png", dpi=200)
    plt.close(fig)
    return summary


FEATURE_NAME = {
    "loan_type": "Loan type", "occupancy_type": "Occupancy", "preapproval": "Preapproval",
    "loan_to_value_ratio": "Loan-to-value", "debt_to_income_ratio": "Debt-to-income", "loan_amount": "Loan amount",
    "property_value": "Property value", "income": "Income", "loan_purpose": "Loan purpose", "lien_status": "Lien status",
    "ffiec_msa_md_median_family_income": "Area median income", "tract_to_msa_income_percentage": "Tract income ratio",
    "tract_minority_population_percent": "Tract minority share",
    "education": "Education", "hours": "Weekly hours", "household": "Household role", "occupation": "Occupation",
    "work_class": "Class of worker",
}


def fig_heatmap():
    """Per-feature residuals, methods as rows and features as columns, both tasks in one row."""
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.gridspec import GridSpec
    path = ROOT / "results/main/primary_per_feature.csv"
    rows = list(csv.DictReader(open(path)))
    ramp = ["#f4f8fe", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
    cmap = LinearSegmentedColormap.from_list("seq_blue", ramp)
    panels = [("hmda_oh", "HMDA Ohio (race)", ["erm", "ltdd", "cot_phi", "dralign", "fairsmote", "reweighing", "hifi_shared", "soft"]),
              ("acs_income", "ACSIncome (sex)", ["erm", "ltdd", "cot_phi", "dralign", "fairsmote", "reweighing", "hifi_shared", "soft"])]
    label = {"erm": "ERM", "ltdd": "LTDD", "cot_phi": "CoT-Phi", "dralign": "DRAlign", "neufair": "NeuFair",
             "hifi_shared": "HIFI", "soft": "InterFair", "fairsmote": "Fair-SMOTE", "reweighing": "Reweighing"}
    fig = plt.figure(figsize=(5.6, 2.85))
    gs = GridSpec(1, 5, figure=fig, width_ratios=[13, 0.35, 3.6, 5, 0.35], wspace=0.08)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 3])]
    caxes = [fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 4])]
    summary = {}
    for ax, cax, (task, title, arms) in zip(axes, caxes, panels):
        vals = {}
        for r in rows:
            if r["task"] == task and r["architecture"] == "mlp" and r["arm"] in arms:
                vals.setdefault((r["arm"], r["feature"]), []).append(float(r["L_R"]))
        feats = sorted({f for a, f in vals}, key=lambda f: -st.mean(vals[("erm", f)]))
        for a in arms:
            for f in feats:
                assert len(vals[(a, f)]) == 10, (task, a, f)
        M = np.array([[st.mean(vals[(a, f)]) for f in feats] for a in arms])
        summary[task] = {FEATURE_NAME[f]: {label[a]: round(float(M[i, j]), 3) for i, a in enumerate(arms)} for j, f in enumerate(feats)}
        im = ax.imshow(M, cmap=cmap, vmin=0, vmax=M.max(), aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                v = M[i, j]
                ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=5.9,
                        color="white" if v > 0.55 * M.max() else INK)
        ax.set_yticks(range(len(arms))); ax.set_yticklabels([label[a] for a in arms], fontsize=6.8)
        ax.set_xticks(range(len(feats))); ax.set_xticklabels([FEATURE_NAME[f] for f in feats], fontsize=6.3, rotation=40, ha="right")
        ax.set_title(title, fontsize=7.6, loc="left", color=INK)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, cax=cax)
        cb.ax.tick_params(labelsize=5.8, length=2)
        cb.outline.set_visible(False)
    axes[1].tick_params(labelleft=True)
    fig.subplots_adjust(left=0.125, right=0.955, top=0.92, bottom=0.3)
    fig.savefig(OUT / "heatmap.pdf"); fig.savefig(OUT / "heatmap.png", dpi=200)
    plt.close(fig)
    (OUT / "heatmap.json").write_text(json.dumps(summary, indent=1))
    return summary


if __name__ == "__main__":
    print(fig_example())
    print(fig_maintenance())
    fig_heatmap()
