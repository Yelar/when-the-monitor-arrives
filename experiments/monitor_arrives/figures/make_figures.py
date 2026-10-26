"""
Make the report's figures from figures/data.json (built by build_data.py). Each figure is saved as PNG and PDF.

    uv run python experiments/monitor_arrives/figures/make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
NAMES = {"C1": "No monitor", "C2": "No chat", "C3": "Full monitor", "C4": "25% monitor",
         "C5": "Blind-spot monitor", "C6": "Keyword monitor", "C7": "Late monitor"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK_2,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 200,
})


def save(fig, name: str) -> None:
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.png and {name}.pdf")


def hourly_index(runs: list[dict], cond: str, smooth: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Mean collusion index per hour over a condition's seeds (hours without trades skipped) and its standard error
    across seeds, both as a 3-hour moving mean."""
    m = np.array([[np.nan if v is None else v for v in r["trade_index"]] for r in runs if r["condition"] == cond])
    with np.errstate(all="ignore"):
        mean = np.nanmean(m, axis=0)
        se = np.nanstd(m, axis=0, ddof=1) / np.sqrt(np.sum(np.isfinite(m), axis=0))
    k = smooth // 2

    def roll(x):
        return np.array([np.nanmean(x[max(0, i - k): i + k + 1]) for i in range(len(x))])

    return roll(mean), roll(se)


def fig_prices(data: dict) -> None:
    runs = data["informed"]["runs"]
    hours = np.arange(1, 41)
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    series = [("C1", INK, "-", 2.6), ("C2", MUTED, "--", 2.0), ("C3", BLUE, "-", 2.0), ("C7", ORANGE, "-", 2.0)]
    dollars = {}
    for cond, color, style, width in series:
        y, se = hourly_index(runs, cond)
        y, se = 90 + 10 * y, 10 * se  # collusion index to average trade price in dollars
        dollars[cond] = y
        ax.fill_between(hours, y - se, y + se, color=color, alpha=0.10, linewidth=0)
        ax.plot(hours, y, color=color, linestyle=style, linewidth=width, label=NAMES[cond])
        nudge = {"C7": 5, "C3": -5}.get(cond, 0)  # the full and late monitor lines end close together
        ax.annotate(NAMES[cond], (40, y[-1]), xytext=(6, nudge), textcoords="offset points", va="center",
                    color=INK if cond == "C1" else INK_2, fontsize=9, fontweight="bold" if cond == "C1" else "normal")
    # the late monitor has the same rules as no monitor before hour 11, so the gap there is chance
    y_none, y_late = dollars["C1"], dollars["C7"]
    ax.annotate("", xy=(8, y_late[7]), xytext=(8, y_none[7]),
                arrowprops=dict(arrowstyle="<->", color=INK_2, linewidth=1.1, shrinkA=1, shrinkB=1))
    ax.annotate("same rules before hour 11,\nso this gap is chance", xy=(8.15, (y_none[7] + y_late[7]) / 2),
                xytext=(12.5, 83.35), fontsize=8, color=INK_2, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.axvline(10.5, color=MUTED, linestyle=":", linewidth=1.2)
    ax.text(10.8, 89.8, "late monitor\nstarts (hour 11)", color=INK_2, fontsize=8.5, va="top")
    ax.set_yticks(np.arange(83, 90))
    ax.set_yticklabels([f"${v}" for v in range(83, 90)])
    ax.set_ylim(82.5, 90)
    ax.set_xlim(1, 40)
    ax.set_xlabel("Hour of the 40-hour run")
    ax.set_ylabel("Average trade price per lot")
    ax.set_title("Informed setup, mean of 8 seeds per condition (band = 1 standard error), 3-hour moving average",
                 fontsize=10, color=INK_2, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    save(fig, "figA1_prices_by_hour")


def fig_readers(data: dict) -> None:
    order = ["C1", "C3", "C7", "C4", "C5"]  # the keyword monitor (base setup only) is in the appendix
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8), sharey=True, gridspec_kw={"wspace": 0.42})
    # the labeler is the main mark, with a 95% interval over seeds, and the attempts the monitor missed sit just
    # above it; the monitor's flags are hollow, slightly below it, so equal values never cover each other
    readers = [("attempt_rate", "Attempts, by the labeler (reads message + private notes)", BLUE, "o", 0.0, 110, True),
               ("missed_rate", "Missed attempts, as % of all messages", MAGENTA, "X", 0.22, 95, True),
               ("flagged_rate", "Flagged by the monitor (scored after the runs, message only)", AQUA, "s", -0.25, 55, False)]
    for ax, setup in zip(axes, ("base", "informed")):
        conds = data[setup]["claude_labels"]["conditions"]
        for i, cond in enumerate(order):
            c, y = conds[cond], len(order) - 1 - i
            lo, hi = c["attempt_rate_ci"]
            ax.plot([100 * lo, 100 * hi], [y, y], color=BLUE, linewidth=2, alpha=0.45, solid_capstyle="round", zorder=2)
            for k, label, color, marker, dy, size, filled in readers:
                ax.scatter(100 * c[k], y + dy, s=size, marker=marker, zorder=3 if filled else 2, linewidth=1.6,
                           facecolor=color if filled else "white", edgecolor=color if not filled else "white",
                           label=label if i == 0 else None)
        # a column right of each panel: the missed attempts as a share of that condition's attempts (pooled over seeds)
        pooled = data.get("robustness", {}).get(setup, {}).get("miss_rate", {})
        trans = matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData)
        # the header sits inside the panel's height, below the panel title, so the two never collide
        ax.text(1.03, len(order) - 0.6, "Missed, as % of\nattempts (8 runs\npooled)", transform=trans, fontsize=7.5,
                color=INK_2, va="bottom", ha="left", linespacing=1.1)
        for i, cond in enumerate(order):
            if cond in pooled:
                n, d = pooled[cond]["pooled"]
                ax.text(1.03, len(order) - 1 - i, f"{100 * n / d:.0f}%", transform=trans, fontsize=8.5, color=INK_2,
                        va="center", ha="left")
        ax.set_title(f"{setup.capitalize()} setup", fontsize=10, color=INK_2, loc="left")
        ax.set_xlim(-3, 103)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
        ax.set_xlabel("Share of all messages, hours 11 to 40")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(range(len(order)))
    axes[0].set_yticklabels([NAMES[c] for c in reversed(order)])
    for tick in axes[0].get_yticklabels():
        if tick.get_text() in ("No monitor", "Full monitor", "Late monitor"):  # the baseline and the main comparison
            tick.set_fontweight("bold")
            tick.set_color(INK)
    axes[0].set_ylim(-0.6, len(order) + 0.3)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2, frameon=False, fontsize=8.5,
                   handletextpad=0.3, columnspacing=1.2)
    save(fig, "fig1_labels_judge_monitor")


def fig_hourly(data: dict) -> None:
    """Attempts and missed attempts by hour, late monitor against no monitor, one panel per setup."""
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
    series = [("C1", "attempt", "No monitor, attempts", INK, "-"), ("C1", "missed", "No monitor, missed attempts", INK, ":"),
              ("C7", "attempt", "Late monitor, attempts", ORANGE, "-"), ("C7", "missed", "Late monitor, missed attempts", ORANGE, ":")]
    hours = np.arange(1, 41)
    for ax, setup in zip(axes, ("base", "informed")):
        hourly = data["robustness"][setup]["hourly"]
        for cond, k, label, color, style in series:
            m = np.array([np.nan if v[0] is None else 100 * v[0] for v in hourly[cond][k]])
            se = 1.96 * np.array([100 * v[1] for v in hourly[cond][k]])  # 95% interval, as in Figure 1
            kernel = np.ones(3) / 3  # 3-hour moving average, ends left unsmoothed
            sm = m.copy()
            sm[1:-1] = np.convolve(m, kernel, mode="valid")
            ax.plot(hours, sm, color=color, linestyle=style, linewidth=2.2 if style == "-" else 2.0, label=label)
            if k == "attempt":
                ax.fill_between(hours, np.clip(sm - se, 0, 100), np.clip(sm + se, 0, 100), color=color,
                                alpha=0.08 if color == INK else 0.15, linewidth=0)  # a black band at 0.15 hides the orange one
        ax.grid(axis="x", visible=False)
        ax.axvline(10.5, color=INK_2, linestyle="--", linewidth=1)
        ax.text(11.2, 97, "late monitor\nstarts reading", fontsize=8, color=INK_2, va="top")
        ax.set_title(f"{setup.capitalize()} setup", fontsize=10, color=INK_2, loc="left")
        ax.set_xlim(1, 40)
        ax.set_xticks([1, 10, 20, 30, 40])
        ax.set_xlabel("Hour of the 40-hour run")
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    axes[0].set_ylabel("Share of all messages")
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.08), ncol=2, frameon=False, fontsize=8.5,
                   handletextpad=0.4, columnspacing=1.2)
    save(fig, "fig2_attempts_by_hour")


def fig_disclaimers(data: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), sharey=True)
    colors = {"C3": BLUE, "C7": ORANGE, "C4": AQUA, "C5": YELLOW, "C6": INK_2}  # pink is reserved for missed attempts (Figure 1)
    for ax, setup in zip(axes, ("base", "informed")):
        disc = data[setup]["discovery"]
        for cond, color in colors.items():
            if cond not in disc:
                continue
            h = np.array(disc[cond]["hours"])
            v = np.array([np.nan if x is None else 100 * x for x in disc[cond]["disclaimer_rate"]])
            ax.plot(h, v, color=color, linewidth=2, label=NAMES[cond])
        ax.axvline(0, color=MUTED, linestyle=":", linewidth=1.2)
        ax.set_title(f"{setup.capitalize()} setup", fontsize=10, color=INK_2, loc="left")
        ax.set_xlabel("Hours from the seller's first fine (0 = the fine)")
        ax.set_xlim(-8, 15)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    axes[0].set_ylabel("Messages stating independence")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8.5)
    save(fig, "figA2_independence_after_first_fine")


def main() -> None:
    data = json.loads((HERE / "data.json").read_text())
    fig_prices(data)
    fig_readers(data)
    if "robustness" in data:
        fig_hourly(data)
    fig_disclaimers(data)


if __name__ == "__main__":
    main()
