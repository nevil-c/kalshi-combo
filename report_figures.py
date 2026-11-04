"""Draw the README report's charts into docs/figures, one light and one dark copy of each.

Prices and quotes come from the Nov 2 scan. Race outcomes come from Kalshi, with
data/rfq/assumed_ak_ms.csv filling in any race that hasn't settled.

    python report_figures.py
    python report_figures.py --png-dir some/folder   # PNG copies, for checking
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rfq_scan import Leg, engine_price
from score import combo_outcome, fetch_results, read_results, score

ROOT = Path(__file__).parent
SCAN = ROOT / "data" / "rfq" / "quotes_20261102T225120Z.csv"
ASSUMED = ROOT / "data" / "rfq" / "assumed_ak_ms.csv"
OUT = ROOT / "docs" / "figures"
OPEN_RACES = ("SENATEAK-26", "SENATEMS-26")
FOUR_STATES = {"MI", "TX", "OHS", "ME"}
MODEL_RHO = 0.38
TRADERS_RHO = 0.9

# senate_rho.py on Nov 2, before the race markets settled: P(Democrats control) by rho.
CONTROL_RHO = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
CONTROL_NOT_COUNTED = (0.8766, 0.8128, 0.7764, 0.7528, 0.7364, 0.7242, 0.7145, 0.7064, 0.6994, 0.6944, 0.6939)
CONTROL_COUNTED = (0.9154, 0.8433, 0.7996, 0.7703, 0.7492, 0.7330, 0.7201, 0.7094, 0.7005, 0.6944, 0.6939)
CONTROL_BID, CONTROL_ASK = 0.74, 0.75
IMPLIED_NOT_COUNTED, IMPLIED_COUNTED = 0.344, 0.424

THEMES = {
    "light": {"model": "#2a78d6", "traders": "#eb6834", "base": "#898781", "ink": "#0b0b0b", "ink2": "#52514e",
              "grid": "#e1e0d9", "axis": "#c3c2b7", "band": "#f0efec", "ring": "#ffffff"},
    "dark": {"model": "#3987e5", "traders": "#d95926", "base": "#898781", "ink": "#f0efec", "ink2": "#c3c2b7",
             "grid": "#2c2c2a", "axis": "#383835", "band": "#2c2c2a", "ring": "#0d1117"},
}


def _axes(c: dict, size: tuple[float, float], title: str, note: str):
    fig, ax = plt.subplots(figsize=size)
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(c["axis"])
    ax.tick_params(colors=c["ink2"], length=0, labelsize=10)
    ax.grid(True, color=c["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    fig.text(0.012, 0.975, title, color=c["ink"], fontsize=13.5, fontweight="semibold", va="top", ha="left")
    fig.text(0.012, 0.975 - 0.27 / size[1], note, color=c["ink2"], fontsize=10.5, va="top", ha="left")
    return fig, ax


def _top(fig, inches: float, size_h: float, **kw) -> None:
    fig.subplots_adjust(top=1 - inches / size_h, **kw)


def _dots(ax, c: dict, xs, y, kinds, size: float = 80) -> None:
    style = {"unrelated": dict(marker="o", facecolors=c["ring"], edgecolors=c["base"], linewidths=1.8, zorder=3),
             "traders": dict(marker="D", color=c["traders"], edgecolors=c["ring"], linewidths=1, zorder=4),
             "model": dict(marker="o", color=c["model"], edgecolors=c["ring"], linewidths=1, zorder=5)}
    for x, k in zip(xs, kinds):
        ax.scatter([x], [y], s=size * (0.8 if k == "traders" else 1), **style[k])


def _legend(ax, c: dict, kinds) -> None:
    label = {"model": "Our model", "traders": "Kalshi's traders", "unrelated": "Races treated as unrelated"}
    style = {"unrelated": dict(marker="o", markerfacecolor=c["ring"], markeredgecolor=c["base"], markeredgewidth=1.8),
             "traders": dict(marker="D", color=c["traders"], markersize=7),
             "model": dict(marker="o", color=c["model"])}
    handles = [plt.Line2D([], [], linestyle="none", label=label[k], **{"markersize": 8, **style[k]}) for k in kinds]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=len(kinds), frameon=False,
              fontsize=10, labelcolor=c["ink2"], handletextpad=0.2, columnspacing=1.2, borderaxespad=0.2, borderpad=0)


def accuracy(c: dict, rows: list[dict]):
    h = 4.5
    fig, ax = _axes(c, (9, h), "Model and traders were almost level in the most likely case",
                    "Forecast error on 195 combos, by how the two open races finish. The axis is zoomed in.")
    _top(fig, 1.05, h, left=0.37, right=0.87, bottom=0.14)
    kinds = ("unrelated", "traders", "model")
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        for x, k, dodge in zip((r["brier_independent"], r["brier_maker_mid"], r["brier_engine"]), kinds, (0, 0.13, -0.13)):
            _dots(ax, c, (x,), y + dodge, (k,))
        ax.text(1.02, y, f"{(r['brier_engine'] / r['brier_maker_mid'] - 1) * 100:.1f}% worse", transform=ax.get_yaxis_transform(),
                va="center", color=c["ink"] if i == 0 else c["ink2"], fontweight="semibold" if i == 0 else "normal", fontsize=10)
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows][::-1], fontsize=10.5)
    ax.get_yticklabels()[-1].set_fontweight("semibold")
    for t in ax.get_yticklabels():
        t.set_color(c["ink"])
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_xlim(0.15, 0.195)
    ax.set_xticks(np.arange(0.15, 0.1951, 0.01))
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
    ax.grid(False, axis="y")
    ax.set_xlabel("Brier score (lower is better)", color=c["ink2"], fontsize=10.5)
    _legend(ax, c, ("model", "traders", "unrelated"))
    return fig


def breakdown(c: dict, rows: list[dict]):
    h = 3.5
    fig, ax = _axes(c, (9, h), "The traders' lead comes from four states",
                    "Forecast error by group of combos, assuming Democrats win Alaska and Mississippi.")
    _top(fig, 1.05, h, left=0.37, right=0.97, bottom=0.18)
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        ax.plot([r["model"], r["traders"]], [y, y], color=c["axis"], linewidth=2, zorder=2)
        _dots(ax, c, (r["traders"], r["model"]), y, ("traders", "model"))
        lo, hi = sorted((r["model"], r["traders"]))
        ax.text(lo - 0.005, y, f"{lo:.3f}", ha="right", va="center", color=c["ink2"], fontsize=10)
        ax.text(hi + 0.005, y, f"{hi:.3f}", ha="left", va="center", color=c["ink2"], fontsize=10)
    ax.set_yticks(range(len(rows)), [r["label"] for r in rows][::-1], fontsize=10.5)
    for t in ax.get_yticklabels():
        t.set_color(c["ink"])
    ax.set_ylim(-0.5, len(rows) - 0.5)
    ax.set_xlim(0.03, 0.19)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.2f}")
    ax.grid(False, axis="y")
    ax.set_xlabel("Brier score (lower is better)", color=c["ink2"], fontsize=10.5)
    _legend(ax, c, ("model", "traders"))
    return fig


def prices(c: dict, t: pd.DataFrame):
    h = 6.6
    gap, corr = 100 * (t.engine - t.maker_mid).abs().mean(), np.corrcoef(t.engine, t.maker_mid)[0, 1]
    fig, ax = _axes(c, (6.9, h), "Model and trader prices mostly agreed",
                    f"195 combos. Average gap {gap:.1f}¢, correlation {corr:.2f}.")
    _top(fig, 1.05, h, left=0.1, right=0.97, bottom=0.095)
    ax.spines["left"].set_visible(True)
    ax.spines["left"].set_color(c["axis"])
    ax.plot([0, 1], [0, 1], color=c["base"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    four = t.leg_markets.map(lambda s: {x.split("-")[0].replace("SENATE", "") for x in s.split("|")} <= FOUR_STATES)
    ax.scatter(t.engine[~four], t.maker_mid[~four], s=46, color=c["model"], alpha=0.7, edgecolors=c["ring"], linewidths=0.7, zorder=3)
    ax.scatter(t.engine[four], t.maker_mid[four], s=62, marker="D", color=c["traders"], edgecolors=c["ring"], linewidths=0.8, zorder=4)
    ax.text(0.17, 0.9, "Traders priced higher", color=c["ink2"], fontsize=10.5, ha="center")
    ax.text(0.83, 0.1, "Model priced higher", color=c["ink2"], fontsize=10.5, ha="center")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal", adjustable="box")
    pct = lambda v, _: f"{v * 100:.0f}¢"
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("Model price", color=c["ink2"], fontsize=10.5)
    ax.set_ylabel("Traders' midpoint", color=c["ink2"], fontsize=10.5)
    handles = [plt.Line2D([], [], linestyle="none", marker="D", color=c["traders"], markersize=7, label="Both legs in MI, TX, OH or ME"),
               plt.Line2D([], [], linestyle="none", marker="o", color=c["model"], markersize=7, label="All other combos")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=10,
              labelcolor=c["ink2"], handletextpad=0.2, columnspacing=1.2, borderaxespad=0.2, borderpad=0)
    return fig


def control(c: dict):
    h = 4.6
    fig, ax = _axes(c, (9, h), "Kalshi's Senate control price implied a correlation near 0.38",
                    "Chance of Democratic control the model gives at each correlation, against Kalshi's price on Nov 2.")
    _top(fig, 1.05, h, left=0.08, right=0.97, bottom=0.15)
    ax.axhspan(CONTROL_BID, CONTROL_ASK, color=c["band"], zorder=1)
    ax.axhline((CONTROL_BID + CONTROL_ASK) / 2, color=c["base"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.text(0.95, CONTROL_ASK + 0.004, "Kalshi's price, 74–75¢", ha="right", va="bottom", color=c["ink"], fontsize=10.5)
    mid = (CONTROL_BID + CONTROL_ASK) / 2
    ax.plot(CONTROL_RHO, CONTROL_NOT_COUNTED, color=c["model"], linewidth=2, solid_capstyle="round", zorder=3)
    ax.plot(CONTROL_RHO, CONTROL_COUNTED, color=c["traders"], linewidth=2, solid_capstyle="round", zorder=3)
    ax.scatter([IMPLIED_NOT_COUNTED], [mid], s=70, color=c["model"], edgecolors=c["ring"], zorder=5)
    ax.scatter([IMPLIED_COUNTED], [mid], s=62, marker="D", color=c["traders"], edgecolors=c["ring"], zorder=5)
    ax.annotate(f"{IMPLIED_NOT_COUNTED:.2f}", (IMPLIED_NOT_COUNTED, mid), xytext=(-2, -19), textcoords="offset points", ha="right", color=c["ink"], fontsize=10.5, fontweight="semibold")
    ax.annotate(f"{IMPLIED_COUNTED:.2f}", (IMPLIED_COUNTED, mid), xytext=(4, 9), textcoords="offset points", ha="left", color=c["ink"], fontsize=10.5, fontweight="semibold")
    handles = [plt.Line2D([], [], color=c["model"], linewidth=2, marker="o", markersize=7, label="Independents not counted as Democratic seats"),
               plt.Line2D([], [], color=c["traders"], linewidth=2, marker="D", markersize=6, label="Independents counted as Democratic seats")]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False, fontsize=10,
              labelcolor=c["ink2"], columnspacing=1.5, borderaxespad=0.2, borderpad=0)
    ax.set_xlim(0, 0.95)
    ax.set_ylim(0.68, 0.93)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v * 100:.0f}¢")
    ax.grid(False, axis="x")
    ax.set_xlabel("Correlation: 0 is unrelated, 1 is moving as one", color=c["ink2"], fontsize=10.5)
    return fig


def fit(c: dict, curve: pd.DataFrame):
    h = 4.6
    best = curve.loc[curve.ratio.idxmax()]
    at = lambda r: float(curve.ratio.iloc[(curve.rho - r).abs().argmin()])
    fig, ax = _axes(c, (9, h), "Looking back, the result weakly favors a correlation above 0.38",
                    "How likely the 18-race result was at each correlation, relative to 0.38. Not part of the locked test.")
    _top(fig, 0.72, h, left=0.08, right=0.97, bottom=0.15)
    ax.axhline(1, color=c["base"], linewidth=1.2, linestyle=(0, (4, 3)), zorder=2)
    ax.plot(curve.rho, curve.ratio, color=c["model"], linewidth=2, solid_capstyle="round", zorder=3)
    points = ((MODEL_RHO, f"Our model, {MODEL_RHO:.2f}", (-8, 9), "right"),
              (float(best.rho), f"Best fit, {best.rho:.2f} ({best.ratio:.2f}×)", (0, 11), "center"),
              (TRADERS_RHO, f"Traders, about {TRADERS_RHO:.1f} ({at(TRADERS_RHO):.2f}×)", (-9, -7), "right"))
    for r, label, off, ha in points:
        ax.scatter([r], [at(r)], s=70, color=c["model"], edgecolors=c["ring"], zorder=5)
        ax.annotate(label, (r, at(r)), xytext=off, textcoords="offset points", ha=ha, va="top" if off[1] < 0 else "bottom",
                    color=c["ink"], fontsize=10.5, fontweight="semibold")
    ax.set_xlim(0, 0.95)
    ax.set_ylim(0, 1.75)
    ax.set_yticks([0, 0.5, 1, 1.5])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:g}×")
    ax.grid(False)
    for v in (0.5, 1.5):
        ax.axhline(v, color=c["grid"], linewidth=0.8, zorder=1)
    ax.set_xlabel("Correlation: 0 is unrelated, 1 is moving as one", color=c["ink2"], fontsize=10.5)
    return fig


def _with_open_races(results: dict[str, str], ak: str, ms: str) -> dict[str, str]:
    out = dict(results)
    for ev, winner in zip(OPEN_RACES, (ak, ms)):
        if f"{ev}-D" not in results:
            out[f"{ev}-D"], out[f"{ev}-R"] = ("yes", "no") if winner == "D" else ("no", "yes")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--png-dir", type=Path, help="also save PNG copies here")
    args = ap.parse_args()
    plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]
    plt.rcParams["svg.hashsalt"] = "kalshi-combo"
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

    d = pd.read_csv(SCAN)
    fetched = fetch_results({t for m in d.leg_markets for t in m.split("|")})
    results = {**read_results(ASSUMED), **fetched}

    scenarios = [("Democrats win AK and MS (most likely)", score(d, results)),
                 ("Democrats win AK, Republicans win MS", score(d, _with_open_races(fetched, "D", "R"))),
                 ("Leaving out AK and MS combos", score(d[~d.leg_markets.str.contains("|".join(OPEN_RACES))], fetched)),
                 ("Republicans win AK, Democrats win MS", score(d, _with_open_races(fetched, "R", "D"))),
                 ("Republicans win AK and MS", score(d, _with_open_races(fetched, "R", "R")))]
    accuracy_rows = [{"label": name, **s} for name, s in scenarios]

    t = d[d.maker_mid.notna()].copy()
    t["y"] = [combo_outcome(m, results) for m in t.leg_markets]
    states = t.leg_markets.map(lambda s: {x.split("-")[0].replace("SENATE", "") for x in s.split("|")})
    four = states.map(lambda s: s <= FOUR_STATES)
    same = t.leg_markets.map(lambda s: len({x.endswith("-R") for x in s.split("|")}) == 1)
    brier = lambda x: {"model": ((x.engine - x.y) ** 2).mean(), "traders": ((x.maker_mid - x.y) ** 2).mean()}
    groups = [(f"Both legs in MI, TX, OH or ME ({four.sum()})", t[four]), (f"All other combos ({(~four).sum()})", t[~four]),
              (f"Same party on both legs ({same.sum()})", t[same]), (f"Mixed parties ({(~same).sum()})", t[~same])]
    breakdown_rows = [{"label": name, **brier(x)} for name, x in groups]

    prob = {}
    for m, p in zip(d.leg_markets, d.leg_probs):
        prob.update(zip(m.split("|"), map(float, p.split())))
    winners = tuple(Leg(k.rsplit("-", 1)[0], k, k, prob[k]) for k, v in results.items() if v == "yes")
    base = engine_price(winners, MODEL_RHO)
    grid = np.round(np.arange(0, 0.951, 0.01), 2)
    curve = pd.DataFrame({"rho": grid, "ratio": [engine_price(winners, float(r)) / base for r in grid]})

    OUT.mkdir(parents=True, exist_ok=True)
    charts = {"accuracy": lambda c: accuracy(c, accuracy_rows), "breakdown": lambda c: breakdown(c, breakdown_rows),
              "prices": lambda c: prices(c, t), "control": control, "fit": lambda c: fit(c, curve)}
    for name, draw in charts.items():
        for theme, c in THEMES.items():
            fig = draw(c)
            fig.savefig(OUT / f"{name}-{theme}.svg", transparent=True, metadata={"Date": None})
            if args.png_dir:
                args.png_dir.mkdir(parents=True, exist_ok=True)
                fig.patch.set_alpha(1)
                fig.savefig(args.png_dir / f"{name}-{theme}.png", dpi=110, facecolor=c["ring"])
            plt.close(fig)
    print(f"wrote {len(charts) * len(THEMES)} files to {OUT}")
    print(pd.DataFrame(accuracy_rows)[["label", "two_sided_resolved", "brier_engine", "brier_maker_mid", "brier_independent"]].round(4).to_string(index=False))
    print(pd.DataFrame(breakdown_rows).round(4).to_string(index=False))
    print("best", curve.loc[curve.ratio.idxmax()].round(3).to_dict(), "at 0.9", round(float(curve.ratio[curve.rho == 0.9].iloc[0]), 3), "winners", len(winners))


if __name__ == "__main__":
    main()
