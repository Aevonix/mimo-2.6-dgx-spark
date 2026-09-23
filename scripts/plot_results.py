"""Render the normal-model comparison from the published result JSON."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def render(result_path, output_dir):
    data = json.loads(Path(result_path).read_text())
    rows = data["rows"]
    long = [row for key, row in sorted(rows.items()) if key.startswith("mimo-decode-")]
    short = [row for key, row in rows.items() if not key.startswith("mimo-decode-")]
    assert len(long) == 4 and len(short) == 8
    assert all(row[mode]["pass"] for row in long for mode in ("native", "spec"))
    assert all(row["native"]["completion_tokens"] == row["spec"]["completion_tokens"] for row in long)
    rate = {mode: sum(row[mode]["completion_tokens"] for row in long) /
            sum(row[mode]["elapsed_s"] for row in long) for mode in ("native", "spec")}
    speedup = rate["spec"] / rate["native"]
    agreement = sum(row["native"]["content_sha256"] == row["spec"]["content_sha256"] for row in rows.values())
    short_pass = [sum(row[mode]["pass"] for row in short) for mode in ("native", "spec")]
    assert short_pass[0] == short_pass[1]
    tokens = {row["native"]["completion_tokens"] for row in long}
    assert len(tokens) == 1
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    bg, fg, muted, grid = "#11171B", "#F2EFE8", "#ABB8BD", "#35434B"
    native_color, spec_color = "#90A0AB", "#AAD7CB"
    fig = plt.figure(figsize=(14, 7.8), facecolor=bg)
    fig.text(.065, .91, "MiMo V2.6 Pro RL on eight DGX Sparks", fontsize=25, color=fg, weight="bold")
    fig.text(.065, .85, "DFlash K7 compared with native decoding", fontsize=15, color=muted)
    ax = fig.add_axes([.205, .34, .48, .37], facecolor=bg)
    values = [rate["native"], rate["spec"]]
    ax.barh([1, 0], values, height=.46, color=[native_color, spec_color], zorder=3)
    for y, mode in [(1, "native"), (0, "spec")]:
        individual = [row[mode]["completion_tokens"] / row[mode]["elapsed_s"] for row in long]
        ax.scatter(individual, np.full(len(individual), y), s=16, color=bg, edgecolors=fg, linewidth=.6, zorder=4)
        ax.text(rate[mode] + 2, y, f"{rate[mode]:.1f}", va="center", color=fg, fontsize=19, weight="bold")
    ax.set_yticks([1, 0], ["Native", "DFlash K7"], color=fg, fontsize=15)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xlim(0, 85)
    ax.set_ylim(-.6, 1.6)
    ax.tick_params(axis="both", colors=muted, length=0, pad=10)
    ax.set_xlabel("End-to-end output tokens / second", color=muted, fontsize=12, labelpad=15)
    ax.grid(axis="x", color=grid, linewidth=.7, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.text(.755, .67, f"{speedup:.1f}×", color=spec_color, fontsize=40, weight="bold")
    fig.text(.755, .62, "long-task speedup", color=muted, fontsize=12)
    fig.text(.755, .49, f"{agreement}/{len(rows)}", color=fg, fontsize=28, weight="bold")
    fig.text(.755, .45, "byte-identical answers", color=muted, fontsize=12)
    fig.text(.755, .36, f"Both modes passed\n{short_pass[0]}/8 short + 4/4 long tasks", color=fg, fontsize=12, linespacing=1.5)
    fig.text(.065, .22, f"Four long tasks · {next(iter(tokens)):,} output tokens each · one request at a time", color=fg, fontsize=13)
    fig.text(.065, .15, "Same patched runtime and official weights. Rate includes prefill and request overhead.", color=muted, fontsize=11)
    fig.text(.065, .105, "Dots show the four tasks. Timings overlapped work on another cohort; these are workload results, not isolated peaks.", color=muted, fontsize=10)
    fig.text(.065, .045, "AEVONIX RESEARCH  /  September 2026", color=muted, fontsize=10)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "normal-results.png", dpi=140, facecolor=bg)
    fig.savefig(output / "normal-results.svg", facecolor=bg, metadata={"Date": None})
    plt.close(fig)
    return {"native_e2e_tps": rate["native"], "dflash_e2e_tps": rate["spec"], "speedup": speedup, "identical_answers": agreement}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results")
    parser.add_argument("output")
    args = parser.parse_args()
    print(json.dumps(render(args.results, args.output), indent=2))
