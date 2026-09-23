"""Render individual async-off DFlash requests with observed endpoint load."""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator


ROOT = Path(__file__).resolve().parents[1]
PROSE_CASES = (
    ("explanation", "Explanatory prose"),
    ("narrative", "Original fiction"),
    ("analysis", "Analytical prose"),
)


def summarize(rows, first_output_key, earlier_validation=False):
    """Retain each request and its load; do not pool unlike traffic conditions."""
    if len(rows) != 2:
        raise ValueError(f"Expected two requests per workload, found {len(rows)}")
    requests = []
    for row in rows:
        tokens = row["usage"]["completion_tokens"]
        elapsed = row["elapsed_s"]
        first = row[first_output_key]
        if not isinstance(tokens, int) or tokens <= 0:
            raise ValueError(f"Invalid completion token count for {row['id']}")
        if not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError(f"Invalid elapsed time for {row['id']}")
        if not math.isfinite(first) or not 0 <= first <= elapsed:
            raise ValueError(f"Invalid first-output time for {row['id']}")
        measured = row.get("e2e_output_tokens_per_s")
        if measured is not None and not math.isclose(
            measured, tokens / elapsed, rel_tol=1e-5, abs_tol=1e-5
        ):
            raise ValueError(f"Recorded rate disagrees with tokens / seconds for {row['id']}")
        peak = None if earlier_validation else row["load"]["observed_peak_running"]
        if not earlier_validation and (
            not isinstance(peak, (int, float)) or not math.isfinite(peak) or peak < 1 or peak != int(peak)
        ):
            raise ValueError(f"Expected a positive sampled peak request count for {row['id']}")
        requests.append({
            "id": row["id"],
            "repeat": row.get("repeat"),
            "completion_tokens": tokens,
            "elapsed_s": elapsed,
            "e2e_output_tokens_per_s": tokens / elapsed,
            "first_model_output_s": first,
            "observed_peak_running": peak,
        })
    rates = [request["e2e_output_tokens_per_s"] for request in requests]
    return {
        "earlier_validation": earlier_validation,
        "requests": requests,
        "range_e2e_output_tokens_per_s": [min(rates), max(rates)],
    }


def render(prose_path, validation_path, output_dir):
    prose = json.loads(Path(prose_path).read_text())
    validation = json.loads(Path(validation_path).read_text())
    extraction = [row for row in validation["frozen"] if row["id"].startswith("mimo-decode-")]
    if not all(row["pass"] for row in extraction):
        raise ValueError("Structured extraction requests must pass validation")
    cases = prose["cases"]
    expected_ids = {case_id for case_id, _ in PROSE_CASES}
    if {row["id"] for row in cases} != expected_ids:
        raise ValueError("Expected explanation, narrative, and analysis prose cases")
    workloads = [("structured_extraction", "Structured extraction\n(earlier validation)",
                  summarize(extraction, "first_model_token_s", earlier_validation=True))]
    for case_id, label in PROSE_CASES:
        rows = sorted((row for row in cases if row["id"] == case_id), key=lambda row: row["repeat"])
        if len({row["repeat"] for row in rows}) != 2:
            raise ValueError(f"Expected two distinct repeat labels for {case_id}")
        workloads.append((case_id, label, summarize(rows, "first_model_output_s")))

    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    bg, fg, muted, grid = "#11171B", "#F2EFE8", "#ABB8BD", "#35434B"
    solo_color, overlap_color, earlier_color = "#AAD7CB", "#EDBA83", "#90A0AB"
    overlap_count = sum(row["load"]["observed_peak_running"] > 1 for row in cases)
    fig = plt.figure(figsize=(14, 8.8), facecolor=bg)
    fig.text(.065, .935, "MiMo V2.6 Pro RL: individual workload requests", fontsize=24, color=fg, weight="bold")
    fig.text(.065, .880, "DFlash K7 · async scheduling off · same eight DGX Sparks", fontsize=15, color=muted)
    fig.text(.065, .835, "Official weights · temperature 0 · thinking off · single benchmark client", fontsize=11, color=muted)
    fig.text(.065, .788, f"Live endpoint: background inference observed during {overlap_count}/{len(cases)} prose requests.",
             fontsize=13, color=overlap_color, weight="bold")
    fig.text(.065, .753, "These requests have different load conditions; this is not a controlled workload comparison.",
             fontsize=11, color=muted)

    ax = fig.add_axes([.255, .365, .49, .335], facecolor=bg)
    positions = list(reversed(range(len(workloads))))
    largest = max(summary["range_e2e_output_tokens_per_s"][1] for _, _, summary in workloads)
    for y, (_, _, summary) in zip(positions, workloads):
        for offset, request in zip((.16, -.16), summary["requests"]):
            rate = request["e2e_output_tokens_per_s"]
            color = earlier_color if summary["earlier_validation"] else (
                solo_color if request["observed_peak_running"] == 1 else overlap_color
            )
            ax.scatter([rate], [y + offset], s=53, color=color, edgecolors=bg, linewidth=.7, zorder=4)
            ax.text(rate + largest * .023, y + offset, f"{rate:.2f}",
                    va="center", color=color, fontsize=12, weight="bold")
        low, high = summary["range_e2e_output_tokens_per_s"]
        ax.text(1.25, y, f"{low:.2f}–{high:.2f}", transform=ax.get_yaxis_transform(),
                ha="center", va="center", color=fg, fontsize=14)
    ax.text(1.25, 1.025, "Observed range\n(tokens / second)", transform=ax.transAxes,
            ha="center", va="bottom", color=muted, fontsize=11, linespacing=1.4)
    ax.set_yticks(positions, [label for _, label, _ in workloads], color=fg, fontsize=13)
    ax.set_xlim(0, largest * 1.16)
    ax.set_ylim(-.55, len(workloads) - .45)
    ax.axhline(2.5, color=grid, linewidth=.8, linestyle=(0, (4, 4)), zorder=1)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.tick_params(axis="both", colors=muted, length=0, pad=10)
    ax.set_xlabel("End-to-end output tokens / second", color=muted, fontsize=12, labelpad=14)
    ax.grid(axis="x", color=grid, linewidth=.7, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    legend = [
        Line2D([], [], color=solo_color, marker="o", linestyle="none", markersize=7, label="Sampled peak: 1 running request"),
        Line2D([], [], color=overlap_color, marker="o", linestyle="none", markersize=7, label="Sampled peak: >1 running request"),
        Line2D([], [], color=earlier_color, marker="o", linestyle="none", markersize=7, label="Earlier extraction validation"),
    ]
    fig.legend(handles=legend, loc="center left", bbox_to_anchor=(.055, .255), ncol=3,
               frameon=False, labelcolor=fg, fontsize=10, handletextpad=.55, columnspacing=2)
    fig.text(.065, .203, "Each dot is one request. Two earlier extraction tasks; each prose prompt repeated twice. All requests shown.", color=muted, fontsize=10)
    fig.text(.065, .160, "Rates include prefill and request overhead. Sampled peak 1 does not prove isolation. No matched native prose baseline.", color=muted, fontsize=10)
    fig.text(.065, .117, "Sources: results/synchronous-validation.json (frozen extraction) and results/prose-throughput.json", color=muted, fontsize=10)
    fig.text(.065, .045, "AEVONIX RESEARCH  /  September 2026", color=muted, fontsize=10)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / "workload-results.png", dpi=140, facecolor=bg)
    fig.savefig(output / "workload-results.svg", facecolor=bg, metadata={"Date": None})
    plt.close(fig)
    return {case_id: summary for case_id, _, summary in workloads}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prose", default=ROOT / "results/prose-throughput.json", type=Path)
    parser.add_argument("--validation", default=ROOT / "results/synchronous-validation.json", type=Path)
    parser.add_argument("--output", default=ROOT / "assets", type=Path)
    args = parser.parse_args()
    print(json.dumps(render(args.prose, args.validation, args.output), indent=2))
