"""Figures for the rewritten manuscript. Every number drawn comes from build.py's ledger (passed in as `V`).

Style: Arial 8 pt at the printed size, no in-figure titles, monochrome boxes; the held-out test path is outlined
in black. fig_pipeline() raises if any text overflows its box.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

MM = 1 / 25.4
FONT = {"family": ["Arial", "Helvetica", "DejaVu Sans"], "size": 8}
LINE = 3.55          # mm between text lines at 8 pt
GREY, EDGE, INK = "#F2F2F2", "#7A7A7A", "#000000"


def _box(ax, x, y, w, h, header, lines, strong=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=GREY,
                                ec=INK if strong else EDGE, lw=1.1 if strong else 0.6))
    texts = [ax.text(x + 2, y + h - 2.2, header, ha="left", va="top", fontweight="bold", **FONT)]
    for i, s in enumerate(lines):
        texts.append(ax.text(x + 2, y + h - 2.2 - LINE * (i + 1), s, ha="left", va="top", **FONT))
    return (x, y, w, h), texts


def _arrow(ax, x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, lw=0.7, color=INK, shrinkA=0, shrinkB=0))


def fig_pipeline(V, out):
    W, H = 180, 84
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    boxes = []
    boxes.append(_box(ax, 1, 3, 40, 78, "Public sources", [
        "Radiology reports", "   IU*, CT-RATE", "Case reports", "   Eurorad", "Examination questions", "   MedMCQA, MedQA,",
        "   MMLU*, PubMedQA*,", "   MedXpertQA*", "Reports and indications", "for teacher-labeled", "questions",
        "   CheXpert Plus,", "   ReXGradient-160K,", "   CT-RATE, Eurorad", "", "* development and test only"]))
    boxes.append(_box(ax, 47, 3, 41, 78, "Record construction", [
        "State: report, case", "or question stem", "", "Questions: yes/no,", "choice or ordinal score", "",
        "Answer key: human,", "CT-RATE classifier or", "agreement of two", "LLM teachers", "",
        "Split by patient or case", "", "One wording per", "question kind withheld", "from training"]))
    rows = {"train": (54, 27), "dev": (30, 19), "test": (3, 22)}
    boxes.append(_box(ax, 94, rows["train"][0], 33, rows["train"][1], "Training split",
                      [f"{V['train_records']} records", f"{V['train_questions']} questions"]))
    boxes.append(_box(ax, 94, rows["dev"][0], 33, rows["dev"][1], "Development split",
                      [f"{V['dev_records']} records", f"{V['dev_questions']} questions"]))
    boxes.append(_box(ax, 94, rows["test"][0], 33, rows["test"][1], "Test split",
                      [f"{V['test_records']} records", f"{V['test_questions']} questions"], strong=True))
    boxes.append(_box(ax, 133, rows["train"][0], 46, rows["train"][1], "Fine-tuning", [
        "Kev-27B → RadKev-27B", "Kev-9B → RadKev-9B", f"LoRA (rank {V['lora_rank']}), pointer head",
        f"1 epoch, {V['steps_27']} optimizer steps", f"+{V['replay']} replayed Kev records"]))
    boxes.append(_box(ax, 133, rows["dev"][0], 46, rows["dev"][1], "Calibration and selection", [
        "Temperature fitted by", "minimum NLL; prespecified", "model-selection rule"]))
    boxes.append(_box(ax, 133, rows["test"][0], 46, rows["test"][1], "Evaluation (read once)", [
        "vs Kev-27B (primary),", f"{V['n_other_generalist']} other decision models,", "two LLMs (human-key questions);",
        "paired cluster bootstrap"], strong=True))
    _arrow(ax, 41, 42, 47, 42)
    for _, (y, h) in rows.items():
        _arrow(ax, 88, y + h / 2, 94, y + h / 2)
        _arrow(ax, 127, y + h / 2, 133, y + h / 2)
    _arrow(ax, 156, rows["train"][0], 156, rows["dev"][0] + rows["dev"][1])
    _arrow(ax, 156, rows["dev"][0], 156, rows["test"][0] + rows["test"][1])
    _check(fig, boxes)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig1_pipeline.pdf"); fig.savefig(out / "fig1_pipeline.png", dpi=300)
    plt.close(fig)


def _check(fig, boxes):
    """Every text line must sit inside its box (1 mm margin)."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    to_mm = lambda px: px / fig.dpi * 25.4
    for (x, y, w, h), texts in boxes:
        for t in texts:
            if not t.get_text(): continue
            bb = t.get_window_extent(r)
            x0, x1, y0, y1 = to_mm(bb.x0), to_mm(bb.x1), to_mm(bb.y0), to_mm(bb.y1)
            if x1 > x + w - 1 or y0 < y + 0.5 or x0 < x:
                raise ValueError(f"text overflows its box: {t.get_text()!r}")


BLUE, ORANGE, GREY2 = "#1F5FAE", "#D9711F", "#8C8C8C"
KIND = {"rand": (GREY2, "o", "Random pool"), "sim": (BLUE, "s", "Similar pool"), "llm": (ORANGE, "^", "LLM-generated")}


def fig_answer_space(S, out):
    """S: summary from build.answer_space_summary(). a, b: accuracy vs answer-space size for RadKev-27B (filled) and
    Kev-27B (open) on Eurorad diagnosis and MedQA, coloured by distractor source; c: median latency (IQR) vs size."""
    from matplotlib.lines import Line2D
    W, H = 180, 66
    fig = plt.figure(figsize=(W * MM, H * MM))
    axes = [fig.add_axes([x, 0.17, 0.245, 0.63]) for x in (0.065, 0.39, 0.715)]
    for ax, src, letter, label in ((axes[0], "eurorad_dx", "a", "Eurorad diagnosis"), (axes[1], "medqa", "b", "MedQA")):
        for kind, (color, mk, _) in KIND.items():
            for m, fill, dx in (("radkev27", True, 1.04), ("kev27", False, 0.96)):
                pts = sorted((K, v) for (mm, s, k, K), v in S["acc"].items() if mm == m and s == src and k == kind)
                xs = [K * dx for K, _ in pts]; ys = [100 * v["acc"] for _, v in pts]
                lo = [100 * (v["acc"] - v["ci"][0]) for _, v in pts]; hi = [100 * (v["ci"][1] - v["acc"]) for _, v in pts]
                ax.errorbar(xs, ys, yerr=[lo, hi], ls="-" if fill else "--", marker=mk, ms=3.4, lw=0.8, color=color,
                            mfc=color if fill else "white", mec=color, mew=0.8, capsize=0, elinewidth=0.6)
        ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16, 32, 64]); ax.set_xticklabels(["2", "4", "8", "16", "32", "64"])
        ax.set_ylim(30 if src == "medqa" else 50, 101)
        ax.set_xlabel("Options offered (K)", **FONT); ax.set_ylabel("Accuracy (%)", **FONT)
        ax.text(-0.2, 1.05, letter, transform=ax.transAxes, fontweight="bold", **{**FONT, "size": 9})
        ax.text(0.0, 1.05, label, transform=ax.transAxes, **FONT)
    ax = axes[2]
    for m, color, lab in (("radkev27", INK, "RadKev-27B"), ("radkev9", GREY2, "RadKev-9B")):
        pts = sorted((K, v) for (mm, K), v in S["lat"].items() if mm == m)
        xs = [K for K, _ in pts]; med = [v["median"] for _, v in pts]
        ax.errorbar(xs, med, yerr=[[v["median"] - v["q1"] for _, v in pts], [v["q3"] - v["median"] for _, v in pts]],
                    marker="o", ms=3.4, lw=0.8, color=color, capsize=0, elinewidth=0.6, label=lab)
    ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16, 32, 64, 128]); ax.set_xticklabels(["2", "4", "8", "16", "32", "64", "128"])
    ax.set_ylim(0, None)
    ax.set_xlabel("Options offered (K)", **FONT); ax.set_ylabel("Latency per request (ms)", **FONT)
    ax.text(-0.2, 1.05, "c", transform=ax.transAxes, fontweight="bold", **{**FONT, "size": 9})
    ax.text(0.0, 1.05, "Eurorad, one question per request", transform=ax.transAxes, **FONT)
    ax.legend(loc="upper left", frameon=False, prop=FONT, handlelength=1.6)
    for a in axes:
        a.tick_params(labelsize=8, width=0.6, length=2.5); a.minorticks_off()
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): a.spines[sp].set_linewidth(0.6)
    h = [Line2D([], [], color=c, marker=mk, ls="", ms=3.4, label=lab) for c, mk, lab in KIND.values()]
    h += [Line2D([], [], color="0.25", marker="o", ls="-", lw=0.8, ms=3.4, label="RadKev-27B (filled)"),
          Line2D([], [], color="0.25", marker="o", mfc="white", ls="--", lw=0.8, ms=3.4, label="Kev-27B (open)")]
    fig.legend(handles=h, loc="upper left", ncol=5, frameon=False, prop=FONT, bbox_to_anchor=(0.055, 1.0), handlelength=2.0,
               columnspacing=1.4)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "figS1_answer_space.pdf"); fig.savefig(out / "figS1_answer_space.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2: decision models vs language models
C_DM, C_LLM, C_GRID = "#2a78d6", "#eb6834", "#e3e3e3"


def fig_compare(rows, out):
    """rows: list of dicts (group, label, specialised, acc, lo, hi, lat_med, lat_p95 or None).
    a: accuracy on human-key questions (95% CI); b: latency per question (median, line to 95th percentile), log scale."""
    import matplotlib.ticker as mticker
    W, H = 180, 70
    fig = plt.figure(figsize=(W * MM, H * MM))
    n = len(rows); groups = []
    ys, y = [], 0.0
    for i, r in enumerate(rows):
        if i and r["group"] != rows[i - 1]["group"]: y += 0.7
        ys.append(y); y += 1.0
    top = 0.78; bot = 0.17
    axa = fig.add_axes([0.20, bot, 0.25, top - bot]); axb = fig.add_axes([0.62, bot, 0.25, top - bot])
    for ax in (axa, axb):
        ax.set_ylim(ys[-1] + 0.6, ys[0] - 0.6)
        ax.set_yticks(ys); ax.tick_params(axis="y", length=0)
        for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.6); ax.tick_params(axis="x", labelsize=8, width=0.6, length=2.5)
        ax.grid(axis="x", color=C_GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
    axa.set_yticklabels([r["label"] for r in rows], **FONT); axb.set_yticklabels([])
    for r, yy in zip(rows, ys):
        c = C_DM if r["group"] == "dm" else C_LLM
        axa.plot([100 * r["lo"], 100 * r["hi"]], [yy, yy], color=c, lw=1.1, zorder=3, solid_capstyle="butt")
        axa.plot([100 * r["acc"]], [yy], marker="o", ms=5, mfc=c if r["specialised"] else "white", mec=c, mew=1.1, ls="none", zorder=5)
        axa.text(1.03, yy, f"{100 * r['acc']:.1f}", transform=axa.get_yaxis_transform(), va="center", ha="left", **FONT)
        if r["lat_med"] is not None:
            axb.plot([r["lat_med"], r["lat_p95"]], [yy, yy], color=c, lw=1.1, zorder=3, solid_capstyle="butt")
            axb.plot([r["lat_med"]], [yy], marker="o", ms=5, mfc=c if r["specialised"] else "white", mec=c, mew=1.1, ls="none", zorder=5)
            axb.text(1.03, yy, f"{r['lat_med']:.0f}", transform=axb.get_yaxis_transform(), va="center", ha="left", **FONT)
        else:
            axb.text(1.03, yy, "–", transform=axb.get_yaxis_transform(), va="center", ha="left", **FONT)
    lo = min(100 * r["lo"] for r in rows); hi = max(100 * r["hi"] for r in rows)
    axa.set_xlim(5 * int(lo // 5) - 1, 5 * int(hi // 5 + 1) + 1)
    axa.set_xlabel("Accuracy on human-labeled questions (%)", **FONT)
    axb.set_xscale("log"); axb.set_xlim(30, 1000)
    axb.xaxis.set_major_locator(mticker.FixedLocator([30, 100, 300, 1000])); axb.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    axb.xaxis.set_minor_locator(mticker.NullLocator())
    axb.set_xlabel("Latency per question (ms)", **FONT)
    for ax, letter, head in ((axa, "a", "%"), (axb, "b", "Median, ms")):
        ax.text(-0.75 if ax is axa else -0.08, 1.17, letter, transform=ax.transAxes, fontweight="bold", **{**FONT, "size": 9})
        ax.text(1.03, 1.03, head, transform=ax.transAxes, ha="left", va="bottom", color="#555555", **FONT)
    # group labels
    for g, name in (("dm", "Decision models"), ("llm", "Language models")):
        idx = [i for i, r in enumerate(rows) if r["group"] == g]
        axa.text(-0.75, ys[idx[0]] - 0.75, name, transform=axa.get_yaxis_transform(), ha="left", va="center", fontweight="bold", **FONT)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ms=5, mfc="0.3", mec="0.3", ls="", label="radiology-specialized"),
         Line2D([], [], marker="o", ms=5, mfc="white", mec="0.3", ls="", label="released")]
    fig.legend(handles=h, loc="upper center", ncol=2, frameon=False, prop=FONT, bbox_to_anchor=(0.5, 1.0), handletextpad=0.4, columnspacing=2.0)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig2_compare.pdf"); fig.savefig(out / "fig2_compare.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- Figure: training loss and development accuracy
C27, C9, CBASE = "#2F6BD8", "#8FB0EE", "#8C8C8C"   # RadKev-27B / RadKev-9B blues, as in Figures 5 and 9 (2026-10-06)


def _smooth(y, w):
    import numpy as np
    y = np.asarray(y, float); k = np.ones(w) / w
    out = np.convolve(y, k, mode="valid")
    return out, (w - 1) // 2


def fig_training(L, dev_rows, out):
    """L: {name: [(step, loss), ...]} logged every 10 optimizer steps. dev_rows: list of (label, acc27_before, acc27_after,
    acc9_before, acc9_after). a: loss of RadKev-27B and RadKev-9B; b: 9B from Kev-9B vs from the plain base model;
    c: development accuracy by source before (released Kev) and after fine-tuning."""
    import numpy as np
    W, H = 180, 70
    fig = plt.figure(figsize=(W * MM, H * MM))
    axa = fig.add_axes([0.065, 0.19, 0.25, 0.66]); axb = fig.add_axes([0.385, 0.19, 0.25, 0.66])
    axc = fig.add_axes([0.80, 0.19, 0.18, 0.62])
    def curve(ax, key, color, label, ls="-"):
        s = np.array([p[0] for p in L[key]]); y = np.array([p[1] for p in L[key]])
        ax.plot(s, y, color=color, lw=0.4, alpha=0.25)
        m, off = _smooth(y, 50); ax.plot(s[off:off + len(m)], m, color=color, lw=1.2, ls=ls, label=label)
    curve(axa, "rk27", C27, "RadKev-27B"); curve(axa, "rk9", C9, "RadKev-9B")
    curve(axb, "rk9", C9, "from Kev-9B"); curve(axb, "base9", CBASE, "from Qwen3.5-9B-Base")
    for ax, letter in ((axa, "a"), (axb, "b")):
        ax.set_xlabel("Optimizer step", **FONT); ax.set_ylabel("Training loss", **FONT)
        ax.set_xlim(0, 8600); ax.set_ylim(0, 1.6)
        ax.legend(loc="upper right", frameon=False, prop=FONT, handlelength=1.6)
        ax.text(-0.2, 1.06, letter, transform=ax.transAxes, fontweight="bold", **{**FONT, "size": 9})
    ys = np.arange(len(dev_rows))
    for i, (lab, b27, a27, b9, a9) in enumerate(dev_rows):
        for b, a, c, dy in ((b27, a27, C27, -0.17), (b9, a9, C9, 0.17)):
            axc.plot([100 * b, 100 * a], [i + dy, i + dy], color=c, lw=0.8, zorder=2)
            axc.plot([100 * b], [i + dy], marker="o", ms=3.6, mfc="white", mec=c, mew=0.9, ls="none", zorder=3)
            axc.plot([100 * a], [i + dy], marker="o", ms=3.6, mfc=c, mec=c, mew=0.9, ls="none", zorder=3)
    axc.set_yticks(ys); axc.set_yticklabels([r[0] for r in dev_rows], **FONT); axc.set_ylim(len(dev_rows) - 0.5, -0.5)
    axc.tick_params(axis="y", length=0); axc.spines["left"].set_visible(False)
    axc.set_xlabel("Development accuracy (%)", **FONT); axc.grid(axis="x", color="#e3e3e3", lw=0.6, zorder=0)
    axc.text(-0.95, 1.10, "c", transform=axc.transAxes, fontweight="bold", **{**FONT, "size": 9})
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ms=3.6, mfc="white", mec="0.3", ls="", label="released Kev"),
         Line2D([], [], marker="o", ms=3.6, mfc="0.3", mec="0.3", ls="", label="fine-tuned")]
    axc.legend(handles=h, loc="lower left", frameon=False, prop=FONT, bbox_to_anchor=(-0.62, 1.0), ncol=2, handletextpad=0.3, columnspacing=0.8)
    for a in (axa, axb, axc):
        a.tick_params(labelsize=8, width=0.6, length=2.5)
        for sp in ("top", "right"): a.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): a.spines[sp].set_linewidth(0.6)
    axc.tick_params(axis="y", length=0); axc.spines["left"].set_visible(False)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_training.pdf"); fig.savefig(out / "fig_training.png", dpi=300)
    plt.close(fig)


def fig_training_v3(L, A, TS, dev_rows, out, smooth=30):
    """Present models (runs v3f-kev-27b-dp, v3f-kev-9b-dp). L, A: {"rk27"|"rk9": [(step, value)]}, rank-0 loss and micro-batch
    accuracy logged every 10 optimizer steps; TS: {"rk27"|"rk9": [seconds]} wall time of every optimizer step; dev_rows: (label,
    acc27_before, acc27_after, acc9_before, acc9_after). a-c: thin, logged values; thick, moving average over 300 optimizer steps.
    d: development accuracy by source before (released Kev) and after fine-tuning."""
    import numpy as np
    W, H = 180, 125
    fig = plt.figure(figsize=(W * MM, H * MM))
    axa = fig.add_axes([0.075, 0.60, 0.385, 0.34]); axb = fig.add_axes([0.595, 0.60, 0.385, 0.34])
    axc = fig.add_axes([0.075, 0.09, 0.385, 0.34]); axd = fig.add_axes([0.745, 0.09, 0.235, 0.34])
    models = (("rk27", C27, "RadKev-27B"), ("rk9", C9, "RadKev-9B"))
    def fit(ax, ms, floor=None):   # y-range from the moving averages (+15% of their span); logged values outside it are clipped
        lo, hi = min(float(m.min()) for m in ms), max(float(m.max()) for m in ms); pad = 0.15 * (hi - lo)
        ax.set_ylim(lo - pad if floor is None else max(floor, lo - pad), hi + pad)
    for ax, D, scale in ((axa, L, 1), (axb, A, 100)):
        ms = []
        for key, color, label in models:
            s = np.array([p[0] for p in D[key]]); y = scale * np.array([p[1] for p in D[key]])
            ax.plot(s, y, color=color, lw=0.3, alpha=0.12)
            m, off = _smooth(y, smooth); ax.plot(s[off:off + len(m)], m, color=color, lw=1.2, label=label); ms.append(m)
        fit(ax, ms, floor=0)
    ms = []
    for key, color, label in models:
        y = np.asarray(TS[key], float); s = np.arange(1, len(y) + 1)
        axc.plot(s, y, color=color, lw=0.2, alpha=0.07)
        m, off = _smooth(y, 10 * smooth); axc.plot(s[off:off + len(m)], m, color=color, lw=1.2, label=label); ms.append(m)
    fit(axc, ms, floor=0)
    axa.set_ylabel("Training loss", **FONT)
    axb.set_ylabel("Training accuracy (%)", **FONT); axb.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(5))
    axc.set_ylabel("Time per optimizer step (s)", **FONT)
    for ax in (axa, axb, axc):
        ax.set_xlabel("Optimizer step", **FONT); ax.set_xlim(0, 4700)
    axa.legend(loc="upper right", frameon=False, prop=FONT, handlelength=1.6)
    ys = np.arange(len(dev_rows))
    for i, (lab, b27, a27, b9, a9) in enumerate(dev_rows):
        for b, a, c, dy in ((b27, a27, C27, -0.17), (b9, a9, C9, 0.17)):
            axd.plot([100 * b, 100 * a], [i + dy, i + dy], color=c, lw=0.8, zorder=2)
            axd.plot([100 * b], [i + dy], marker="o", ms=3.4, mfc="white", mec=c, mew=0.9, ls="none", zorder=3)
            axd.plot([100 * a], [i + dy], marker="o", ms=3.4, mfc=c, mec=c, mew=0.9, ls="none", zorder=3)
    axd.set_yticks(ys); axd.set_yticklabels([r[0] for r in dev_rows], **FONT); axd.set_ylim(len(dev_rows) - 0.5, -0.5)
    axd.set_xlabel("Development accuracy (%)", **FONT); axd.grid(axis="x", color="#e3e3e3", lw=0.6, zorder=0)
    from matplotlib.lines import Line2D
    h = [Line2D([], [], marker="o", ms=3.4, mfc="white", mec="0.3", ls="", label="released Kev"),
         Line2D([], [], marker="o", ms=3.4, mfc="0.3", mec="0.3", ls="", label="fine-tuned")]
    axd.legend(handles=h, loc="lower left", frameon=False, prop=FONT, bbox_to_anchor=(-0.62, 1.0), ncol=2, handletextpad=0.3, columnspacing=0.8)
    for ax, lab, x in ((axa, "a", -0.15), (axb, "b", -0.15), (axc, "c", -0.15), (axd, "d", -0.80)):
        ax.text(x, 1.08, lab, transform=ax.transAxes, fontweight="bold", **{**FONT, "size": 9})
        ax.tick_params(labelsize=8, width=0.6, length=2.5)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_linewidth(0.6); ax.spines[sp].set_color("#BDBDBD")
    axd.tick_params(axis="y", length=0); axd.spines["left"].set_visible(False)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_training.pdf"); fig.savefig(out / "fig_training.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- Figure: two-teacher agreement, worked examples
C_MG, C_QW, C_T = "#9C7B5B", "#D2AE82", "#4A4F59"   # LLM teachers in the tan family of the results figures; soft target slate


def fig_teacher(examples, out):
    """examples: [(title, state_text, question, [(key, label)], p_medgemma, p_qwen, retained_bool)]."""
    import textwrap
    import numpy as np
    W = 180; rowh = 52; H = rowh * len(examples) + 10
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    boxes = []
    for r, (title, state, question, opts, pm, pq, kept) in enumerate(examples):
        y0 = H - 8 - (r + 1) * rowh + 4
        txt = textwrap.wrap("Case: " + state, 33) + [""] + textwrap.wrap("Question: " + question, 33)
        boxes.append(_box(ax, 1, y0, 56, rowh - 6, title, txt))
        # bars
        bx = fig.add_axes([99 / W, (y0 + 8) / H, 37 / W, (rowh - 16) / H])
        n = len(opts); ys = np.arange(n)
        mean = {k: (pm[k] + pq[k]) / 2 for k, _ in opts}
        series = [("MedGemma-27B-text", pm, C_MG), ("Qwen3.8-27B", pq, C_QW)] + ([("Mean (soft target)", mean, C_T)] if kept else [])
        hgt = 0.8 / len(series)
        for j, (lab, p, c) in enumerate(series):
            bx.barh(ys - 0.4 + hgt * (j + 0.5), [100 * p[k] for k, _ in opts], height=hgt * 0.9, color=c, label=lab)
        bx.set_yticks(ys); bx.set_yticklabels([l for _, l in opts], **FONT); bx.set_ylim(n - 0.5, -0.5)
        bx.set_xlim(0, 100); bx.set_xlabel("Probability (%)", **FONT)
        bx.tick_params(labelsize=8, width=0.6, length=2.5); bx.tick_params(axis="y", length=0)
        for sp in ("top", "right"): bx.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): bx.spines[sp].set_linewidth(0.6); bx.spines[sp].set_color("#BDBDBD")
        if r == 0:
            hs, ls_ = bx.get_legend_handles_labels()
            fig.legend(hs, ls_, loc="upper center", ncol=3, frameon=False, prop=FONT, bbox_to_anchor=(0.62, 0.995), handlelength=1.2, columnspacing=1.6)
        # outcome
        tm, tq = max(pm, key=pm.get), max(pq, key=pq.get)
        lab = dict(opts)
        wr = lambda t: textwrap.wrap(t, 21)
        if kept:
            res = wr(f"Both teachers rank “{lab[tm]}” first.") + ["", "Retained.", "Label:"] + wr(lab[tm]) + ["Target: mean of the", "two distributions"]
        else:
            res = wr(f"MedGemma ranks “{lab[tm]}” first; Qwen ranks “{lab[tq]}” first.") + ["", "Discarded."]
        boxes.append(_box(ax, 144, y0, 35, rowh - 6, "Outcome", res, strong=kept))
        _arrow(ax, 138.5, y0 + (rowh - 6) / 2, 144, y0 + (rowh - 6) / 2)
    _check(fig, boxes)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_teacher.pdf"); fig.savefig(out / "fig_teacher.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- Figure: Kev on the Qwen backbone
C_FROZEN, C_TRAIN, C_TXT = "#E6E6E6", "#CFE0F5", "#1a1a1a"


def _rect(ax, x, y, w, h, fc, ec="#7A7A7A", lw=0.6, text=None, bold=False, size=8, color=C_TXT, rounding=0.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}", fc=fc, ec=ec, lw=lw))
    if text is not None:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontweight="bold" if bold else "normal",
                **{**FONT, "size": size}, color=color)


def fig_kev(V, out):
    W, H = 180, 96
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")
    T = lambda x, y, s, **k: ax.text(x, y, s, **{**FONT, **k})
    BLUE_EDGE, MASK = "#1F5FAE", "#7A9CC6"
    def line(xs, ys): ax.plot(xs, ys, color=INK, lw=0.7, solid_capstyle="butt")

    # ---- a: one request
    T(1, 93, "a", fontweight="bold", size=9, va="top"); T(5, 93, "One request: a state and its questions", va="top")
    _rect(ax, 1, 77, 56, 7, C_FROZEN, text="State (report, case or question)")
    for yq, q in ((67, "Q1"), (58.5, "Q2")):
        x = 1
        for s, w in ((q, 8.5), ("option 1", 12), ("option 2", 12), ("…", 5.5), ("option K", 12), ("d", 6)):
            _rect(ax, x, yq, w - 0.7, 6.5, "white", text=s, bold=(s == "d"), size=7.5)
            x += w
    T(1, 54.5, "d, decision token of the question", va="center", size=7.5, color="#555555")
    T(1, 48.5, "Attention mask (row attends to column)", va="center")
    labs, allow = ["State", "Q1", "Q2"], [[1, 0, 0], [1, 1, 0], [1, 0, 1]]
    x0, y0, cs = 11, 20, 7
    for i in range(3):
        T(x0 - 1.5, y0 + (2 - i) * cs + cs / 2, labs[i], ha="right", va="center")
        T(x0 + i * cs + cs / 2, y0 + 3 * cs + 1.0, labs[i], ha="center", va="bottom")
        for j in range(3):
            ax.add_patch(plt.Rectangle((x0 + j * cs, y0 + (2 - i) * cs), cs, cs, fc=MASK if allow[i][j] else "white", ec=EDGE, lw=0.6))
    for k, (fc, lab) in enumerate(((MASK, "attends"), ("white", "masked"))):
        ax.add_patch(plt.Rectangle((x0 + 3 * cs + 4, y0 + 14 - 6 * k), 4, 4, fc=fc, ec=EDGE, lw=0.6))
        T(x0 + 3 * cs + 9.5, y0 + 16 - 6 * k, lab, va="center", size=7.5)
    for k, s in enumerate(("Each question attends to the state and to", "itself but not to the other questions, so", "the state is encoded once for all of them.")):
        T(1, 13 - 3.4 * k, s, va="center", size=7.5)
    _arrow(ax, 57.5, 64.5, 62.6, 64.5)

    # ---- b: backbone (bottom to top)
    T(63, 93, "b", fontweight="bold", size=9, va="top"); T(67, 93, "Qwen3.8-27B backbone", va="top")
    ax.add_patch(FancyBboxPatch((63, 11), 52, 74, boxstyle="round,pad=0,rounding_size=1.2", fc="white", ec=EDGE, lw=0.6))
    T(66, 16.6, f"{V['bb_layers']} layers ({V['bb_linear']} Gated DeltaNet, {V['bb_full']} gated", va="center", size=7.5, color="#555555")
    T(66, 13.6, f"full attention); hidden size {V['bb_hidden']}", va="center", size=7.5, color="#555555")
    _rect(ax, 66, 20, 46, 6.5, C_FROZEN, text="Token embeddings")
    _arrow(ax, 89, 26.5, 89, 29.5)
    ax.add_patch(FancyBboxPatch((65.5, 29.5), 47, 39, boxstyle="round,pad=0,rounding_size=1.0", fc="none", ec=EDGE, lw=0.6, ls=(0, (3, 2))))
    for k, (yb, name) in enumerate(((31.5, "Gated DeltaNet"), (39.0, "Gated DeltaNet"), (46.5, "Gated DeltaNet"), (54.0, "Gated attention"))):
        _rect(ax, 67.5, yb, 30, 6, C_FROZEN, text=name)
        _rect(ax, 99.5, yb + 0.75, 11, 4.5, C_TRAIN, ec=BLUE_EDGE, text="LoRA", size=7.5)
    T(67.5, 64.5, "repeated 16 times", va="center", size=7.5)
    _arrow(ax, 89, 68.5, 89, 75.5)
    _rect(ax, 66, 75.5, 46, 7, "white", ec=INK, lw=0.7, text="Hidden state of every token")

    # ---- c: pointer head
    T(121, 93, "c", fontweight="bold", size=9, va="top"); T(125, 93, "Pointer head", va="top")
    _rect(ax, 121, 75.5, 27, 7, C_FROZEN, text="h(d)")
    _rect(ax, 152, 75.5, 27, 7, C_FROZEN, text="h(option k)")
    line([112, 118], [79, 79]); _arrow(ax, 118, 79, 121, 79)            # to h(d)
    line([118, 118, 165.5], [79, 87, 87]); _arrow(ax, 165.5, 87, 165.5, 82.5)   # to h(option k)
    _rect(ax, 121, 61.5, 27, 7, C_TRAIN, ec=BLUE_EDGE, text=f"{V['bb_hidden']} → {V['head_dim']}")
    _rect(ax, 152, 61.5, 27, 7, C_TRAIN, ec=BLUE_EDGE, text=f"{V['bb_hidden']} → {V['head_dim']}")
    _arrow(ax, 134.5, 75.5, 134.5, 68.5); _arrow(ax, 165.5, 75.5, 165.5, 68.5)
    _rect(ax, 128, 46, 44, 8, "white", text="score: scaled dot product")
    _arrow(ax, 134.5, 61.5, 141, 54); _arrow(ax, 165.5, 61.5, 159, 54)
    _rect(ax, 128, 31.5, 44, 8, "white", text="softmax over the K options (÷ T)")
    _arrow(ax, 150, 46, 150, 39.5)
    _rect(ax, 128, 17, 44, 8, "white", ec=INK, lw=1.0, text="P(option k)", bold=True)
    _arrow(ax, 150, 31.5, 150, 25)

    # ---- legend
    _rect(ax, 63, 3, 5, 3.6, C_FROZEN); T(69.5, 4.8, "frozen backbone weights", va="center", size=7.5)
    _rect(ax, 112, 3, 5, 3.6, C_TRAIN, ec=BLUE_EDGE); T(118.5, 4.8, f"trained: LoRA (rank {V['lora_rank']}) and pointer head", va="center", size=7.5)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_kev.pdf"); fig.savefig(out / "fig_kev.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- Figure: primary comparison by decision family
def fig_primary(rows, out):
    """rows: list of (group, label, n, diff, lo, hi) in percentage points; group 'human' or 'machine'."""
    import matplotlib.ticker as mticker
    W, H = 180, 92
    fig = plt.figure(figsize=(W * MM, H * MM))
    ys, y = [], 0.0
    for i, r in enumerate(rows):
        if i and r[0] != rows[i - 1][0]: y += 0.9
        ys.append(y); y += 1.0
    ax = fig.add_axes([0.37, 0.15, 0.40, 0.74])
    ax.set_ylim(ys[-1] + 0.6, ys[0] - 0.9)
    ax.axvline(0, color="#7A7A7A", lw=0.7, zorder=1)
    for (g, lab, n, d, lo, hi), yy in zip(rows, ys):
        c = C_DM if g == "human" else "#1a1a1a" if g == "summary" else "#8C8C8C"
        ax.plot([lo, hi], [yy, yy], color=c, lw=1.1, zorder=3, solid_capstyle="butt")
        ax.plot([d], [yy], marker="o", ms=5, mfc=c, mec=c, ls="none", zorder=4)
        s = lambda v: ("+" if v > 0 else "−" if v < 0 else "") + f"{abs(v):.1f}"
        ax.text(1.03, yy, f"{s(d)} ({s(lo)} to {s(hi)})", transform=ax.get_yaxis_transform(), va="center", ha="left", **FONT)
    ax.set_yticks(ys); ax.set_yticklabels([f"{r[1]} (n = {r[2]:,})" if r[0] != "summary" else r[1] for r in rows], **FONT)
    ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6); ax.tick_params(axis="x", labelsize=8, width=0.6, length=2.5)
    ax.grid(axis="x", color="#e3e3e3", lw=0.6, zorder=0); ax.set_axisbelow(True)
    ax.set_xlabel("RadKev-27B minus Kev-27B, accuracy (percentage points)", **FONT)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: (f"{v:+g}" if v else "0").replace("-", "\u2212")))
    ax.text(1.03, 1.0, "Difference (95% CI)", transform=ax.transAxes, ha="left", va="bottom", color="#555555", **FONT)
    for g, name in (("summary", "Task-averaged accuracy"), ("human", "Human answer keys"), ("machine", "Machine-derived answer keys")):
        idx = [i for i, r in enumerate(rows) if r[0] == g]
        ax.text(-0.86, ys[idx[0]] - 0.85, name, transform=ax.get_yaxis_transform(), ha="left", va="center", fontweight="bold", **FONT)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_primary.pdf"); fig.savefig(out / "fig_primary.png", dpi=300)
    plt.close(fig)


def fig_forest(rows, out, name="fig_primary"):
    """Journal-style forest table. rows: dicts with kind ('summary'|'header'|'human'|'machine'), label, n, a (Kev-27B %),
    b (RadKev-27B %), d, lo, hi (pp). Columns: label | n | Kev-27B | RadKev-27B | forest | difference (95% CI)."""
    import numpy as np
    import matplotlib.transforms as mtrans
    W = 180; rh = 4.6; head = 15; bottom = 15
    H = head + bottom + rh * len(rows)
    fig = plt.figure(figsize=(W * MM, H * MM))
    X0, X1 = 106, 143                      # forest region (mm)
    ax = fig.add_axes([X0 / W, bottom / H, (X1 - X0) / W, rh * len(rows) / H])
    ax.set_zorder(2); ax.patch.set_alpha(0)
    ax.set_ylim(len(rows) - 0.5, -0.5); ax.set_xlim(-10, 25)
    tr = mtrans.blended_transform_factory(fig.transFigure, ax.transData)
    xs = lambda mm: mm / W
    yf = lambda mm: mm / H
    INKC, GREYC, BLUEC, SUB = "#1a1a1a", "#8C8C8C", "#1F5FAE", "#555555"
    for i, r in enumerate(rows):
        if r.get("band"):
            fig.add_artist(plt.Rectangle((xs(1), i - 0.5), xs(178), 1.0, transform=tr, fc="#F2F4F7", ec="none", zorder=0))
    ax.axvline(0, color=SUB, lw=0.7, ls=(0, (3, 2)), zorder=1)
    for v in (-10, -5, 5, 10, 15, 20, 25): ax.axvline(v, color="#E3E3E3", lw=0.5, zorder=0)
    nmax = max(r["n"] for r in rows if r["kind"] in ("human", "machine"))
    sg = lambda v: ("+" if v > 0.05 else "\u2212" if v < -0.05 else "") + f"{abs(v):.1f}"
    ab = lambda v: ("\u2212" if v < -0.05 else "") + f"{abs(v):.1f}"
    for i, r in enumerate(rows):
        k = r["kind"]
        if k == "header":
            fig.text(xs(2), i, r["label"], transform=tr, va="center", ha="left", fontweight="bold", **FONT, zorder=3)
            continue
        col = INKC if k == "summary" else BLUEC if k == "human" else GREYC
        tcol = INKC if k != "machine" else SUB
        bold = "bold" if k == "summary" else "normal"
        fig.text(xs(2 if k == "summary" else 5), i, r["label"], transform=tr, va="center", ha="left", color=tcol, fontweight=bold, **FONT, zorder=3)
        fig.text(xs(68), i, f"{r['n']:,}", transform=tr, va="center", ha="right", color=tcol, **FONT, zorder=3)
        fig.text(xs(81), i, f"{r['a']:.1f}", transform=tr, va="center", ha="right", color=tcol, **FONT, zorder=3)
        fig.text(xs(101), i, f"{r['b']:.1f}", transform=tr, va="center", ha="right", color=tcol, fontweight=bold, **FONT, zorder=3)
        ax.plot([r["lo"], r["hi"]], [i, i], color=col, lw=1.0, zorder=3, solid_capstyle="butt")
        if k == "summary":
            h = 0.34
            ax.fill([r["lo"], r["d"], r["hi"], r["d"]], [i, i - h, i, i + h], color=INKC, zorder=4, lw=0)
        else:
            ms = 2.6 + 4.0 * np.sqrt(r["n"] / nmax)
            ax.plot([r["d"]], [i], marker="s", ms=ms, mfc=col, mec=col, ls="none", zorder=4)
        fig.text(xs(147), i, f"{sg(r['d'])} ({ab(r['lo'])} to {ab(r['hi'])})", transform=tr, va="center", ha="left", color=tcol,
                 fontweight=bold, **FONT, zorder=3)
    # header: rules, spanner and column titles
    y_top, y_span, y_cols, y_rule = H - 1.5, H - 5.5, H - 11.0, H - 13.0
    for yy in (y_top, y_rule):
        fig.add_artist(plt.Line2D([xs(1), xs(179)], [yf(yy)] * 2, color=INKC, lw=0.8, transform=fig.transFigure))
    fig.text(xs(88), yf(y_span), "Accuracy (%)", ha="center", va="center", color=SUB, **FONT)
    fig.add_artist(plt.Line2D([xs(70), xs(101)], [yf(y_span - 2.3)] * 2, color=SUB, lw=0.5, transform=fig.transFigure))
    for x, t_, ha in ((2, "Decision family", "left"), (68, "Questions", "right"), (81, "Kev-27B", "right"), (101, "RadKev-27B", "right"),
                      ((X0 + X1) / 2, "Difference (pp)", "center"), (147, "Difference (95% CI)", "left")):
        fig.text(xs(x), yf(y_cols), t_, ha=ha, va="center", fontweight="bold", **FONT)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.6); ax.set_yticks([])
    ax.set_xticks([-10, 0, 10, 20]); ax.set_xticklabels(["\u221210", "0", "+10", "+20"])
    ax.tick_params(axis="x", labelsize=8, width=0.6, length=2.5)
    ax.text(-1.2, -0.12, "\u2190 Kev-27B better", transform=ax.get_xaxis_transform(), ha="right", va="top", color=SUB, **FONT)
    ax.text(1.2, -0.12, "RadKev-27B better \u2192", transform=ax.get_xaxis_transform(), ha="left", va="top", color=SUB, **FONT)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf"); fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- whitepaper-style grouped bars
WP_BASE, WP_ACC, WP_INK, WP_SUB, WP_GRID = "#C5CAD3", "#2F6BD8", "#1a1a1a", "#6B6B6B", "#E9E9E9"


def fig_bars(panels, out, name="fig_primary", models=("Kev-27B", "RadKev-27B"), pad=5):
    vals = [v for p in panels for it in p[1] for v in it[2:4]]
    ymin, ytop = min(vals) - pad, max(vals) + pad
    import matplotlib.transforms as mtrans
    """panels: list of (title, [(label, n, a, b, delta_pp)], muted). a, b in %."""
    import numpy as np
    from matplotlib.patches import Patch
    W, H = 180, 80
    fig = plt.figure(figsize=(W * MM, H * MM))
    counts = [max(len(p[1]), 2.25) for p in panels]
    x0, x1, gap = 0.075, 0.995, 0.045
    unit = (x1 - x0 - gap * (len(panels) - 1)) / sum(counts)
    x = x0; axes = []
    for (title, items, muted), cw in zip(panels, counts):
        c = len(items)
        ax = fig.add_axes([x, 0.25, unit * cw, 0.55]); axes.append(ax); x += unit * cw + gap
        xs = np.arange(c); w = 0.36
        ca, cb = WP_BASE, WP_ACC
        for i, (lab, n_, a, b, d) in enumerate(items):
            ax.bar(i - w / 2 - 0.01, a - ymin, w, bottom=ymin, color=ca, zorder=2)
            ax.bar(i + w / 2 + 0.01, b - ymin, w, bottom=ymin, color=cb, zorder=2)
            off = (ytop - ymin) * 0.012
            ax.text(i - w / 2 - 0.01, a + off, f"{a:.1f}", ha="center", va="bottom", color=WP_SUB, **{**FONT, "size": 6.5})
            ax.text(i + w / 2 + 0.01, b + off, f"{b:.1f}", ha="center", va="bottom", color=WP_INK, **{**FONT, "size": 6.5})
            ds = ("+" if d > 0.05 else "−" if d < -0.05 else "") + f"{abs(d):.1f}"
            ax.text(i, 1.05, ds, transform=mtrans.blended_transform_factory(ax.transData, ax.transAxes), ha="center", va="center", color=cb if d > 0 else WP_SUB, fontweight="bold", **{**FONT, "size": 7.5})
        ax.set_xticks(xs); ax.set_xticklabels([lab for lab, *_ in items], **{**FONT, "size": 7.2}, linespacing=1.1)
        for i, (lab, n_, *_ ) in enumerate(items):
            if n_: ax.text(i, -0.205, f"n = {n_:,}", transform=ax.get_xaxis_transform(), ha="center", va="top", color=WP_SUB, **{**FONT, "size": 6.8})
        ax.set_ylim(ymin, ytop); pad_ = (cw - c) / 2; ax.set_xlim(-0.6 - pad_, c - 0.4 + pad_)
        ax.set_yticks([t for t in range(0, 101, 10) if ymin <= t <= ytop]); ax.grid(axis="y", color=WP_GRID, lw=0.6, zorder=0)
        for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#BDBDBD"); ax.spines["bottom"].set_linewidth(0.6)
        ax.tick_params(axis="x", length=0, pad=3); ax.tick_params(axis="y", length=0, labelsize=7.5, colors=WP_SUB)
        if ax is not axes[0]: ax.set_yticklabels([])
        k = len(axes) - 1
        ax.text(0.0, 1.13, "abcdefgh"[k], transform=ax.transAxes, ha="left", va="bottom", fontweight="bold", color=WP_INK, **{**FONT, "size": 9})
        from matplotlib.transforms import offset_copy
        ax.text(0.0, 1.13, title, transform=offset_copy(ax.transAxes, fig=fig, x=11, y=0, units="points"), ha="left", va="bottom", color=WP_INK, **FONT)
    axes[0].set_ylabel("Accuracy (%)", color=WP_SUB, **FONT)
    axes[0].text(-0.6 - (counts[0] - len(panels[0][1])) / 2, 1.05, "Δ (pp)", transform=mtrans.blended_transform_factory(axes[0].transData, axes[0].transAxes), ha="right", va="center", color=WP_SUB, **{**FONT, "size": 7.5})
    fig.legend(handles=[Patch(color=WP_BASE, label=models[0]), Patch(color=WP_ACC, label=models[1])], loc="upper left",
               ncol=2, frameon=False, prop=FONT, bbox_to_anchor=(0.065, 1.0), handlelength=1.0, handleheight=0.8, columnspacing=1.2)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf"); fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


def fig_bars_one(groups, out, name="fig_primary", models=("Kev-27B", "RadKev-27B")):
    """One axis, grouped bars. groups: list of (bracket_label, muted, [(label, n, a, b, delta)]). Groups are separated by a gap
    and labeled by a bracket under the category labels."""
    import numpy as np
    from matplotlib.patches import Patch
    W, H = 180, 76
    fig = plt.figure(figsize=(W * MM, H * MM))
    ax = fig.add_axes([0.07, 0.30, 0.92, 0.55])
    pos, labels, gap, w = [], [], 0.7, 0.36
    x = 0.0; spans = []
    for title, muted, items in groups:
        start = x
        for lab, n_, a, b, d in items:
            ca, cb = (WP_BASE, WP_ACC) if not muted else ("#DADDE2", "#9DB5E6")
            ax.bar(x - w / 2 - 0.01, a, w, color=ca, zorder=2); ax.bar(x + w / 2 + 0.01, b, w, color=cb, zorder=2)
            ax.text(x - w / 2 - 0.01, a + 1.2, f"{a:.1f}", ha="center", va="bottom", color=WP_SUB, **{**FONT, "size": 6.5})
            ax.text(x + w / 2 + 0.01, b + 1.2, f"{b:.1f}", ha="center", va="bottom", color=WP_INK, **{**FONT, "size": 6.5})
            ds = ("+" if d > 0.05 else "−" if d < -0.05 else "") + f"{abs(d):.1f}"
            ax.text(x, 112, ds, ha="center", va="center", color=(cb if d > 0 else WP_SUB), fontweight="bold", **{**FONT, "size": 7.5})
            pos.append(x); labels.append((lab, n_)); x += 1.0
        spans.append((title, start, x - 1.0, muted)); x += gap
    ax.set_xticks(pos); ax.set_xticklabels([l for l, _ in labels], **{**FONT, "size": 7.2}, linespacing=1.1)
    for p, (_, n_) in zip(pos, labels):
        ax.text(p, -0.19, f"n = {n_:,}", transform=ax.get_xaxis_transform(), ha="center", va="top", color=WP_SUB, **{**FONT, "size": 6.6})
    for title, a, b, muted in spans:     # brackets
        yb = -0.30
        ax.plot([a - 0.45, b + 0.45], [yb, yb], transform=ax.get_xaxis_transform(), color="#9A9A9A", lw=0.7, clip_on=False)
        for xx in (a - 0.45, b + 0.45):
            ax.plot([xx, xx], [yb, yb + 0.025], transform=ax.get_xaxis_transform(), color="#9A9A9A", lw=0.7, clip_on=False)
        ax.text((a + b) / 2, yb - 0.03, title, transform=ax.get_xaxis_transform(), ha="center", va="top",
                color=WP_INK if not muted else WP_SUB, fontweight="bold", **{**FONT, "size": 7.5})
    ax.set_ylim(0, 118); ax.set_xlim(pos[0] - 0.6, pos[-1] + 0.6)
    ax.set_yticks([0, 20, 40, 60, 80, 100]); ax.grid(axis="y", color=WP_GRID, lw=0.6, zorder=0)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#BDBDBD"); ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="x", length=0, pad=3); ax.tick_params(axis="y", length=0, labelsize=7.5, colors=WP_SUB)
    ax.set_ylabel("Accuracy (%)", color=WP_SUB, **FONT)
    ax.text(pos[0] - 0.6, 112, "Δ (pp)", ha="right", va="center", color=WP_SUB, **{**FONT, "size": 7.5})
    fig.legend(handles=[Patch(color=WP_BASE, label=models[0]), Patch(color=WP_ACC, label=models[1])], loc="upper left",
               ncol=2, frameon=False, prop=FONT, bbox_to_anchor=(0.06, 1.0), handlelength=1.0, handleheight=0.8, columnspacing=1.2)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf"); fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- Figure: decision models and language models
C_RAD, C_KEV, C_DMO, C_LLM, C_LLMR = "#2F6BD8", "#9AA3B2", "#C9CED6", "#D2AE82", "#9C7B5B"   # LLMs in a muted tan family (2026-10-06)
C_LLMQ, C_ODEC = "#E9D8C1", "#7C8594"   # LLM with reasoning; OpenAI Decisions (hosted decision model, slate)


def _hbars(ax, rows, fmt, xmax, log=False, xmin=0):
    """rows: (label, value, color, bold)."""
    import numpy as np
    ys = np.arange(len(rows))
    for y, (lab, v, c, bold) in zip(ys, rows):
        ax.barh(y, v - xmin, left=xmin, height=0.68, color=c, zorder=2)
        ax.text(v * (1.06 if log else 1) + (0 if log else (xmax - xmin) * 0.012), y, fmt(v), va="center", ha="left",
                color=WP_INK, fontweight="bold" if bold else "normal", **{**FONT, "size": 7.2})
    ax.set_yticks(ys); ax.set_yticklabels([r[0] for r in rows], **{**FONT, "size": 7.5})
    for t, r in zip(ax.get_yticklabels(), rows):
        if r[3]: t.set_fontweight("bold")
    ax.set_ylim(len(rows) - 0.4, -0.6)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#BDBDBD"); ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="y", length=0, pad=3); ax.tick_params(axis="x", length=0, labelsize=7.5, colors=WP_SUB)
    ax.grid(axis="x", color=WP_GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)


def fig_llm(acc_rows, reason_rows, lat_rows, out, name="fig_llm"):
    import matplotlib.ticker as mticker
    from matplotlib.patches import Patch
    W, H = 180, 84
    fig = plt.figure(figsize=(W * MM, H * MM))
    top = 0.84
    axa = fig.add_axes([0.165, 0.13, 0.20, top - 0.13]); axb = fig.add_axes([0.535, 0.40, 0.16, top - 0.40]); axc = fig.add_axes([0.835, 0.40, 0.135, top - 0.40])
    lim = lambda rows: (min(r[1] for r in rows) - 5, max(r[1] for r in rows) + 5)
    la, ha_ = lim(acc_rows); lb, hb = lim(reason_rows)
    _hbars(axa, acc_rows, lambda v: f"{v:.1f}", ha_, xmin=la); axa.set_xlim(la, ha_); axa.set_xticks([t for t in range(0, 101, 10) if la <= t <= ha_])
    axa.set_xlabel("Accuracy (%)", color=WP_SUB, **FONT)
    _hbars(axb, reason_rows, lambda v: f"{v:.1f}", hb, xmin=lb); axb.set_xlim(lb, hb); axb.set_xticks([t for t in range(0, 101, 5) if lb <= t <= hb])
    axb.set_xlabel("Accuracy (%)", color=WP_SUB, **FONT)
    _hbars(axc, lat_rows, lambda v: (f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.0f} ms"), 1, log=True)
    axc.set_xscale("log"); axc.set_xlim(20, 150000)
    axc.xaxis.set_major_locator(mticker.FixedLocator([100, 1000, 10000]))
    axc.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: {100: "0.1", 1000: "1", 10000: "10"}.get(int(round(v)), "")))
    axc.xaxis.set_minor_locator(mticker.NullLocator()); axc.set_xlabel("Seconds per question", color=WP_SUB, **FONT)
    yt = top + 0.07
    for x, l, t in ((0.010, "a", "Accuracy on human-labeled questions"), (0.405, "b", "Reasoning sample (1,800 questions)"), (0.765, "c", "Latency")):
        fig.text(x, yt, l, ha="left", va="bottom", fontweight="bold", color=WP_INK, **{**FONT, "size": 9})
        fig.text(x + 0.022, yt, t, ha="left", va="bottom", color=WP_INK, **FONT)
    h = [Patch(color=C_RAD, label="RadKev"), Patch(color=C_KEV, label="Kev"), Patch(color=C_DMO, label="other decision models"),
         Patch(color=C_LLM, label="LLM, no reasoning"), Patch(color=C_LLMR, label="LLM, reasoning")]
    fig.legend(handles=h, loc="lower left", ncol=3, frameon=False, prop={**FONT, "size": 7.5}, bbox_to_anchor=(0.41, 0.05),
               handlelength=1.0, handleheight=0.8, columnspacing=1.2)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf"); fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


def _gbar(fig, ax, items, colors, pad=5, letter=None, title=None, extra=(), delta=True, tick_size=7.2):
    """One grouped-bar panel in the whitepaper style. items: (label, a, b, delta)."""
    import numpy as np
    import matplotlib.transforms as mtrans
    from matplotlib.transforms import offset_copy
    vals = [v for it in items for v in it[1:3]] + list(extra)
    lo, hi = max(0, min(vals) - pad), max(vals) + pad
    w = 0.36; tr = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    for i, (lab, a, b, d) in enumerate(items):
        ax.bar(i - w / 2 - 0.01, a - lo, w, bottom=lo, color=colors[0], zorder=2)
        ax.bar(i + w / 2 + 0.01, b - lo, w, bottom=lo, color=colors[1], zorder=2)
        off = (hi - lo) * 0.012
        ax.text(i - w / 2 - 0.01, a + off, f"{a:.1f}", ha="center", va="bottom", color=WP_SUB, **{**FONT, "size": 6.5})
        ax.text(i + w / 2 + 0.01, b + off, f"{b:.1f}", ha="center", va="bottom", color=WP_INK, **{**FONT, "size": 6.5})
        if not delta: continue
        ds = ("+" if d > 0.05 else "−" if d < -0.05 else "") + f"{abs(d):.1f}"
        ax.text(i, 1.05, ds, transform=tr, ha="center", va="center", color=colors[1] if d > 0 else WP_SUB, fontweight="bold", **{**FONT, "size": 7.5})
    ax.set_xticks(range(len(items))); ax.set_xticklabels([it[0] for it in items], **{**FONT, "size": tick_size}, linespacing=1.1)
    ax.set_ylim(lo, hi); ax.set_xlim(-0.6, len(items) - 0.4)
    step = 1 if hi - lo < 8 else 2 if hi - lo < 16 else 5
    ax.set_yticks([t for t in np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)])
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(axis="y", color=WP_GRID, lw=0.6, zorder=0)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#BDBDBD"); ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(axis="x", length=0, pad=3); ax.tick_params(axis="y", length=0, labelsize=7.2, colors=WP_SUB)
    if letter:
        ax.text(0.0, 1.17, letter, transform=offset_copy(ax.transAxes, fig=fig, x=-22, y=0, units="points"), ha="left", va="bottom",
                fontweight="bold", color=WP_INK, **{**FONT, "size": 9})
        ax.text(0.0, 1.17, title, transform=offset_copy(ax.transAxes, fig=fig, x=-11, y=0, units="points"), ha="left", va="bottom", color=WP_INK, **FONT)


def fig_spec(pa, pb, pc, pd, out, name="fig_spec"):
    """2 x 2. Top: specialization vs scale (a all human-labeled, b radiology). Bottom: starting point (c accuracy, d general ability)."""
    from matplotlib.patches import Patch
    from matplotlib.transforms import offset_copy
    import matplotlib.ticker as mticker
    W, H = 180, 150
    fig = plt.figure(figsize=(W * MM, H * MM))
    CQ = "#E3B98F"
    axa = fig.add_axes([0.09, 0.60, 0.37, 0.25]); axb = fig.add_axes([0.60, 0.60, 0.37, 0.25])
    axc = fig.add_axes([0.09, 0.10, 0.37, 0.25]); axd = fig.add_axes([0.60, 0.10, 0.37, 0.25])
    _gbar(fig, axa, pa, (WP_BASE, WP_ACC)); _gbar(fig, axb, pb, (WP_BASE, WP_ACC)); _gbar(fig, axc, pc, (CQ, WP_ACC))
    for ax in (axa, axb, axc): ax.set_ylabel("Accuracy (%)", color=WP_SUB, **FONT)
    w = 0.36
    for g, (lab, bars) in enumerate(pd):
        for k, (d, lo, hi, c) in enumerate(bars):
            x = g + (k - 0.5) * (w + 0.04)
            if abs(d) < 0.05:   # a zero change: a short line at 0 in the bar colour, so the slot does not read as empty
                axd.plot([x - w / 2, x + w / 2], [0, 0], color=c, lw=2.4, solid_capstyle="butt", zorder=3)
            else:
                axd.bar(x, d, w, color=c, zorder=2)
            axd.errorbar(x, d, yerr=[[d - lo], [hi - d]], fmt="none", ecolor="#4D4D4D", elinewidth=0.8, capsize=2.4, capthick=0.8, zorder=4)
            s_ = ("+" if d > 0.05 else "\u2212" if d < -0.05 else "") + f"{abs(d):.1f}"
            up = d >= 0
            axd.text(x, (hi + 0.7) if up else (lo - 0.7), s_, ha="center", va="bottom" if up else "top", color=WP_INK, **{**FONT, "size": 6.8})
    axd.set_xticks(range(len(pd))); axd.set_xticklabels([p[0] for p in pd], **{**FONT, "size": 7.2})
    axd.set_xlim(-0.6, len(pd) - 0.4)
    allv = [v for _, bars in pd for b_ in bars for v in b_[1:3]]
    axd.set_ylim(min(allv) - 3.5, max(allv) + 3.5); axd.yaxis.set_major_locator(mticker.MultipleLocator(5))
    axd.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+g}".replace("-", "\u2212") if v else "0"))
    axd.axhline(0, color="#9A9A9A", lw=0.7, zorder=1)
    for sp in ("top", "right", "left", "bottom"): axd.spines[sp].set_visible(False)
    axd.grid(axis="y", color=WP_GRID, lw=0.6, zorder=0)
    axd.tick_params(axis="x", length=0, pad=3); axd.tick_params(axis="y", length=0, labelsize=7.2, colors=WP_SUB)
    axd.set_ylabel("Change in accuracy (pp)", color=WP_SUB, **FONT)
    heads = [(axa, "a", "All human-labeled questions", "\u0394 = RadKev \u2212 Kev"),
             (axb, "b", "Radiology questions only", "\u0394 = RadKev \u2212 Kev"),
             (axc, "c", "Accuracy after fine-tuning", "All human-labeled questions; \u0394 = Kev-9B start \u2212 base-model start"),
             (axd, "d", "General ability outside radiology", "Kev's transfer suite, relative to the released Kev-9B")]
    for ax, l, t, sub in heads:
        tr = lambda dx: offset_copy(ax.transAxes, fig=fig, x=dx, y=0, units="points")
        ax.text(0.0, 1.20, l, transform=tr(-30), ha="left", va="bottom", fontweight="bold", color=WP_INK, **{**FONT, "size": 9})
        ax.text(0.0, 1.20, t, transform=tr(-19), ha="left", va="bottom", fontweight="bold", color=WP_INK, **FONT)
        ax.text(0.0, 1.11, sub, transform=tr(-19), ha="left", va="bottom", color=WP_SUB, **{**FONT, "size": 7.2})
    # row headings with one key each
    for y, head, keys in ((0.965, "Specializing a released model vs. scaling it up", [(WP_BASE, "Kev (released)"), (WP_ACC, "RadKev (specialized)")]),
                          (0.475, "Fine-tuning a 9B model from Kev-9B vs. from its base model", [(CQ, "started from Qwen3.5-9B-Base"), (WP_ACC, "started from Kev-9B")])):
        fig.text(0.01, y, head, ha="left", va="center", fontweight="bold", color=WP_INK, **{**FONT, "size": 8.5})
        fig.legend(handles=[Patch(color=c, label=lab) for c, lab in keys], loc="center left", ncol=2, frameon=False, prop={**FONT, "size": 7.5},
                   bbox_to_anchor=(0.55, y), handlelength=1.0, handleheight=0.8, columnspacing=1.2)
    fig.add_artist(plt.Line2D([0.01, 0.99], [0.505, 0.505], color="#DDDDDD", lw=0.6, transform=fig.transFigure))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf"); fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


def fig_calib(curves, cov_rows, ece_rows, out, name="fig_calib"):
    """curves: [(label, color, coverage[], error[] %, lw)]; cov_rows / ece_rows: [(label, as_scored, recalibrated)]."""
    import numpy as np
    from matplotlib.patches import Patch
    from matplotlib.transforms import offset_copy
    W, H = 180, 96
    fig = plt.figure(figsize=(W * MM, H * MM))
    axa = fig.add_axes([0.075, 0.12, 0.40, 0.66])
    axb = fig.add_axes([0.66, 0.58, 0.32, 0.25]); axc = fig.add_axes([0.66, 0.12, 0.32, 0.25])
    for lab, c, cov, err, lw in curves:
        axa.plot(np.array(cov) * 100, err, color=c, lw=lw, label=lab, zorder=3)
    axa.axhline(5, color=WP_SUB, lw=0.7, ls=(0, (3, 2)), zorder=2)
    axa.text(12, 5.4, "5% error budget", color=WP_SUB, va="bottom", ha="left", **{**FONT, "size": 7})
    axa.set_xlim(10, 100); axa.set_ylim(0, max(max(c[3]) for c in curves) + 2)
    axa.set_xlabel("Questions answered, most confident first (%)", color=WP_SUB, **FONT); axa.set_ylabel("Error among answered questions (%)", color=WP_SUB, **FONT)
    axa.legend(loc="upper left", frameon=False, prop={**FONT, "size": 7.2}, handlelength=1.6)
    def hbars(ax, rows, fmt, xlab, pad):
        y = np.arange(len(rows)); h = 0.38
        vals = [v for r in rows for v in r[1:3]]
        lo = max(0, min(vals) - pad); hi = max(vals) + pad
        for i, (lab, a, b) in enumerate(rows):
            ax.barh(i - h / 2 - 0.01, a - lo, h, left=lo, color="#5A6577", zorder=2)
            ax.barh(i + h / 2 + 0.01, b - lo, h, left=lo, color="#B7C0CC", zorder=2)
            ax.text(a + (hi - lo) * 0.012, i - h / 2 - 0.01, fmt(a), va="center", ha="left", color=WP_INK, **{**FONT, "size": 6.5})
            ax.text(b + (hi - lo) * 0.012, i + h / 2 + 0.01, fmt(b), va="center", ha="left", color=WP_SUB, **{**FONT, "size": 6.5})
        ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], **{**FONT, "size": 7.2}); ax.set_ylim(len(rows) - 0.45, -0.55)
        ax.set_xlim(lo, hi + (hi - lo) * 0.08); ax.set_xlabel(xlab, color=WP_SUB, **FONT)
    hbars(axb, cov_rows, lambda v: f"{v:.1f}", "Questions answered at \u22645% error (%)", 5)
    hbars(axc, ece_rows, lambda v: f"{v:.3f}", "Expected calibration error", 0.01)
    for ax in (axa, axb, axc):
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_color("#BDBDBD"); ax.spines[sp].set_linewidth(0.6)
        ax.grid(axis="x" if ax is not axa else "y", color=WP_GRID, lw=0.6, zorder=0); ax.set_axisbelow(True)
        ax.tick_params(length=0, labelsize=7.2, colors=WP_SUB); ax.tick_params(axis="y", colors=WP_INK if ax is not axa else WP_SUB)
    axa.spines["left"].set_visible(False); axb.spines["left"].set_visible(False); axc.spines["left"].set_visible(False)
    heads = [(axa, "a", "Selective prediction", "Human-labeled questions; error as more questions are answered", -30),
             (axb, "b", "Coverage at a 5% error budget", "Human-labeled questions; 27B models", -62),
             (axc, "c", "Calibration error", "Human-labeled questions; 27B models", -62)]
    for ax, l, t, sub, dx in heads:
        tr = lambda d: offset_copy(ax.transAxes, fig=fig, x=d, y=0, units="points")
        yt = 1.19 if ax is axa else 1.30
        ax.text(0.0, yt, l, transform=tr(dx), ha="left", va="bottom", fontweight="bold", color=WP_INK, **{**FONT, "size": 9})
        ax.text(0.0, yt, t, transform=tr(dx + 11), ha="left", va="bottom", fontweight="bold", color=WP_INK, **FONT)
        ax.text(0.0, yt - (0.05 if ax is axa else 0.13), sub, transform=tr(dx + 11), ha="left", va="bottom", color=WP_SUB, **{**FONT, "size": 7.2})
    fig.legend(handles=[Patch(color="#5A6577", label="as scored"), Patch(color="#B7C0CC", label="recalibrated on the test split")],
               loc="upper left", ncol=2, frameon=False, prop={**FONT, "size": 7.5}, bbox_to_anchor=(0.53, 1.0), handlelength=1.0, handleheight=0.8)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf"); fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


def fig_robust(pa, pb, pc, pd, chance, out, name="fig_robust"):
    """2 x 2. Top: case withheld (a Eurorad diagnosis, b MedQA): (label, Kev, RadKev, delta). Bottom: Eurorad gain
    (RadKev-27B minus Kev-27B) by answer space (c) and by whether an answer word occurs in the case (d): (label, d, lo, hi)."""
    from matplotlib.patches import Patch
    from matplotlib.transforms import offset_copy
    import matplotlib.ticker as mticker
    import numpy as np
    W, H = 180, 150
    fig = plt.figure(figsize=(W * MM, H * MM))
    axa = fig.add_axes([0.09, 0.60, 0.33, 0.25]); axb = fig.add_axes([0.59, 0.60, 0.33, 0.25])
    axc = fig.add_axes([0.09, 0.10, 0.50, 0.25]); axd = fig.add_axes([0.72, 0.10, 0.25, 0.25])
    for ax, items, ch in ((axa, pa, chance[0]), (axb, pb, chance[1])):
        vals = [v for it in items for v in it[1:3]] + [ch]
        _gbar(fig, ax, items, (WP_BASE, WP_ACC), extra=(ch,))
        lo, hi = max(0, min(vals) - 5), max(vals) + 5
        ax.set_yticks([t for t in range(0, 101, 10) if lo <= t <= hi])
        ax.axhline(ch, color=WP_SUB, lw=0.7, ls=(0, (3, 2)), zorder=4)
        # chance label in the right margin, level with its line, clear of the bars
        ax.text(1.015, ch, f"chance\n{ch:.0f}%", transform=ax.get_yaxis_transform(), ha="left", va="center", color=WP_SUB,
                linespacing=1.0, clip_on=False, **{**FONT, "size": 6.8})
        ax.set_ylabel("Accuracy (%)", color=WP_SUB, **FONT)
    def dbars(ax, rows, ylab):
        x = np.arange(len(rows))
        for i, (lab, d, lo, hi) in enumerate(rows):
            ax.bar(i, d, 0.55, color=WP_ACC, zorder=2)
            ax.errorbar(i, d, yerr=[[d - lo], [hi - d]], fmt="none", ecolor="#4D4D4D", elinewidth=0.8, capsize=2.4, capthick=0.8, zorder=4)
            s = ("+" if d > 0.05 else "−" if d < -0.05 else "") + f"{abs(d):.1f}"
            ax.text(i, hi + 0.8, s, ha="center", va="bottom", color=WP_INK, **{**FONT, "size": 6.8})
        ax.axhline(0, color="#9A9A9A", lw=0.7, zorder=1)
        ax.set_xticks(x); ax.set_xticklabels([r[0] for r in rows], **{**FONT, "size": 7.0}, linespacing=1.15)
        ax.set_xlim(-0.6, len(rows) - 0.4); ax.set_ylim(min(0, min(r[2] for r in rows)) - 2, max(r[3] for r in rows) + 3.5)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(5))
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:+g}".replace("-", "−") if v else "0"))
        for sp in ("top", "right", "left", "bottom"): ax.spines[sp].set_visible(False)
        ax.grid(axis="y", color=WP_GRID, lw=0.6, zorder=0)
        ax.tick_params(axis="x", length=0, pad=3); ax.tick_params(axis="y", length=0, labelsize=7.2, colors=WP_SUB)
        ax.set_ylabel(ylab, color=WP_SUB, **FONT)
    dbars(axc, pc, "RadKev-27B − Kev-27B (pp)"); dbars(axd, pd, "RadKev-27B − Kev-27B (pp)")
    heads = [(axa, "a", "Eurorad diagnosis", "Δ = RadKev − Kev"), (axb, "b", "MedQA", "Δ = RadKev − Kev"),
             (axc, "c", "Gain by options offered", "Eurorad diagnosis, case available; 95% CI"),
             (axd, "d", "Gain by answer word in case", "Eurorad diagnosis; 95% CI")]
    for ax, l, t, sub in heads:
        tr = lambda dx: offset_copy(ax.transAxes, fig=fig, x=dx, y=0, units="points")
        ax.text(0.0, 1.20, l, transform=tr(-30), ha="left", va="bottom", fontweight="bold", color=WP_INK, **{**FONT, "size": 9})
        ax.text(0.0, 1.20, t, transform=tr(-19), ha="left", va="bottom", fontweight="bold", color=WP_INK, **FONT)
        ax.text(0.0, 1.11, sub, transform=tr(-19), ha="left", va="bottom", color=WP_SUB, **{**FONT, "size": 7.2})
    fig.text(0.01, 0.965, "Options-only control: accuracy with and without the case", ha="left", va="center", fontweight="bold", color=WP_INK, **{**FONT, "size": 8.5})
    fig.legend(handles=[Patch(color=WP_BASE, label="Kev-27B"), Patch(color=WP_ACC, label="RadKev-27B")], loc="center left", ncol=2, frameon=False,
               prop={**FONT, "size": 7.5}, bbox_to_anchor=(0.62, 0.965), handlelength=1.0, handleheight=0.8, columnspacing=1.2)
    fig.text(0.01, 0.475, "Where the Eurorad gain comes from", ha="left", va="center", fontweight="bold", color=WP_INK, **{**FONT, "size": 8.5})
    fig.add_artist(plt.Line2D([0.01, 0.99], [0.505, 0.505], color="#DDDDDD", lw=0.6, transform=fig.transFigure))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf"); fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------- Figure: answer space, all four models
AS_STYLE = {"radkev27": ("RadKev-27B", "#2F6BD8", "-", "o"), "radkev9": ("RadKev-9B (fine-tuned)", "#8FB3F0", "-", "o"),
            "kev27": ("Kev-27B", "#5F6672", "--", "s"), "kev9": ("Kev-9B", "#B3B9C3", "--", "s")}


def fig_answerspace(S, out, name="fig_answer_space"):
    """Rows: Eurorad diagnosis, MedQA (accuracy vs options offered, one panel per distractor source, one line per model);
    bottom: latency per request vs options offered (RadKev-27B, RadKev-9B; median, IQR band)."""
    import numpy as np
    from matplotlib.lines import Line2D
    from matplotlib.transforms import offset_copy
    import matplotlib.ticker as mticker
    W, H = 180, 168
    fig = plt.figure(figsize=(W * MM, H * MM))
    kinds = [("rand", "Random options from the dataset"), ("sim", "Most similar options from the dataset"), ("llm", "Options written by an LLM")]
    srcs = [("eurorad_dx", "Eurorad diagnosis"), ("medqa", "MedQA")]
    xs0 = [0.085, 0.395, 0.705]; wd = 0.27; rows_y = [0.655, 0.385]; hh = 0.19
    letters = iter("abcdefgh")
    for r, (src, src_name) in enumerate(srcs):
        vals = [100 * v["acc"] for (m, s, k, K), v in S["acc"].items() if s == src and K is not None]
        lo, hi = min(vals) - 5, min(100, max(vals)) + 5
        for c, (kind, kname) in enumerate(kinds):
            ax = fig.add_axes([xs0[c], rows_y[r], wd, hh])
            for m, (lab, col, ls, mk) in AS_STYLE.items():
                pts = sorted((K, 100 * v["acc"]) for (mm, s, k, K), v in S["acc"].items() if mm == m and s == src and k == kind and K)
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=col, ls=ls, marker=mk, ms=3.0, lw=1.2, zorder=3)
            ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16, 32, 64]); ax.set_xticklabels(["2", "4", "8", "16", "32", "64"])
            ax.xaxis.set_minor_locator(mticker.NullLocator()); ax.set_xlim(1.7, 75); ax.set_ylim(lo, hi)
            ax.grid(axis="y", color=WP_GRID, lw=0.6, zorder=0)
            for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color("#BDBDBD"); ax.spines["bottom"].set_linewidth(0.6)
            ax.tick_params(length=0, labelsize=7.2, colors=WP_SUB)
            if c == 0: ax.set_ylabel(f"{src_name}\naccuracy (%)", color=WP_INK, **FONT)
            else: ax.set_yticklabels([])
            ax.set_xlabel("Options offered", color=WP_SUB, **{**FONT, "size": 7.2})
            if r == 0: fig.text(xs0[c], rows_y[0] + hh + 0.045, kname, ha="left", va="bottom", fontweight="bold", color=WP_INK, **{**FONT, "size": 7.8})
            tr = lambda dx: offset_copy(ax.transAxes, fig=fig, x=dx, y=0, units="points")
            ax.text(0.0, 1.04, next(letters), transform=tr(-14), ha="left", va="bottom", fontweight="bold", color=WP_INK, **{**FONT, "size": 9})
    # latency
    ax = fig.add_axes([0.075, 0.07, 0.86, 0.15])
    for m, lab, col in (("radkev27", "RadKev-27B", "#2F6BD8"), ("radkev9", "RadKev-9B", "#8FB3F0")):
        pts = sorted((K, v) for (mm, K), v in S["lat"].items() if mm == m)
        K = [p[0] for p in pts]
        ax.fill_between(K, [p[1]["q1"] for p in pts], [p[1]["q3"] for p in pts], color=col, alpha=0.18, lw=0, zorder=2)
        ax.plot(K, [p[1]["median"] for p in pts], color=col, marker="o", ms=3.0, lw=1.4, zorder=3, label=lab)
        ax.text(K[-1] * 1.08, pts[-1][1]["median"], f"{pts[-1][1]['median']:.0f} ms", va="center", ha="left", color=WP_INK, **{**FONT, "size": 6.8})
    ax.set_xscale("log", base=2); ax.set_xticks([2, 4, 8, 16, 32, 64, 128]); ax.set_xticklabels(["2", "4", "8", "16", "32", "64", "128"])
    ax.xaxis.set_minor_locator(mticker.NullLocator()); ax.set_xlim(1.7, 200); ax.set_ylim(0, None)
    ax.grid(axis="y", color=WP_GRID, lw=0.6, zorder=0)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#BDBDBD"); ax.spines["bottom"].set_linewidth(0.6)
    ax.tick_params(length=0, labelsize=7.2, colors=WP_SUB)
    ax.set_xlabel("Options offered", color=WP_SUB, **{**FONT, "size": 7.2}); ax.set_ylabel("Latency per request (ms)", color=WP_SUB, **FONT)
    ax.legend(loc="upper left", frameon=False, prop={**FONT, "size": 7.2}, handlelength=1.6)
    tr = lambda dx: offset_copy(ax.transAxes, fig=fig, x=dx, y=0, units="points")
    ax.text(0.0, 1.10, "g", transform=tr(-14), ha="left", va="bottom", fontweight="bold", color=WP_INK, **{**FONT, "size": 9})
    ax.text(0.0, 1.10, "Latency per request, one question per request (line, median; band, interquartile range)", transform=tr(-3), ha="left", va="bottom", fontweight="bold", color=WP_INK, **{**FONT, "size": 7.8})
    # headings and key
    fig.text(0.01, 0.955, "Accuracy as alternatives are added to the correct answer", ha="left", va="center", fontweight="bold", color=WP_INK, **{**FONT, "size": 8.5})
    h = [Line2D([], [], color=c, ls=ls, marker=mk, ms=3.0, lw=1.2, label=lab) for lab, c, ls, mk in AS_STYLE.values()]
    fig.legend(handles=h, loc="center left", ncol=4, frameon=False, prop={**FONT, "size": 7.5}, bbox_to_anchor=(0.50, 0.955), handlelength=2.0, columnspacing=1.1)
    fig.add_artist(plt.Line2D([0.01, 0.99], [0.295, 0.295], color="#DDDDDD", lw=0.6, transform=fig.transFigure))
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf"); fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)


SIMPLE = {"radkev27": ("RadKev-27B (fine-tuned)", "#2F6BD8", "-", "o", 1.7),
          "kev27": ("Kev-27B (released)", "#6B7280", "--", "s", 1.3),
          "qwen38": ("Qwen3.8-27B (LLM)", "#E8833A", "-", "^", 1.3),
          "medgemma_fix": ("MedGemma-27B-text (LLM)", "#B4561A", ":", "v", 1.3),
          "radkev9": ("RadKev-9B (fine-tuned)", "#8FB3F0", "-", "o", 1.3)}


def fig_answerspace_simple(acc, lat, out, name="fig_answer_space"):
    """acc: {src: {model: [(K, acc%)]}}; lat: {model: [(K, median, q1, q3)]}."""
    import matplotlib.ticker as mticker
    from matplotlib.transforms import offset_copy
    from matplotlib.lines import Line2D
    W, H = 180, 76
    fig = plt.figure(figsize=(W * MM, H * MM))
    axs = [fig.add_axes([x, 0.17, 0.26, 0.54]) for x in (0.065, 0.395, 0.725)]
    for ax, (src, title) in zip(axs[:2], (("eurorad_dx", "Eurorad diagnosis"), ("medqa", "MedQA"))):
        vals = [v for m in acc[src] for _, v in acc[src][m]]
        for m in ("medgemma_fix", "qwen38", "kev27", "radkev27"):
            if m not in acc[src]: continue
            lab, c, ls, mk, lw = SIMPLE[m]; pts = sorted(acc[src][m])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, ls=ls, marker=mk, ms=3.2, lw=lw, zorder=3)
        ax.set_ylim(min(vals) - 5, min(100, max(vals)) + 5); ax.set_ylabel("Accuracy (%)", color=WP_SUB, **FONT)
        ax.text(0, 1.05, title, transform=offset_copy(ax.transAxes, fig=fig, x=0, y=0, units="points"), ha="left", va="bottom", color=WP_INK, **FONT)
    ax = axs[2]
    for m in ("medgemma_fix", "qwen38", "radkev27", "radkev9"):
        if m not in lat: continue
        lab, c, ls, mk, lw = SIMPLE[m]; pts = sorted(lat[m])
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=c, ls=ls, marker=mk, ms=3.2, lw=lw, zorder=3)
    ax.set_ylim(0, None); ax.set_ylabel("Latency per request (ms)", color=WP_SUB, **FONT)
    ax.text(0, 1.05, "Latency, one question per request", transform=offset_copy(ax.transAxes, fig=fig, x=0, y=0, units="points"), ha="left", va="bottom", color=WP_INK, **FONT)
    for ax, l in zip(axs, "abc"):
        ax.set_xscale("log", base=2)
        ks = [2, 4, 8, 16, 32, 64] if ax is not axs[2] else [2, 4, 8, 16, 32, 64, 128]
        ax.set_xticks(ks); ax.set_xticklabels([str(k) for k in ks]); ax.xaxis.set_minor_locator(mticker.NullLocator())
        ax.set_xlim(1.7, ks[-1] * 1.18); ax.set_xlabel("Options offered", color=WP_SUB, **FONT)
        ax.grid(axis="y", color=WP_GRID, lw=0.6, zorder=0)
        for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color("#BDBDBD"); ax.spines["bottom"].set_linewidth(0.6)
        ax.tick_params(length=0, labelsize=7.2, colors=WP_SUB)
        ax.text(0, 1.05, l, transform=offset_copy(ax.transAxes, fig=fig, x=-14, y=0, units="points"), ha="left", va="bottom", fontweight="bold", color=WP_INK, **{**FONT, "size": 9})
    h = [Line2D([], [], color=SIMPLE[m][1], ls=SIMPLE[m][2], marker=SIMPLE[m][3], ms=3.2, lw=SIMPLE[m][4], label=SIMPLE[m][0])
         for m in ("radkev27", "kev27", "qwen38", "medgemma_fix", "radkev9")]
    fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, prop={**FONT, "size": 7.2}, bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.6)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf"); fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)
