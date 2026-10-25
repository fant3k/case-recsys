"""Report figures from saved result JSON only (no training, no new evaluation).

Usage: python scripts/make_report_figures.py [--out report/figures]
Every number is read from artifacts/*/report.json, artifacts/*/status.json or research/results/*.json.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"   # validated categorical order
NEUTRAL = "#b8b6ae"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "legend.frameon": False, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def load(path):
    """artifacts/ is not in Git; its small JSON copies live in research/results/runs/."""
    file = ROOT / path
    if not file.exists() and path.startswith("artifacts/"):
        file = ROOT / "research/results/runs" / path[len("artifacts/"):]
    return json.loads(file.read_text(encoding="utf-8"))


COMMA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))


def finish(fig, title=None):
    """Long titles go to the figure, not the axes, so they are never clipped."""
    if title:
        fig.suptitle(title, x=0.01, ha="left", fontsize=11, fontweight="bold")


def save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, dpi=180)
    plt.close(fig)
    print("wrote", out / name)


def fig_methods(out):
    rer = load("artifacts/reranker-20261022-02/report.json")
    sft = load("artifacts/sft-20261021-01/status.json")
    pps = load("artifacts/pps-20261022-01/report.json")["arms"]
    rows = [
        ("MostPop", sft["baselines"]["MostPop"]["NDCG@10"], NEUTRAL),
        ("GPT-2, исходный SFT (одна цель, 12 мин)", sft["best_selection_ndcg"], BLUE),
        ("GPT-2, все позиции (10 мин)", pps["full-d0.1-wd0.01"]["history"][-1]["NDCG@10"], BLUE),
        ("GPT-2 + PPS + отклик (10 мин)", pps["full+pps+fb-d0.1-wd0.01"]["history"][-1]["NDCG@10"], BLUE),
        ("Личная частота, 128 событий", rer["baselines"]["PersonalFrequency128"]["NDCG@10"], NEUTRAL),
        ("Смесь недавней и полной частоты", rer["baselines"]["Memory_recent_full_0.5"]["NDCG@10"], NEUTRAL),
        ("Реранкер LightGBM", rer["rerankers"]["with_gpt"]["main_gpt_features"]["NDCG@10"], AQUA),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    y = range(len(rows))[::-1]
    ax.barh(list(y), [r[1] for r in rows], height=0.6, color=[r[2] for r in rows])
    for yy, (_, v, _) in zip(y, rows):
        ax.text(v + 0.003, yy, f"{v:.3f}".replace(".", ","), va="center", color=INK2, fontsize=9)
    ax.set_yticks(list(y), [r[0] for r in rows])
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.set_xlabel("NDCG@10, selection, 492 активных пользователя")
    ax.set_xlim(0, 0.27)
    ax.grid(axis="y", visible=False)
    finish(fig, "Качество рекомендаций: GPT-2, простые ориентиры и реранкер")
    ax.xaxis.set_major_formatter(COMMA)
    save(fig, out, "fig1_methods_ndcg.png")


def fig_training(out):
    full = load("artifacts/fullsft-20261022-01/report.json")
    pps = load("artifacts/pps-20261022-01/report.json")
    freq = pps["PersonalFrequency128"]["NDCG@10"]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    panels = [
        (axes[0], full["arms"], [("last", "одна цель на пример", INK2), ("full", "все позиции", ORANGE)],
         "А. Одна цель против всех позиций (25 мин)"),
        (axes[1], pps["arms"], [("full-d0.1-wd0.01", "все позиции", ORANGE), ("full+pps-d0.1-wd0.01", "+ PPS", AQUA),
                                ("full+pps+fb-d0.1-wd0.01", "+ PPS + отклик", BLUE),
                                ("full+pps+fb+pos-d0.1-wd0.01", "+ PPS + отклик + цели >50%", YELLOW)],
         "Б. Лесенка PPS и отклика (10 мин)"),
    ]
    for ax, arms, series, title in panels:
        for key, label, color in series:
            h = arms[key]["history"]
            x = [e["training_seconds"] / 60 for e in h]
            ax.plot(x, [e["NDCG@10"] for e in h], color=color, linewidth=2, marker="o", markersize=4, label=label)
        ax.axhline(freq, color=MUTED, linewidth=1)
        ax.text(0.2, freq + 0.004, "личная частота 0,181", color=INK2, fontsize=8.5)
        ax.set_title(title)
        ax.set_xlabel("минуты обучения")
        ax.legend(fontsize=8.5, loc="lower right")
    axes[0].set_ylabel("NDCG@10, selection")
    axes[0].set_ylim(0, 0.2)
    for ax in axes:
        ax.xaxis.set_major_formatter(COMMA); ax.yaxis.set_major_formatter(COMMA)
    save(fig, out, "fig2_gpt2_training.png")


def fig_gap(out):
    g = load("artifacts/gap-diagnostics-20261022-02/report.json")
    rows = []
    for v in sorted(g["in_sample_train_days"].values(), key=lambda v: -v["days_before_train_end"]):
        rows.append((f"train, за {v['days_before_train_end']} дн. до конца" if v["days_before_train_end"] else "train, последний день",
                     v["GPT2_NDCG@10"], v["Frequency128_NDCG@10"]))
    s = g["selection"]
    rows.append(("selection (новый день)", s["GPT2_NDCG@10"], s["Frequency128_NDCG@10"]))
    o = g["ordering_within_recent128"]
    rows.append(("selection, только треки из истории", o["GPT2_order_NDCG@10"], o["Frequency_order_NDCG@10"]))
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    y = list(range(len(rows)))[::-1]
    for yy, (_, a, b) in zip(y, rows):
        ax.plot([a, b], [yy, yy], color=AXIS, linewidth=2, zorder=1)
    ax.scatter([r[1] for r in rows], y, s=64, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=2, label="GPT-2 (все позиции)")
    ax.scatter([r[2] for r in rows], y, s=64, color=ORANGE, edgecolor=SURFACE, linewidth=2, zorder=2, label="личная частота")
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("NDCG@10")
    ax.set_xlim(0.1, 0.25)
    ax.legend(loc="lower right", fontsize=9)
    finish(fig, "GPT-2 проигрывает частоте и на днях, на которых обучалась")
    ax.xaxis.set_major_formatter(COMMA)
    save(fig, out, "fig3_gap_diagnostics.png")


def fig_hits(out):
    rer = load("artifacts/reranker-20261022-02/report.json")["slices"]
    pps = load("artifacts/pps-20261022-01/report.json")["arms"]["full-d0.1-wd0.01"]["slices_at_best"]["slices"]
    rows = [("GPT-2 (все позиции)", pps), ("Личная частота", rer["PersonalFrequency128"]["slices"]),
            ("Смесь частоты", rer["Memory_recent_full_0.5"]["slices"]), ("Реранкер", rer["with_gpt/main_gpt_features"]["slices"])]
    cats = [("seen_recent128", "в последних 128 событиях", BLUE), ("seen_older_only", "только в старой истории", ORANGE),
            ("never_seen", "новые для пользователя", AQUA)]
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    y = list(range(len(rows)))[::-1]
    left = [0] * len(rows)
    for key, label, color in cats:
        vals = [r[1][key]["hits"] for r in rows]
        ax.barh(y, vals, left=left, height=0.6, color=color, edgecolor=SURFACE, linewidth=2, label=label)
        left = [a + b for a, b in zip(left, vals)]
    for yy, total in zip(y, left):
        ax.text(total + 8, yy, str(total), va="center", color=INK2, fontsize=9)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("попаданий в завтрашние цели (сумма по 492 пользователям)")
    ax.set_xlim(0, max(left) * 1.08)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=3, fontsize=8.5)
    finish(fig, "Откуда берутся попадания: повторы, старые любимые, новые треки")
    save(fig, out, "fig4_hit_sources.png")


def fig_reward(out):
    r = load("artifacts/reward-v2-20261022-01/report.json")["models"]
    names = [("constant_train_mean", "константа"), ("item_train_mean", "среднее по треку"), ("v1_history_RM", "RM v1"),
             ("v2_no_features_seed0", "RM v2 без признаков"), ("v2_ensemble_mean", "RM v2, ансамбль")]
    colors = [NEUTRAL, NEUTRAL, ORANGE, BLUE, BLUE]
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for ax, metric, title, xlabel, lim in [
        (axes[0], lambda m: m["overall"]["MAE_event"], "А. Ошибка (меньше — лучше)", "MAE доли дослушивания", (0, 0.45)),
        (axes[1], lambda m: m["pair_accuracy"]["macro_pair_accuracy"], "Б. Порядок треков одного пользователя",
         "PairAccuracy (0,5 — случайно)", (0.45, 0.58)),
    ]:
        vals = [metric(r[k]) for k, _ in names]
        y = list(range(len(names)))[::-1]
        ax.barh(y, [v - lim[0] for v in vals], left=lim[0], height=0.6, color=colors)
        for yy, v in zip(y, vals):
            ax.text(v + (lim[1] - lim[0]) * 0.015, yy, f"{v:.3f}".replace(".", ","), va="center", color=INK2, fontsize=9)
        ax.set_yticks(y, [n for _, n in names] if ax is axes[0] else [""] * len(names))
        ax.tick_params(axis="y", colors=INK, length=0)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(*lim)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
    axes[1].axvline(0.5, color=MUTED, linewidth=1)
    for ax in axes:
        ax.xaxis.set_major_formatter(COMMA)
    save(fig, out, "fig5_reward_models.png")


def fig_rl_v1(out):
    m = load("research/results/rl-comparison-2026-10-21.json")["models"]
    ref = m["reference"]
    def gain(model, key):
        w, w0 = model["all_warm_query_reward"], ref["all_warm_query_reward"]
        return 100 * (w[key] / w0[key] - 1)
    def dndcg(model):
        return 100 * (model["ranking"]["NDCG@10"] / ref["ranking"]["NDCG@10"] - 1)
    points = []
    for name, model in m.items():
        if name == "reference":
            continue
        seconds = "seconds" in name
        key = "expected_primary_equivalent_seconds" if seconds else "expected_primary_completion"
        alt = "expected_alternate_equivalent_seconds" if seconds else "expected_alternate_completion"
        method = "analytic" if name.startswith("analytic") else ("sft" if name == "continued_SFT" else "rl")
        beta = name.split("beta")[-1].replace("_seed2", " (повтор)") if "beta" in name else ""
        points.append((name, method, seconds, gain(model, key), gain(model, alt), dndcg(model), beta))
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    ax = axes[0]
    for _, method, seconds, g, _, d, beta in points:
        color = NEUTRAL if method == "sft" else (ORANGE if seconds else BLUE)
        marker = {"rl": "o", "analytic": "D", "sft": "s"}[method]
        ax.scatter(g, d, s=70, color=color, marker=marker, edgecolor=SURFACE, linewidth=2, zorder=3)
        if beta and d < -3 and not (method == "analytic" and seconds and "0.03" in beta):
            ax.annotate(f"β={beta.replace('.', ',')}", (g, d), xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK2)
    ax.axhline(0, color=AXIS, linewidth=1); ax.axvline(0, color=AXIS, linewidth=1)
    ax.axhline(-5, color=MUTED, linewidth=1)
    ax.text(11.8, -4.4, "допуск −5% NDCG", fontsize=8, color=INK2, ha="right")
    ax.text(0.5, 1.2, "остальные RL и оптимум β=0,3: сдвиг < 3%", fontsize=8, color=INK2)
    ax.set_xlim(-1, 12); ax.set_ylim(-35, 3)
    ax.set_xlabel("рост предсказанной награды, % (обучающая RM)")
    ax.set_ylabel("изменение NDCG@10, %")
    ax.set_title("А. RL v1: награда против релевантности")
    ax.text(11.8, -20, "вне шкалы: оптимум «секунды», β=0,03 —\nнаграда +185%, NDCG −94%", ha="right",
            fontsize=7.5, color=INK2)
    ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, linestyle="", label="RL, доля дослушивания"),
                       Line2D([], [], marker="o", color=ORANGE, linestyle="", label="RL, секунды"),
                       Line2D([], [], marker="D", color=MUTED, linestyle="", label="аналитический оптимум"),
                       Line2D([], [], marker="s", color=NEUTRAL, linestyle="", label="продолжение SFT")],
              fontsize=8, loc="lower left")
    ax = axes[1]
    shown = [p for p in points if p[1] != "sft" and not (p[1] == "analytic" and p[2] and "0.03" in p[6])]
    shown.sort(key=lambda p: p[3])
    y = list(range(len(shown)))
    for yy, (_, method, seconds, g, a, _, beta) in zip(y, shown):
        ax.plot([a, g], [yy, yy], color=AXIS, linewidth=2, zorder=1)
    ax.scatter([p[3] for p in shown], y, s=60, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=2, label="обучающая RM")
    ax.scatter([p[4] for p in shown], y, s=90, facecolor="none", edgecolor=ORANGE, linewidth=2, zorder=3, label="альтернативная RM")
    labels = [f"{'оптимум' if p[1] == 'analytic' else 'RL'}, {'секунды' if p[2] else 'дослуш.'}, β={p[6].replace('.', ',')}" for p in shown]
    ax.set_yticks(y, labels, fontsize=8)
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_xlabel("рост предсказанной награды, %")
    ax.set_title("Б. Та же политика по двум RM")
    ax.legend(fontsize=8, loc="lower right")
    for a in axes:
        a.xaxis.set_major_formatter(COMMA)
    axes[0].yaxis.set_major_formatter(COMMA)
    save(fig, out, "fig6_rl_v1_tradeoff.png")


def fig_rl_v2(out):
    p = load("artifacts/rl-v2-20261023-01/report.json")["policies"]
    ref = p["reference"]
    def point(name):
        seconds = "/seconds/" in name
        key = "judge_seconds_norm" if seconds else "judge_completion"
        gain = 100 * (p[name]["expected_reward"][key] / ref["expected_reward"][key] - 1)
        return gain, 100 * (p[name]["ranking"]["NDCG@10"] / ref["ranking"]["NDCG@10"] - 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2))
    ax = axes[0]
    off = []
    for name in p:
        parts = name.split("/")
        if parts[0] not in ("rl", "analytic") or parts[1] == "completion_lcb":
            continue
        g, d = point(name)
        beta = parts[2].replace("beta", "").replace(".", ",")
        if g > 45 or d < -15:
            off.append(f"{'оптимум' if parts[0] == 'analytic' else 'RL'}, {'секунды' if parts[1] == 'seconds' else 'дослуш.'}, β={beta}: "
                       f"награда {g:+.0f}%, NDCG {d:+.0f}%".replace("-", "−"))
            continue
        color = ORANGE if parts[1] == "seconds" else BLUE
        ax.scatter(g, d, s=70, color=color, marker="D" if parts[0] == "analytic" else "o", edgecolor=SURFACE, linewidth=2, zorder=3)
        if abs(d) > 1.5:  # points near zero are explained in the notes below
            ax.annotate(f"β={beta}", (g, d), xytext=(5, 4) if parts[0] == "rl" else (6, -11), textcoords="offset points",
                        fontsize=7.5, color=INK2)
    ax.axhline(0, color=AXIS, linewidth=1); ax.axvline(0, color=AXIS, linewidth=1)
    ax.axhline(-5, color=MUTED, linewidth=1)
    ax.annotate("секунды, β=0,3", (7.3, -0.6), xytext=(9, -2.6), fontsize=7.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.8))
    ax.text(9, 1.6, "у нуля: β=0,3–1 и RL дослуш. β=0,03–0,1", fontsize=7.5, color=INK2)
    ax.text(44, -4.5, "допуск −5% NDCG", fontsize=8, color=INK2, ha="right")
    ax.set_xlim(-6, 45); ax.set_ylim(-15, 10)
    ax.set_xlabel("рост предсказанной награды по судье, %")
    ax.set_ylabel("изменение NDCG@10, %")
    ax.set_title("А. RL v2: награда против релевантности")
    ax.text(44, -14.3, "вне шкалы:\n" + "\n".join(off), fontsize=7, color=INK2, ha="right", va="bottom")
    ax.legend(handles=[Line2D([], [], marker="o", color=BLUE, linestyle="", label="RL, доля дослушивания"),
                       Line2D([], [], marker="o", color=ORANGE, linestyle="", label="RL, секунды"),
                       Line2D([], [], marker="D", color=MUTED, linestyle="", label="аналитический оптимум")],
                   fontsize=8, loc="upper right")
    v3 = load("artifacts/rl-v3-20261023-01/report.json")["policies"]
    base = v3["reference"]["ranking"]["NDCG@10"]
    rows = [("оптимум, весь каталог", "analytic/completion/beta0.03/full", NEUTRAL)]
    rows += [(f"оптимум, только топ-{k}", f"analytic/completion/beta0.03/top{k}", NEUTRAL) for k in (20, 50, 200)]
    rows += [("обратная KL, 8192 контекста", "reverse/small/completion/beta0.03", BLUE),
             ("обратная KL, 65 536", "reverse/pool/completion/beta0.03", BLUE),
             ("прямая KL, 65 536", "forward/pool/completion/beta0.03", BLUE),
             ("Джеффрис, 65 536", "jeffreys/pool/completion/beta0.03", BLUE),
             ("прямая KL, эмбеддинги заморожены", "forward/pool/frozen/completion/beta0.03", BLUE)]
    ax = axes[1]
    y = list(range(len(rows)))[::-1]
    for yy, (label, key, color) in zip(y, rows):
        d = 100 * (v3[key]["ranking"]["NDCG@10"] / base - 1)
        ax.barh(yy, d, color=color, height=0.6)
        share = v3[key].get("share_of_analytic_gain")
        text = f"{d:+.1f}%".replace(".", ",").replace("-", "−") + (f"  доля J {share:.2f}".replace(".", ",") if share is not None else "")
        ax.text(max(d, 0) + 0.2, yy, text, va="center", fontsize=7.5, color=INK2)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=8)
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=AXIS, linewidth=1)
    ax.set_xlim(-1.5, 11)
    ax.set_xlabel("изменение NDCG@10, % (дослушивание, β=0,03)")
    ax.set_title("Б. RL v3: оптимум против обучения")
    for a in axes:
        a.xaxis.set_major_formatter(COMMA)
    axes[0].yaxis.set_major_formatter(COMMA)
    save(fig, out, "fig7_rl_v2_v3.png")


def fig_final(out):
    r = load("artifacts/final-test-20261023-01/report.json")
    names = {"Reranker_LightGBM": "реранкер LightGBM", "Memory_recent_full_0.5": "память 50/50",
             "Frequency128_recency_tiebreak": "частота + давность", "PersonalFrequency128": "личная частота",
             "A_rm_rerank_top50": "A: GPT-2 + RM, топ-50", "B_rl_seconds_beta0.3": "B: RL секунды β=0,3",
             "reference": "GPT-2 + PPS (опорная)", "GPT2_original_next_item_day": "GPT-2 исходная"}
    order = sorted(names, key=lambda n: r["rows"][n]["ranking"]["NDCG@10"])
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), gridspec_kw={"width_ratios": [1.15, 1]})
    ax = axes[0]
    for yy, n in enumerate(order):
        v = r["rows"][n]["ranking"]["NDCG@10"]
        color = BLUE if n.startswith("A_") else ORANGE if n.startswith("B_") else INK2 if n == "reference" else NEUTRAL
        ax.barh(yy, v, color=color, height=0.6)
        ax.text(v + 0.003, yy, f"{v:.3f}".replace(".", ","), va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(order)), [names[n] for n in order], fontsize=8)
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, 0.24)
    ax.set_xlabel(f"NDCG@10, {r['active_positive_users']} активных пользователей")
    ax.set_title("А. Финальное окно, все строки")
    ax = axes[1]
    c = r["confirmatory"]
    rows = [("H1: NDCG A − опорная", c["H1_ndcg_delta_A_minus_reference"], BLUE),
            ("H2: NDCG B − 0,95·опорная", c["H2_margin_B_minus_0.95_reference"], ORANGE),
            ("справочно: NDCG B − опорная", c["H2_ndcg_delta_B_minus_reference"], NEUTRAL)]
    for yy, (label, v, color) in zip([2, 1, 0], rows):
        lo, hi = v["interval"]
        ax.plot([lo, hi], [yy, yy], color=color, linewidth=3, solid_capstyle="round")
        ax.scatter(v["mean"], yy, s=60, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set_yticks([2, 1, 0], [r_[0] for r_ in rows], fontsize=8)
    ax.tick_params(axis="y", colors=INK, length=0)
    ax.grid(axis="y", visible=False)
    ax.axvline(0, color=INK2, linewidth=1)
    s = c["H2_judge_equivalent_seconds_delta"]
    ax.text(0.99, 0.04, f"H2: секунды по судье B − опорная\n{s['mean']:+.2f} с [{s['interval'][0]:.2f}; {s['interval'][1]:.2f}]".replace(".", ","),
            transform=ax.transAxes, ha="right", fontsize=8, color=INK2)
    ax.set_xlabel("разница, 97,5% интервал (Бонферрони)")
    ax.set_title("Б. Подтверждающие гипотезы")
    for a in axes:
        a.xaxis.set_major_formatter(COMMA)
    save(fig, out, "fig8_final_test.png")


def fig_multistep(out):
    r = load("artifacts/multistep-20261025-01/report.json")["rollouts"]
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.7))
    steps = [1, 2, 3, 4, 5]
    ax = axes[0]
    for feedback, color, label in (("listened", BLUE, "«дослушал»"), ("simulated", AQUA, "прогноз судьи"), ("skipped", ORANGE, "«пропустил»")):
        rows = r[f"reference/greedy/{feedback}"]["steps"]
        ax.plot(steps, [s["judge_completion_imagined"] for s in rows], color=color, marker="o", linewidth=2, label=label)
    ax.plot(steps, [s["judge_completion_real_context"] for s in r["reference/greedy/listened"]["steps"]], color=INK2,
            linestyle="--", linewidth=1.5, label="те же треки, реальная история")
    ax.set_xticks(steps); ax.set_xlabel("шаг генерации"); ax.set_ylabel("прогноз доли дослушивания (судья)")
    ax.set_title("А. Оценка зависит от отклика")
    ax.legend(fontsize=7.5, loc="lower left")
    ax = axes[1]
    for decoding, style in (("greedy", "-"), ("sample", ":")):
        rows = r[f"reference/{decoding}/simulated"]["steps"]
        ax.plot(steps[1:], [100 * s["repeat_of_earlier_generated"] for s in rows[1:]], color=ORANGE, linestyle=style, marker="o",
                linewidth=2, label=f"повтор уже сгенерированного, {'жадно' if decoding == 'greedy' else 'сэмплирование'}")
        ax.plot(steps, [100 * s["top10_overlap_with_step1"] for s in rows], color=BLUE, linestyle=style, marker="s", linewidth=2,
                label=f"общий топ-10 с шагом 1, {'жадно' if decoding == 'greedy' else 'сэмплирование'}")
    ax.set_xticks(steps); ax.set_xlabel("шаг генерации"); ax.set_ylabel("%"); ax.set_ylim(0, 160); ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("Б. Петля повторов")
    ax.legend(fontsize=7, loc="upper center")
    ax = axes[2]
    names = [("reference", "опорная GPT-2", INK2), ("A_rm_rerank_top50", "A: + RM, топ-50", BLUE),
             ("B_rl_seconds_beta0.3", "B: RL секунды β=0,3", ORANGE), ("rl_seconds_beta0.1", "RL секунды β=0,1", YELLOW)]
    for name, label, color in names:
        rows = r[f"{name}/sample/simulated"]["steps"]
        ax.plot(steps, [s["mean_capped_length_seconds"] for s in rows], color=color, marker="o", linewidth=2, label=label)
    ax.set_xticks(steps); ax.set_xlabel("шаг генерации"); ax.set_ylabel("средняя длина трека, с")
    ax.set_title("В. Длина треков по шагам")
    ax.legend(fontsize=7.5, loc="upper left")
    for a in axes:
        a.yaxis.set_major_formatter(COMMA)
    save(fig, out, "fig9_multistep.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="report/figures")
    out = ROOT / parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_methods, fig_training, fig_gap, fig_hits, fig_reward, fig_rl_v1, fig_rl_v2, fig_final, fig_multistep):
        f(out)


if __name__ == "__main__":
    main()
