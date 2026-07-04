#!/usr/bin/env python3
"""_full_search aggregator: success-vs-sims & success-vs-time curves (variance bands) + a/b/c/d.

Consumes the INSTRUMENTED time_bestfirst leaf jsonls (fields: n_sim, t_wall, solved, depth_hist, solve_ranks,
model, tier, xml, object_id). Each --model-dir is one model ckpt-seed; each --random-dir is one rng-seed.

Curves (best-first explores in a budget-INDEPENDENT order, so one budget-900 run records the exact sim index
& wall-time at which each instance solved -> solve@B = frac(solved & n_sim<=B), solve@T = frac(solved & t_wall<=T)):
  * success-vs-sims : mean +/- std of per-seed curves  (machine-INDEPENDENT; random band n=10, model band n=3)
  * success-vs-time : same over t_wall               (fair only because all seeds ran emeraldrapids-exclusive)
Aggregations:
  (a) breadth:depth  = total sims at push-depth 0 (first push, root) : total at depth 1 (second push, dive)
  (b) histogram of sims by tree-depth (where the budget goes)
  (c) solve_ranks    : priority-rank of the winning plan's pushes (is the model's #1 the winner?)
  (d) solved-but-slow: per-instance count where the MODEL uses MORE sims than random (ranking actively hurts)
"""
import os, sys, json, glob, argparse
from collections import defaultdict
import numpy as np

SIM_CUTS = [2, 10, 30, 100, 300, 900]
SIM_GRID = list(range(1, 901))


def load_seed(d, model_filter):
    rows = []
    for f in glob.glob(os.path.join(d, "shard_*.jsonl")):
        for l in open(f):
            if not l.strip():
                continue
            r = json.loads(l)
            if model_filter is None or r.get("model") == model_filter:
                rows.append(r)
    return rows


def _base(x):
    return os.path.basename(x)


def key_of(r):
    return (_base(r["xml"]), r["object_id"])


def sims_curve(rows, grid):
    n = len(rows)
    solved = [(r["n_sim"] if r.get("n_sim") is not None else r.get("sims")) for r in rows if r.get("solved")]
    solved = np.asarray(solved)
    return np.array([100.0 * np.count_nonzero(solved <= b) / n for b in grid]) if n else np.zeros(len(grid))


def time_curve(rows, grid):
    n = len(rows)
    solved_t = np.asarray([r["t_wall"] for r in rows if r.get("solved")])
    return np.array([100.0 * np.count_nonzero(solved_t <= t) / n for t in grid]) if n else np.zeros(len(grid))


def band(curves):
    M = np.vstack(curves)
    return M.mean(0), M.std(0)


def at_cut(rows, kind, cut):
    n = len(rows)
    if not n:
        return 0.0
    if kind == "sim":
        s = [(r.get("n_sim") if r.get("n_sim") is not None else r.get("sims")) for r in rows if r.get("solved")]
        return 100.0 * sum(1 for v in s if v <= cut) / n
    s = [r["t_wall"] for r in rows if r.get("solved")]
    return 100.0 * sum(1 for v in s if v <= cut) / n


def depth_totals(rows):
    """(a)/(b): sum sims by push-depth across all instances; also solved-rate. hmax=2 -> depths {0,1}."""
    tot = defaultdict(int)
    for r in rows:
        for k, v in (r.get("depth_hist") or {}).items():
            tot[int(k)] += v
    return dict(tot)


def solve_rank_stats(rows):
    """(c): winning-plan priority ranks (0-indexed). first = winning first push's rank in the root pool."""
    firsts, seconds, plan_lens = [], [], []
    for r in rows:
        if not r.get("solved") or not r.get("solve_ranks"):
            continue
        sr = r["solve_ranks"]
        plan_lens.append(len(sr))
        firsts.append(sr[0])
        if len(sr) >= 2:
            seconds.append(sr[1])
    return firsts, seconds, plan_lens


def per_instance_sims(dirs, model_filter):
    """Map (xml,obj) -> list of n_sim over seeds that SOLVED it (unsolved -> excluded, sims were budget-capped)."""
    solved_sims = defaultdict(list)
    n_seeds_solved = defaultdict(int)
    for d in dirs:
        for r in load_seed(d, model_filter):
            k = key_of(r)
            if r.get("solved"):
                solved_sims[k].append(r.get("n_sim") if r.get("n_sim") is not None else r.get("sims"))
                n_seeds_solved[k] += 1
    return solved_sims, n_seeds_solved


def fmt_row(label, vals, w=8):
    return f"{label:14s}" + "".join(f"{v:>{w}.1f}" for v in vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dirs", nargs="+", required=True)
    ap.add_argument("--random-dirs", nargs="+", required=True)
    ap.add_argument("--model-filter", default="NoHz")
    ap.add_argument("--random-filter", default="random")
    ap.add_argument("--model-label", default="NoHz-v3")
    ap.add_argument("--random-label", default="random")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--plot-dirs", nargs="+", default=[], help="dirs to save PNGs into (e.g. eval dir + repo assets)")
    a = ap.parse_args()

    mseeds = [load_seed(d, a.model_filter) for d in a.model_dirs]
    rseeds = [load_seed(d, a.random_filter) for d in a.random_dirs]
    mseeds = [s for s in mseeds if s]
    rseeds = [s for s in rseeds if s]
    print(f"model seeds: {[len(s) for s in mseeds]}  random seeds: {[len(s) for s in rseeds]}", file=sys.stderr)

    # ---- curves + bands ----
    m_sim = [sims_curve(s, SIM_GRID) for s in mseeds]; r_sim = [sims_curve(s, SIM_GRID) for s in rseeds]
    m_sim_mu, m_sim_sd = band(m_sim); r_sim_mu, r_sim_sd = band(r_sim)

    all_solved_t = np.asarray([r["t_wall"] for s in mseeds + rseeds for r in s if r.get("solved")])
    tmax = float(np.percentile(all_solved_t, 99.0)) if all_solved_t.size else 1.0  # cap tail for a readable x-axis
    T_GRID = np.linspace(0, tmax, 400)
    m_t = [time_curve(s, T_GRID) for s in mseeds]; r_t = [time_curve(s, T_GRID) for s in rseeds]
    m_t_mu, m_t_sd = band(m_t); r_t_mu, r_t_sd = band(r_t)

    # time cutoffs: round numbers spanning the FULL data (table is independent of the plot's capped x-axis)
    tmax_full = float(all_solved_t.max()) if all_solved_t.size else 1.0
    TIME_CUTS = [c for c in [1, 2, 5, 10, 30, 60, 120, 240, 480] if c <= tmax_full * 1.05]

    # ---- tables ----
    def cut_band(seeds, kind, cuts):
        M = np.array([[at_cut(s, kind, c) for c in cuts] for s in seeds])
        return M.mean(0), M.std(0)
    m_scut_mu, m_scut_sd = cut_band(mseeds, "sim", SIM_CUTS)
    r_scut_mu, r_scut_sd = cut_band(rseeds, "sim", SIM_CUTS)
    m_tcut_mu, m_tcut_sd = cut_band(mseeds, "time", TIME_CUTS)
    r_tcut_mu, r_tcut_sd = cut_band(rseeds, "time", TIME_CUTS)

    # overall solve-rate + avg sims (per seed then mean)
    def overall(seeds):
        srate = [100.0 * sum(r.get("solved", False) for r in s) / len(s) for s in seeds]
        avg_all = [np.mean([r.get("n_sim") or r.get("sims") for r in s]) for s in seeds]
        avg_solve = [np.mean([r.get("n_sim") or r.get("sims") for r in s if r.get("solved")]) for s in seeds]
        avg_twall = [np.mean([r["t_wall"] for r in s]) for s in seeds]
        return (np.mean(srate), np.std(srate), np.mean(avg_all), np.mean(avg_solve), np.mean(avg_twall))
    m_ov = overall(mseeds); r_ov = overall(rseeds)

    # ---- (a)/(b) depth ----
    m_depth = depth_totals([r for s in mseeds for r in s])
    r_depth = depth_totals([r for s in rseeds for r in s])
    def bd_ratio(d):
        d0, d1 = d.get(0, 0), d.get(1, 0)
        return (d0 / d1) if d1 else float("inf"), d0, d1
    m_bd = bd_ratio(m_depth); r_bd = bd_ratio(r_depth)

    # ---- (c) solve ranks ----
    m_f, m_s, m_pl = solve_rank_stats([r for s in mseeds for r in s])
    r_f, r_s, r_pl = solve_rank_stats([r for s in rseeds for r in s])
    def rank_summary(firsts, seconds, plens):
        firsts = np.asarray(firsts); seconds = np.asarray(seconds); plens = np.asarray(plens)
        return {
            "n_solved_with_ranks": int(firsts.size),
            "first_rank0_pct": round(100.0 * np.count_nonzero(firsts == 0) / firsts.size, 1) if firsts.size else 0.0,
            "first_rank_le2_pct": round(100.0 * np.count_nonzero(firsts <= 2) / firsts.size, 1) if firsts.size else 0.0,
            "first_rank_median": float(np.median(firsts)) if firsts.size else None,
            "first_rank_mean": round(float(firsts.mean()), 2) if firsts.size else None,
            "twopush_win_pct": round(100.0 * np.count_nonzero(plens == 2) / plens.size, 1) if plens.size else 0.0,
            "second_rank_median": float(np.median(seconds)) if seconds.size else None,
            "second_rank_mean": round(float(seconds.mean()), 2) if seconds.size else None,
        }
    m_rank = rank_summary(m_f, m_s, m_pl); r_rank = rank_summary(r_f, r_s, r_pl)

    # ---- (d) solved-but-slow (per-instance, model vs random) ----
    m_solved_sims, m_ns = per_instance_sims(a.model_dirs, a.model_filter)
    r_solved_sims, r_ns = per_instance_sims(a.random_dirs, a.random_filter)
    nM, nR = len(mseeds), len(rseeds)
    both, model_slower, model_faster, deltas = 0, 0, 0, []
    model_only, random_only = 0, 0
    for k in set(list(m_solved_sims) + list(r_solved_sims)):
        m_all = m_ns.get(k, 0) == nM  # solved in ALL model seeds
        r_all = r_ns.get(k, 0) == nR  # solved in ALL random seeds
        if m_all and r_all:
            both += 1
            dm = np.mean(m_solved_sims[k]); dr = np.mean(r_solved_sims[k])
            deltas.append(dm - dr)
            if dm > dr:
                model_slower += 1
            elif dm < dr:
                model_faster += 1
        elif m_ns.get(k, 0) > 0 and r_ns.get(k, 0) == 0:
            model_only += 1
        elif r_ns.get(k, 0) > 0 and m_ns.get(k, 0) == 0:
            random_only += 1
    deltas = np.asarray(deltas)
    d_stats = {
        "n_both_all_seeds_solved": both,
        "model_slower_count": model_slower,
        "model_slower_pct": round(100.0 * model_slower / both, 1) if both else 0.0,
        "model_faster_count": model_faster,
        "model_faster_pct": round(100.0 * model_faster / both, 1) if both else 0.0,
        "median_sims_delta_model_minus_random": float(np.median(deltas)) if deltas.size else None,
        "mean_sims_delta_model_minus_random": round(float(deltas.mean()), 1) if deltas.size else None,
        "model_only_solved_instances": model_only,
        "random_only_solved_instances": random_only,
    }

    # ---- console tables ----
    print(f"\n=== HEADLINE ({a.model_label} n={nM} vs {a.random_label} n={nR}) ===")
    print(f"{a.model_label:12s}: solve={m_ov[0]:.1f}%+-{m_ov[1]:.1f}  avg_sims_all={m_ov[2]:.1f}  avg_to_solve={m_ov[3]:.1f}  avg_twall={m_ov[4]:.1f}s")
    print(f"{a.random_label:12s}: solve={r_ov[0]:.1f}%+-{r_ov[1]:.1f}  avg_sims_all={r_ov[2]:.1f}  avg_to_solve={r_ov[3]:.1f}  avg_twall={r_ov[4]:.1f}s")
    print(f"\n=== solve@sims (%) ===\n{'':14s}" + "".join(f"{'@'+str(c):>8s}" for c in SIM_CUTS))
    print(fmt_row(a.model_label, m_scut_mu)); print(fmt_row("  std", m_scut_sd))
    print(fmt_row(a.random_label, r_scut_mu)); print(fmt_row("  std", r_scut_sd))
    print(f"\n=== solve@time (%) ===\n{'':14s}" + "".join(f"{str(c)+'s':>8s}" for c in TIME_CUTS))
    print(fmt_row(a.model_label, m_tcut_mu)); print(fmt_row("  std", m_tcut_sd))
    print(fmt_row(a.random_label, r_tcut_mu)); print(fmt_row("  std", r_tcut_sd))
    print(f"\n(a) breadth:depth  {a.model_label}: d0={m_bd[1]} d1={m_bd[2]} ratio={m_bd[0]:.3f}   {a.random_label}: d0={r_bd[1]} d1={r_bd[2]} ratio={r_bd[0]:.3f}")
    print(f"(b) sims-by-depth  {a.model_label}: {m_depth}   {a.random_label}: {r_depth}")
    print(f"(c) solve_ranks    {a.model_label}: {m_rank}")
    print(f"                   {a.random_label}: {r_rank}")
    print(f"(d) solved-but-slow: {d_stats}")

    out = {
        "model_label": a.model_label, "random_label": a.random_label,
        "n_model_seeds": nM, "n_random_seeds": nR,
        "headline": {
            "model": {"solve_pct": round(m_ov[0], 1), "solve_std": round(m_ov[1], 1), "avg_sims_all": round(m_ov[2], 1), "avg_sims_to_solve": round(m_ov[3], 1), "avg_twall_s": round(m_ov[4], 1)},
            "random": {"solve_pct": round(r_ov[0], 1), "solve_std": round(r_ov[1], 1), "avg_sims_all": round(r_ov[2], 1), "avg_sims_to_solve": round(r_ov[3], 1), "avg_twall_s": round(r_ov[4], 1)},
        },
        "solve_at_sims": {"cuts": SIM_CUTS,
                          "model_mean": [round(x, 1) for x in m_scut_mu], "model_std": [round(x, 1) for x in m_scut_sd],
                          "random_mean": [round(x, 1) for x in r_scut_mu], "random_std": [round(x, 1) for x in r_scut_sd]},
        "solve_at_time": {"cuts": TIME_CUTS,
                          "model_mean": [round(x, 1) for x in m_tcut_mu], "model_std": [round(x, 1) for x in m_tcut_sd],
                          "random_mean": [round(x, 1) for x in r_tcut_mu], "random_std": [round(x, 1) for x in r_tcut_sd]},
        "a_breadth_depth": {"model": {"depth0_sims": m_bd[1], "depth1_sims": m_bd[2], "ratio": round(m_bd[0], 3)},
                             "random": {"depth0_sims": r_bd[1], "depth1_sims": r_bd[2], "ratio": round(r_bd[0], 3)}},
        "b_sims_by_depth": {"model": m_depth, "random": r_depth},
        "c_solve_ranks": {"model": m_rank, "random": r_rank},
        "d_solved_but_slow": d_stats,
    }
    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    json.dump(out, open(a.out_json, "w"), indent=1)
    print(f"\nwrote {a.out_json}", file=sys.stderr)

    # ---- plots ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"matplotlib unavailable ({e}); skipping plots", file=sys.stderr)
        return

    MC, RC = "#1f77b4", "#d62728"

    def curve_fig(x, m_mu, m_sd, r_mu, r_sd, xlabel, title, fname, logx=False, vlines=None):
        fig, ax = plt.subplots(figsize=(7.2, 4.6))
        ax.plot(x, m_mu, color=MC, lw=2, label=f"{a.model_label} (n={nM})")
        ax.fill_between(x, m_mu - m_sd, m_mu + m_sd, color=MC, alpha=0.22, lw=0)
        ax.plot(x, r_mu, color=RC, lw=2, label=f"{a.random_label} (n={nR})")
        ax.fill_between(x, r_mu - r_sd, r_mu + r_sd, color=RC, alpha=0.22, lw=0)
        if logx:
            ax.set_xscale("log")
        if vlines:
            for v in vlines:
                ax.axvline(v, color="0.8", lw=0.7, zorder=0)
        ax.set_xlabel(xlabel); ax.set_ylabel("% instances solved"); ax.set_title(title)
        ax.set_ylim(0, 100); ax.grid(True, alpha=0.25); ax.legend(loc="lower right", frameon=False)
        fig.tight_layout()
        for pd in a.plot_dirs:
            os.makedirs(pd, exist_ok=True); fig.savefig(os.path.join(pd, fname), dpi=130)
        plt.close(fig)

    curve_fig(np.asarray(SIM_GRID), m_sim_mu, m_sim_sd, r_sim_mu, r_sim_sd,
              "simulations budget B (log)", "Full best-first: success vs sims (budget 900, pure2push car)",
              "fullsearch_success_vs_sims.png", logx=True, vlines=SIM_CUTS)
    curve_fig(T_GRID, m_t_mu, m_t_sd, r_t_mu, r_t_sd,
              "wall-clock budget T (s)", "Full best-first: success vs wall-time (emeraldrapids-exclusive)",
              "fullsearch_success_vs_time.png", logx=False, vlines=TIME_CUTS)

    # aggregation figure: (b) sims-by-depth  +  (c) first-push winning rank hist
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    depths = [0, 1]
    mw = [m_depth.get(d, 0) for d in depths]; rw = [r_depth.get(d, 0) for d in depths]
    mw = [100.0 * v / sum(mw) for v in mw] if sum(mw) else mw
    rw = [100.0 * v / sum(rw) for v in rw] if sum(rw) else rw
    xb = np.arange(2); w = 0.36
    axs[0].bar(xb - w / 2, mw, w, color=MC, label=a.model_label)
    axs[0].bar(xb + w / 2, rw, w, color=RC, label=a.random_label)
    axs[0].set_xticks(xb); axs[0].set_xticklabels(["depth 0\n(first push)", "depth 1\n(second push / dive)"])
    axs[0].set_ylabel("% of all sims"); axs[0].set_title("(b) where the budget goes: sims by tree-depth")
    axs[0].legend(frameon=False); axs[0].grid(True, axis="y", alpha=0.25)

    RB = list(range(0, 11))
    mf = np.asarray(m_f); mf = mf[mf < 11]
    rf = np.asarray(r_f); rf = rf[rf < 11]
    mh = np.array([np.count_nonzero(mf == b) for b in RB], float); mh = 100 * mh / max(len(m_f), 1)
    rh = np.array([np.count_nonzero(rf == b) for b in RB], float); rh = 100 * rh / max(len(r_f), 1)
    axs[1].bar(np.array(RB) - w / 2, mh, w, color=MC, label=a.model_label)
    axs[1].bar(np.array(RB) + w / 2, rh, w, color=RC, label=a.random_label)
    axs[1].set_xticks(RB); axs[1].set_xlabel("priority-rank of the WINNING first push (0 = model's #1)")
    axs[1].set_ylabel("% of solved instances"); axs[1].set_title("(c) is the ranker's top pick the winner?")
    axs[1].legend(frameon=False); axs[1].grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    for pd in a.plot_dirs:
        fig.savefig(os.path.join(pd, "fullsearch_aggregation.png"), dpi=130)
    plt.close(fig)
    print(f"wrote plots to {a.plot_dirs}", file=sys.stderr)


if __name__ == "__main__":
    main()
