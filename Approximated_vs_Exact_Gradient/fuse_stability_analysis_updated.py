# fuse_stability_analysis.py
"""
FUSE Gradient Stability Analysis
=================================
Empirical evidence that the exact (non-approximated) modularity gradient
is significantly more unstable than the approximate gradient used in FUSE.

Mathematical background
-----------------------
Let B = A − ddᵀ/(2m) be the modularity matrix.  FUSE maximises Tr(SᵀBS).

  Approximate gradient (original FUSE):
      ∇_approx S = (1/2m)(AS − d·S.sum(axis=0)/(2m))
      Treats the degree-weighted projection as a scalar global mean.
      Exact only on regular graphs (all degrees equal).

  Exact gradient:
      ∇_exact S = (2/2m)(AS − d(dᵀS)/(2m))
      Proper outer-product correction; correct for all degree distributions.
      Both variants share O(nk) complexity — no asymptotic speed difference
      on regular graphs — but ∇_exact carries per-node degree weighting that
      amplifies updates for high-degree hubs, producing larger and more
      variable gradient norms on real-world (heterogeneous-degree) graphs.

Timing methodology
------------------
Total embedding time is measured exactly as in run_one_experiment() in
benchmarking_utils.py:

    tstart = time.time()
    E = fuse_approx_embedding(...)   # full ITERATIONS-step loop, uninterrupted
    t_elapsed = time.time() - tstart

Both variants are run from the SAME S₀, with the SAME supervised/semi-supervised
terms, for the SAME number of iterations, so the only timing difference is the
modularity gradient computation itself. This is a fair apples-to-apples comparison
matching how the main benchmark records embedding time.

Gradient geometry metrics (cosine similarity, norm, ratio, objective) are recorded
in a SEPARATE diagnostic pass that does not affect the timing numbers.

Outputs
-------
  ./fuse_stability_results/
    gradient_data.json       — full per-iteration diagnostic records + timing summary
    stability_summary.csv    — one row per dataset 
    convergence_curves.csv   — per-iteration gradient geometry for plotting
    snapshot_accuracy.csv    — GNN accuracy at final embeddings per classifier
    console                  — formatted table
"""

import os, json, random, time
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
import pandas as pd

import scipy.linalg
if not hasattr(scipy.linalg, 'triu'):
    scipy.linalg.triu = np.triu  # gensim compat patch

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from benchmarking_utils_stability import (
    load_dataset,
    create_label_mask,
    perform_labeled_random_walks,
    compute_attention_weights,
    train_and_evaluate_classifier,
    _approx_modularity_gradient,
    _exact_modularity_gradient,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
DEFAULT_EMB_DIM   = 150
ITERATIONS        = 200       # must match fuse_iterations in run_benchmark()
MASK_FRAC         = 0.7
SEED              = 42
CLASSIFIER_EPOCHS = 200       # must match main benchmark
CLASSIFIERS       = ['gcn', 'gat', 'graphsage']
DATASETS          = ['cora', 'citeseer', 'wikics', 'photo', 'pubmed']
OUT_DIR           = "./fuse_stability_results"
N_TIMING_RUNS     = 3         # repeat end-to-end timing runs and average (reduces noise)


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def modularity_objective(A, degrees, m, S):
    """Tr(SᵀBS) = Tr(SᵀAS) - ||dᵀS||^2/(2m)"""
    trSAS = float(np.trace(S.T @ (A @ S)))
    dTS   = degrees @ S
    return trSAS - float(np.dot(dTS, dTS)) / (2 * m)

def cosine_similarity_matrices(G1, G2):
    """Mean per-row cosine similarity between two (n,k) gradient matrices."""
    n1 = np.linalg.norm(G1, axis=1, keepdims=True).clip(1e-10)
    n2 = np.linalg.norm(G2, axis=1, keepdims=True).clip(1e-10)
    return float(np.sum((G1 / n1) * (G2 / n2), axis=1).mean())

def frobenius_norm(M):
    return float(np.linalg.norm(M, 'fro'))


# ─────────────────────────────────────────────────────────────────────────────
# Shared supervised + semi-supervised gradient terms
# ─────────────────────────────────────────────────────────────────────────────

def make_auxiliary_grad_fns(labels_int, label_mask, attention_weights, n):
    """Return closures for supervised and semi-supervised gradient terms."""

    def supervised_grad(S):
        g = np.zeros_like(S)
        if np.any(label_mask):
            for lab in np.unique(labels_int[label_mask]):
                mask = (labels_int == lab) & label_mask
                if mask.sum() == 0:
                    continue
                g[mask] = S[mask] - np.mean(S[mask], axis=0, keepdims=True)
        return g

    def semi_grad(S):
        g = np.zeros_like(S)
        for i in range(n):
            if (not label_mask[i]) and (i in attention_weights):
                g[i] = S[i] - sum(w * S[j]
                                   for j, w in attention_weights[i].items())
        return g

    return supervised_grad, semi_grad


# ─────────────────────────────────────────────────────────────────────────────
# TIMED end-to-end embedding runs — mirrors run_one_experiment() exactly
# ─────────────────────────────────────────────────────────────────────────────

def _run_timed(grad_fn, A, degrees, m, S0,
               supervised_grad, semi_grad,
               iterations, eta=0.01, lambda_sup=1.0, lambda_semi=2.0):
    """
    Run gradient ascent to completion using grad_fn, returning (S_final, elapsed_seconds).

    The timing block wraps the ENTIRE loop exactly as tstart/t_elapsed in
    run_one_experiment() wraps fuse_embedding(). No metrics computed inside
    to avoid contaminating timing.
    """
    S = S0.copy()
    t_start = time.time()                        # <- same as tstart in run_one_experiment()
    for _ in range(iterations):
        gm  = grad_fn(A, degrees, m, S)
        gs  = supervised_grad(S)
        gss = semi_grad(S)
        S  += eta * (gm - lambda_sup * gs - lambda_semi * gss)
        S, _ = np.linalg.qr(S)
    t_elapsed = time.time() - t_start            # <- same as t_elapsed in run_one_experiment()
    return S, t_elapsed


def run_timing_phase(A, degrees, m, S0,
                     supervised_grad, semi_grad,
                     iterations, n_runs=N_TIMING_RUNS):
    """
    Run both variants n_runs times each from the same S0 and report
    mean +/- std total embedding time (seconds) — matching how
    run_one_experiment() measures embedding time for every other method.

    Returns timing dict including final embeddings from the last run
    (used for GNN evaluation).
    """
    exact_times, approx_times = [], []
    S_exact_final = S_approx_final = None

    for run_idx in range(n_runs):
        S_e, t_e = _run_timed(_exact_modularity_gradient,
                               A, degrees, m, S0,
                               supervised_grad, semi_grad, iterations)
        S_a, t_a = _run_timed(_approx_modularity_gradient,
                               A, degrees, m, S0,
                               supervised_grad, semi_grad, iterations)
        exact_times.append(t_e)
        approx_times.append(t_a)
        S_exact_final  = S_e
        S_approx_final = S_a
        print(f"    [timing run {run_idx+1}/{n_runs}]"
              f"  exact={t_e:.3f}s  approx={t_a:.3f}s"
              f"  ratio={t_e/max(t_a,1e-9):.3f}x")

    em = float(np.mean(exact_times))
    am = float(np.mean(approx_times))
    return {
        "exact_times_s":   exact_times,
        "approx_times_s":  approx_times,
        "exact_mean_s":    round(em, 4),
        "exact_std_s":     round(float(np.std(exact_times)),  4),
        "approx_mean_s":   round(am, 4),
        "approx_std_s":    round(float(np.std(approx_times)), 4),
        "speedup":         round(em / max(am, 1e-9), 4),
        "S_exact_final":   S_exact_final,
        "S_approx_final":  S_approx_final,
    }


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC gradient-geometry loop — no timing pressure
# ─────────────────────────────────────────────────────────────────────────────

def run_diagnostic_phase(A, degrees, m, S0,
                          supervised_grad, semi_grad,
                          iterations,
                          eta=0.01, lambda_sup=1.0, lambda_semi=2.0):
    """
    Run both variants in lock-step recording gradient geometry at every iteration.
    This pass is NOT timed — it exists solely to produce instability evidence:
    norm variance, cosine similarity, and objective curves.
    """
    per_iter = {k: [] for k in [
        "iteration",
        "exact_grad_norm",   "approx_grad_norm",
        "grad_norm_ratio",
        "cosine_similarity",
        "exact_obj",         "approx_obj",
    ]}

    S_e, S_a = S0.copy(), S0.copy()

    for it in range(iterations):
        gm_e   = _exact_modularity_gradient(A, degrees, m, S_e)
        gs_e   = supervised_grad(S_e)
        gss_e  = semi_grad(S_e)
        grad_e = gm_e - lambda_sup * gs_e - lambda_semi * gss_e
        S_e   += eta * grad_e
        S_e, _ = np.linalg.qr(S_e)

        gm_a   = _approx_modularity_gradient(A, degrees, m, S_a)
        gs_a   = supervised_grad(S_a)
        gss_a  = semi_grad(S_a)
        grad_a = gm_a - lambda_sup * gs_a - lambda_semi * gss_a
        S_a   += eta * grad_a
        S_a, _ = np.linalg.qr(S_a)

        cos_sim = cosine_similarity_matrices(gm_e, gm_a)
        norm_e  = frobenius_norm(grad_e)
        norm_a  = frobenius_norm(grad_a)
        ratio   = norm_e / max(norm_a, 1e-10)
        obj_e   = modularity_objective(A, degrees, m, S_e)
        obj_a   = modularity_objective(A, degrees, m, S_a)

        per_iter["iteration"].append(it)
        per_iter["exact_grad_norm"].append(round(norm_e,    6))
        per_iter["approx_grad_norm"].append(round(norm_a,   6))
        per_iter["grad_norm_ratio"].append(round(ratio,     6))
        per_iter["cosine_similarity"].append(round(cos_sim, 6))
        per_iter["exact_obj"].append(round(obj_e, 6))
        per_iter["approx_obj"].append(round(obj_a, 6))

    return per_iter


# ─────────────────────────────────────────────────────────────────────────────
# GNN evaluation on final embeddings
# ─────────────────────────────────────────────────────────────────────────────

def run_gnn_phase(S_exact, S_approx, adjacency_csr, y_onehot,
                   labels_int, label_mask, classifiers, clf_epochs, seed):
    """
    Evaluate final embeddings from both variants with all classifiers.
    Uses train_and_evaluate_classifier() unchanged — identical to main benchmark.
    """
    gnn_results = {}
    for clf in classifiers:
        print(f"    [GNN] {clf} ...", end=" ", flush=True)
        res_e = train_and_evaluate_classifier(
            S_exact, adjacency_csr, y_onehot, labels_int, label_mask,
            classifier_name=clf, epochs=clf_epochs, seed=seed, verbose=False
        )
        res_a = train_and_evaluate_classifier(
            S_approx, adjacency_csr, y_onehot, labels_int, label_mask,
            classifier_name=clf, epochs=clf_epochs, seed=seed, verbose=False
        )
        gnn_results[clf] = {
            "exact_accuracy":  round(res_e["accuracy"], 4),
            "approx_accuracy": round(res_a["accuracy"], 4),
            "exact_f1":        round(res_e["f1_score"],  4),
            "approx_f1":       round(res_a["f1_score"],  4),
        }
        print(f"exact={res_e['accuracy']:.4f}  approx={res_a['accuracy']:.4f}")
    return gnn_results


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_stability_analysis(dataset_name, root=".", k=DEFAULT_EMB_DIM,
                            iterations=ITERATIONS, mask_frac=MASK_FRAC,
                            seed=SEED, n_timing_runs=N_TIMING_RUNS,
                            classifiers=CLASSIFIERS, clf_epochs=CLASSIFIER_EPOCHS):
    print(f"\n{'='*65}")
    print(f"  Dataset : {dataset_name}")
    print(f"{'='*65}")

    np.random.seed(seed)
    random.seed(seed)

    ds         = load_dataset(dataset_name, root=root)
    labels_int = ds['labels']
    G          = ds['G']
    A_full     = ds['a']
    y_onehot   = ds['y']
    n          = labels_int.shape[0]
    m          = G.number_of_edges()

    _, label_mask, _ = create_label_mask(labels_int, mask_frac, seed=seed)

    A       = csr_matrix(nx.to_scipy_sparse_array(G, format='csr'))
    degrees = np.array(A.sum(axis=1)).flatten()

    # Shared initialisation — identical to fuse_embedding() in benchmarking_utils.py:
    #   S = np.random.randn(n, k); S, _ = np.linalg.qr(S)
    S0    = np.random.randn(n, k)
    S0, _ = np.linalg.qr(S0)

    labeled_walks     = perform_labeled_random_walks(G, label_mask, labels_int)
    attention_weights = compute_attention_weights(S0, labeled_walks)
    supervised_grad, semi_grad = make_auxiliary_grad_fns(
        labels_int, label_mask, attention_weights, n
    )

    # ── Phase 1: end-to-end timing (mirrors run_one_experiment) ──────────────
    print(f"\n  Phase 1 — end-to-end timing ({n_timing_runs} runs each) ...")
    timing = run_timing_phase(
        A, degrees, m, S0,
        supervised_grad, semi_grad,
        iterations=iterations,
        n_runs=n_timing_runs,
    )
    print(f"\n  Timing result:")
    print(f"    exact  : {timing['exact_mean_s']:.3f}s +/- {timing['exact_std_s']:.3f}s")
    print(f"    approx : {timing['approx_mean_s']:.3f}s +/- {timing['approx_std_s']:.3f}s")
    print(f"    speedup (approx is faster by): {timing['speedup']:.3f}x")

    # ── Phase 2: diagnostic geometry (untimed) ────────────────────────────────
    print(f"\n  Phase 2 — gradient geometry diagnostics ({iterations} iterations) ...")
    per_iter = run_diagnostic_phase(
        A, degrees, m, S0,
        supervised_grad, semi_grad,
        iterations=iterations,
    )
    ne_arr    = np.array(per_iter["exact_grad_norm"])
    na_arr    = np.array(per_iter["approx_grad_norm"])
    cos_arr   = np.array(per_iter["cosine_similarity"])
    ratio_arr = np.array(per_iter["grad_norm_ratio"])

    print(f"    ||g_e|| : {ne_arr.mean():.4f} +/- {ne_arr.std():.4f}  "
          f"(instability ratio vs approx: {ne_arr.std()/max(na_arr.std(),1e-10):.3f}x)")
    print(f"    ||g_a|| : {na_arr.mean():.4f} +/- {na_arr.std():.4f}")
    print(f"    cos     : {cos_arr.mean():.4f} +/- {cos_arr.std():.4f}  "
          f"(min={cos_arr.min():.4f})")
    print(f"    ratio   : {ratio_arr.mean():.4f} +/- {ratio_arr.std():.4f}")

    # ── Phase 3: GNN evaluation on final embeddings ───────────────────────────
    print(f"\n  Phase 3 — GNN evaluation on final embeddings ...")
    gnn_results = run_gnn_phase(
        timing["S_exact_final"], timing["S_approx_final"],
        A_full, y_onehot, labels_int, label_mask,
        classifiers=classifiers, clf_epochs=clf_epochs, seed=seed,
    )

    # ── Assemble summary ──────────────────────────────────────────────────────
    summary = {
        "dataset":  dataset_name,
        "n_nodes":  int(n),
        "n_edges":  int(m),
        # End-to-end timing — directly comparable to embedding_time_seconds
        # column in per_run_results_all.csv from run_benchmark()
        "exact_total_time_s":        timing["exact_mean_s"],
        "exact_total_time_std_s":    timing["exact_std_s"],
        "approx_total_time_s":       timing["approx_mean_s"],
        "approx_total_time_std_s":   timing["approx_std_s"],
        "speedup_approx_over_exact": timing["speedup"],
        "n_timing_runs":             n_timing_runs,
        # Gradient geometry — instability evidence
        "mean_exact_grad_norm":      round(float(ne_arr.mean()),    4),
        "std_exact_grad_norm":       round(float(ne_arr.std()),     4),
        "mean_approx_grad_norm":     round(float(na_arr.mean()),    4),
        "std_approx_grad_norm":      round(float(na_arr.std()),     4),
        "instability_ratio":         round(float(ne_arr.std()) /
                                           max(float(na_arr.std()), 1e-10), 4),
        "mean_cosine_sim":           round(float(cos_arr.mean()),   4),
        "std_cosine_sim":            round(float(cos_arr.std()),    4),
        "min_cosine_sim":            round(float(cos_arr.min()),    4),
        "mean_norm_ratio":           round(float(ratio_arr.mean()), 4),
        "std_norm_ratio":            round(float(ratio_arr.std()),  4),
    }
    for clf, r in gnn_results.items():
        summary[f"exact_acc_{clf}"]  = r["exact_accuracy"]
        summary[f"approx_acc_{clf}"] = r["approx_accuracy"]
        summary[f"exact_f1_{clf}"]   = r["exact_f1"]
        summary[f"approx_f1_{clf}"]  = r["approx_f1"]

    print(f"\n  Quality (final embedding accuracy, exact / approx):")
    for clf in classifiers:
        print(f"    {clf:<12}  exact={summary[f'exact_acc_{clf}']:.4f}"
              f"  approx={summary[f'approx_acc_{clf}']:.4f}")

    return {
        "per_iter": per_iter,
        "timing":   timing,
        "gnn":      gnn_results,
        "summary":  summary,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CSV export helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_convergence_curves_df(all_data):
    """One row per (dataset x iteration) with gradient geometry columns."""
    rows = []
    for ds, v in all_data.items():
        pi = v["per_iter"]
        for i, it in enumerate(pi["iteration"]):
            rows.append({
                "dataset":           ds,
                "iteration":         it,
                "exact_grad_norm":   pi["exact_grad_norm"][i],
                "approx_grad_norm":  pi["approx_grad_norm"][i],
                "grad_norm_ratio":   pi["grad_norm_ratio"][i],
                "cosine_similarity": pi["cosine_similarity"][i],
                "exact_obj":         pi["exact_obj"][i],
                "approx_obj":        pi["approx_obj"][i],
            })
    return pd.DataFrame(rows)


def build_snapshot_accuracy_df(all_data, classifiers):
    """
    One row per (dataset x classifier).
    exact_total_time_s / approx_total_time_s are the end-to-end embedding
    times, directly comparable to embedding_time_seconds in the benchmark CSVs.
    """
    rows = []
    for ds, v in all_data.items():
        t = v["timing"]
        for clf in classifiers:
            r = v["gnn"].get(clf, {})
            rows.append({
                "dataset":             ds,
                "classifier":          clf,
                "exact_total_time_s":  t["exact_mean_s"],
                "approx_total_time_s": t["approx_mean_s"],
                "exact_accuracy":      r.get("exact_accuracy"),
                "approx_accuracy":     r.get("approx_accuracy"),
                "exact_f1":            r.get("exact_f1"),
                "approx_f1":           r.get("approx_f1"),
            })
    return pd.DataFrame(rows)


def print_table(all_data, classifiers):
    SEP = "-"
    clf_abbr = {"gcn": "GCN", "gat": "GAT", "graphsage": "SAGE"}

    print("\n" + "=" * 115)
    print(f"{'Dataset':<14}  "
          f"{'Exact time (s)':>20}  {'Approx time (s)':>20}  "
          f"{'Speedup':>9}  "
          f"{'std(g_e)/std(g_a)':>19}  "
          f"{'cos(g_e,g_a)':>14}")
    print(SEP * 115)
    for ds, v in all_data.items():
        s = v["summary"]
        print(f"{ds:<14}  "
              f"{s['exact_total_time_s']:.3f} +/- {s['exact_total_time_std_s']:.3f}  "
              f"{s['approx_total_time_s']:.3f} +/- {s['approx_total_time_std_s']:.3f}  "
              f"{s['speedup_approx_over_exact']:>8.3f}x  "
              f"{s['instability_ratio']:>19.4f}  "
              f"{s['mean_cosine_sim']:>14.4f} +/- {s['std_cosine_sim']:.4f}")
    print("=" * 115)
    print(f"Timing: mean +/- std over {N_TIMING_RUNS} independent end-to-end runs (same S0, same iterations).")
    print("Methodology matches run_one_experiment() in benchmarking_utils.py.")
    print("std(g_e)/std(g_a) > 1 means exact gradient has higher norm variance = more unstable.")
    print()

    clf_header = "  ".join(f"{'Acc(E/A) ' + clf_abbr.get(c,c):>16}" for c in classifiers)
    print(f"{'Dataset':<14}  {clf_header}")
    print(SEP * (14 + len(classifiers) * 18 + 2))
    for ds, v in all_data.items():
        s = v["summary"]
        clf_vals = "  ".join(
            f"{s.get(f'exact_acc_{c}', 0):.4f}/{s.get(f'approx_acc_{c}', 0):.4f}"
            for c in classifiers
        )
        print(f"{ds:<14}  {clf_vals}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    all_data = {}
    for ds in DATASETS:
        try:
            all_data[ds] = run_stability_analysis(ds)
        except Exception as exc:
            print(f"\n  [SKIP] {ds}: {exc}")
            import traceback; traceback.print_exc()

    if not all_data:
        print("No datasets completed — exiting.")
        raise SystemExit(1)

    df_summary = pd.DataFrame([v["summary"] for v in all_data.values()])
    df_summary.to_csv(os.path.join(OUT_DIR, "stability_summary.csv"), index=False)
    print(f"\n stability_summary.csv  ({len(df_summary)} rows)")

    df_curves = build_convergence_curves_df(all_data)
    df_curves.to_csv(os.path.join(OUT_DIR, "convergence_curves.csv"), index=False)
    print(f" convergence_curves.csv ({len(df_curves)} rows)")

    df_snaps = build_snapshot_accuracy_df(all_data, CLASSIFIERS)
    df_snaps.to_csv(os.path.join(OUT_DIR, "snapshot_accuracy.csv"), index=False)
    print(f" snapshot_accuracy.csv  ({len(df_snaps)} rows)")

    def make_serialisable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: make_serialisable(v) for k, v in obj.items()
                    if k not in ("S_exact_final", "S_approx_final")}
        if isinstance(obj, list):
            return [make_serialisable(v) for v in obj]
        return obj

    with open(os.path.join(OUT_DIR, "gradient_data.json"), "w") as f:
        json.dump(make_serialisable(all_data), f, indent=2)
    print(f" gradient_data.json")

    print_table(all_data, CLASSIFIERS)
