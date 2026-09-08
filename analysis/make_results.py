"""Regenerate RESULTS.md from the result directories.

ROOT is the project root that contains all the results_* directories as
siblings of this github_code/ folder. Override with the PROJECT_ROOT env
var if your layout differs, e.g.:
    PROJECT_ROOT=/path/to/project python analysis/make_results.py
"""
import json, glob, os
import numpy as np
from scipy import stats

ROOT = os.environ.get(
    "PROJECT_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
)
OUT_PATH = os.environ.get(
    "RESULTS_OUT",
    os.path.join(os.path.dirname(__file__), "RESULTS.md"),
)
EXPS = ["cl_prior", "cl_adaptive_raw", "cl_combined_staged", "cl_combined_optimized"]
SCHED = ["reorder", "subsets", "weights"]
ENCS = ["CONCHv1.5", "UNI2", "VIRCHOW2"]
OUT = []


def w(s=""):
    OUT.append(s)


def load_grid(d):
    r = {}
    for f in glob.glob(os.path.join(ROOT, d, "*_test.json")):
        s = os.path.basename(f)[: -len("_test.json")]
        ep, _, rest = s.partition("_dataset-")
        exp, _, _p = ep.partition("_pacing-")
        enc, _, tail = rest.partition("_seed-")
        seed, _, strat = tail.partition("_")
        r[(exp, enc, strat, int(seed))] = json.load(open(f))["f1"]
    return r


def load_flat(d):
    r = {}
    for f in glob.glob(os.path.join(ROOT, d, "*_test.json")):
        s = os.path.basename(f)[: -len("_test.json")]
        meth, _, rest = s.partition("_dataset-")
        enc, _, tail = rest.partition("_seed-")
        seed, _, _x = tail.partition("_")
        r[(meth, enc, int(seed))] = json.load(open(f))["f1"]
    return r


def paired(cl, base):
    """cl, base: equal-length arrays of matched (encoder, seed) observations."""
    cl, base = np.asarray(cl), np.asarray(base)
    d = cl - base
    t, p = stats.ttest_rel(cl, base)
    return cl.mean(), cl.std(ddof=1), d.mean(), d.mean() / base.mean() * 100, p


def sig(p, alpha):
    return "‡" if p < alpha else ("†" if p < 0.05 else "")


def pf(p):
    """Never print a p-value as 0.000."""
    if p < 1e-4:
        return "<0.0001"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


SEEDS = [42, 53, 78, 102, 294]


# ------------------------------------------------- 1-3 main grids -----------
GRIDS = [("AI4SKIN", "results_ai4skin_full/logs", 6),
         ("MHIST", "results_mhist/logs", 2),
         ("CrowdGleason", "results_crowdgleason/logs", 4)]



for name, d, ncls in GRIDS:
    g = load_grid(d)
    if not g:
        w(f"### {name}\n\n_(no data)_\n")
        continue
    w(f"### 1.{GRIDS.index((name,d,ncls))+1} {name} ({ncls} classes)")
    w()
    w("| Encoder | Schedule | non-CL F1 | CL F1 (mean of 4) | ΔF1 abs | ΔF1 rel | p | sig |")
    w("|---|---|---|---|---|---|---|---|")
    alpha = 0.05 / 9
    for enc in ENCS:
        for st in SCHED:
            b = [g.get(("non_cl", enc, st, s)) for s in SEEDS]
            if any(x is None for x in b):
                continue
            cl = np.mean([[g[(e, enc, st, s)] for s in SEEDS] for e in EXPS], axis=0)
            m, sd, da, dr, p = paired(cl, b)
            w(f"| {enc} | {st} | {np.mean(b):.3f} ± {np.std(b,ddof=1):.3f} | "
              f"{m:.3f} ± {sd:.3f} | {da:+.4f} | {dr:+.2f}% | {pf(p)} | {sig(p,alpha)} |")
    w()
    w(f"*Correction family: 9 comparisons (3 encoders × 3 schedules), Bonferroni α = {alpha:.4f}.*")
    w()
    # pooled per schedule
    w("**Pooled over encoders (n=15 matched pairs):**")
    w()
    w("| Schedule | non-CL F1 | CL F1 | ΔF1 abs | ΔF1 rel | p (n=15) | p (enc as unit, n=3) | sig+ |")
    w("|---|---|---|---|---|---|---|---|")
    for st in SCHED:
        cl_all, b_all, per = [], [], []
        nsig = 0
        for enc in ENCS:
            b = np.array([g[("non_cl", enc, st, s)] for s in SEEDS])
            cl = np.mean([[g[(e, enc, st, s)] for s in SEEDS] for e in EXPS], axis=0)
            cl_all.append(cl); b_all.append(b); per.append(cl.mean() - b.mean())
            for e in EXPS:
                v = np.array([g[(e, enc, st, s)] for s in SEEDS])
                if stats.ttest_rel(v, b)[1] < 0.05 and v.mean() > b.mean():
                    nsig += 1
        C, B = np.concatenate(cl_all), np.concatenate(b_all)
        m, sd, da, dr, p = paired(C, B)
        p3 = stats.ttest_1samp(per, 0)[1]
        w(f"| {st} | {B.mean():.3f} | {C.mean():.3f} | {da:+.4f} | {dr:+.2f}% | "
          f"{pf(p)} | {pf(p3)} | {nsig}/12 |")
    w()

# ------------------------------------------------- 4. controls --------------
w("---")
w()

b = load_flat("results_baselines_crowdgleason/logs")
ORDER = [("Soft labels (vote distribution)", "soft_labels", "annotator-derived"),
         ("Prior CL (weights)", "prior_curriculum_weights", "annotator-derived"),
         ("Agreement weighting (static)", "agreement_weighting", "annotator-derived"),
         ("Prior CL (subsets)", "prior_curriculum_subsets", "annotator-derived"),
         ("GCE (q=0.7)", "gce", "generic robust loss"),
         ("Label smoothing (ε=0.1)", "label_smoothing", "generic robust loss"),
         ("Focal (γ=2)", "focal", "generic robust loss"),
         ("Class-balanced (β=0.999)", "class_balanced", "generic robust loss"),
         ("Random curriculum (subsets)", "random_curriculum", "control"),
         ("Random curriculum (weights)", "random_curriculum_weights", "control"),
         ("Reverse curriculum (subsets)", "reverse_curriculum", "control"),
         ("Reverse curriculum (weights)", "reverse_curriculum_weights", "control")]
base_all = np.concatenate([[b[("non_cl", e, s)] for s in SEEDS] for e in ENCS])
alpha = 0.05 / len(ORDER)
w(f"non-CL reference: **{base_all.mean():.4f} ± {base_all.std(ddof=1):.4f}**")
w()
w("| Group | Method | F1 | ΔF1 abs | ΔF1 rel | p (n=15) | p (n=3) | sig |")
w("|---|---|---|---|---|---|---|---|")
for label, key, grp in ORDER:
    try:
        cl = np.concatenate([[b[(key, e, s)] for s in SEEDS] for e in ENCS])
    except KeyError:
        continue
    per = [np.mean([b[(key, e, s)] for s in SEEDS]) - np.mean([b[("non_cl", e, s)] for s in SEEDS]) for e in ENCS]
    m, sd, da, dr, p = paired(cl, base_all)
    p3 = stats.ttest_1samp(per, 0)[1]
    w(f"| {grp} | {label} | {m:.3f} ± {sd:.3f} | {da:+.4f} | {dr:+.2f}% | "
      f"{pf(p)} | {pf(p3)} | {sig(p,alpha)} |")
w()
w(f"*Correction family: {len(ORDER)} methods, Bonferroni α = {alpha:.4f}.*")
w()

# ------------------------------------------------- 5. Wei ------------------
w("---")
w()

for budget, desc in [("total", "13/13/12/12 = 50 epochs; baseline 50"),
                     ("perstage", "50 per stage = 200 epochs; baseline 200")]:
    d = f"results_wei_mhist_{budget}/logs"
    files = glob.glob(os.path.join(ROOT, d, "*_test.json"))
    if not files:
        w(f"### `{budget}` — _(no data)_\n"); continue
    w(f"### 3.{1 if budget=='total' else 2} Budget = `{budget}` ({desc})")
    w()
    w("| Encoder | non-CL F1 | Wei CL F1 | ΔF1 abs | ΔF1 rel | p | sig |")
    w("|---|---|---|---|---|---|---|")
    alpha = 0.05 / 3
    allc, allb, per = [], [], []
    for enc in ENCS:
        def g(m):
            return np.array([json.load(open(os.path.join(ROOT, d, f"{m}_dataset-{enc}_seed-{s}_{budget}_test.json")))["f1"] for s in SEEDS])
        try:
            cl, bb = g("wei_curriculum"), g("non_cl")
        except FileNotFoundError:
            continue
        allc.append(cl); allb.append(bb); per.append(cl.mean() - bb.mean())
        m, sd, da, dr, p = paired(cl, bb)
        w(f"| {enc} | {bb.mean():.3f} ± {bb.std(ddof=1):.3f} | {m:.3f} ± {sd:.3f} | "
          f"{da:+.4f} | {dr:+.2f}% | {pf(p)} | {sig(p,alpha)} |")
    if allc:
        C, B = np.concatenate(allc), np.concatenate(allb)
        m, sd, da, dr, p = paired(C, B)
        p3 = stats.ttest_1samp(per, 0)[1]
        w(f"| **pooled** | {B.mean():.3f} | {C.mean():.3f} | {da:+.4f} | {dr:+.2f}% | "
          f"{pf(p)} | {sig(p,0.05/3)} |")
        w()
        w(f"*Correction family: 3 encoders, Bonferroni α = {0.05/3:.4f}. "
          f"Encoder as unit of replication (n=3): p = {p3:.3f}.*")
    w()

# ------------------------------------------------- 6. cold start -----------
w("---")

r = {}
for f in glob.glob(os.path.join(ROOT, "results_ablation_warmup_crowdgleason/logs", "*_test.json")):
    s = os.path.basename(f)[: -len("_test.json")]
    tag, _, rest = s.partition("_dataset-")
    enc, _, tail = rest.partition("_seed-")
    seed, _, _x = tail.partition("_")
    r[(int(tag.split("-")[1]), enc, int(seed))] = json.load(open(f))["f1"]
ks = sorted({k for k, _, _ in r})
if ks:
    base = np.concatenate([[r[(0, e, s)] for s in SEEDS] for e in ENCS])
    alpha = 0.05 / (len(ks) - 1)
    w("| k (warm-up epochs) | F1 | ΔF1 vs k=0 | p | sig |")
    w("|---|---|---|---|---|")
    for k in ks:
        v = np.concatenate([[r[(k, e, s)] for s in SEEDS] for e in ENCS])
        if k == 0:
            w(f"| **0 (current)** | {v.mean():.4f} ± {v.std(ddof=1):.4f} | — | — | |")
        else:
            m, sd, da, dr, p = paired(v, base)
            w(f"| {k} | {m:.4f} ± {sd:.4f} | {da:+.4f} | {pf(p)} | {sig(p,alpha)} |")
    w()
    w(f"*Correction family: {len(ks)-1} comparisons vs k=0, Bonferroni α = {alpha:.4f}. "
      f"Seed-level SD at k=0 is {base.std(ddof=1):.4f} — larger than any difference observed.*")
    w()

# ------------------------------------------------- 7. staged ---------------

r = {}
for f in glob.glob(os.path.join(ROOT, "results_ablation_staged_crowdgleason/logs", "*_test.json")):
    s = os.path.basename(f)[: -len("_test.json")]
    tag, _, rest = s.partition("_dataset-")
    enc, _, tail = rest.partition("_seed-")
    seed, _, _x = tail.partition("_")
    p_ = tag.split("-")
    r[(float(p_[1][1:]), float(p_[2][1:]), enc, int(seed))] = json.load(open(f))["f1"]
if r:
    def cell(wv, tv):
        return np.concatenate([[r[(wv, tv, e, s)] for s in SEEDS] for e in ENCS])
    ref = cell(0.30, 0.70)
    cfgs = sorted({(a, b_) for a, b_, _, _ in r})
    alpha = 0.05 / (len(cfgs) - 1)
    w("| p_warm | p_tran | F1 | ΔF1 vs default | p | sig |")
    w("|---|---|---|---|---|---|")
    for a, b_ in cfgs:
        v = cell(a, b_)
        if (a, b_) == (0.30, 0.70):
            w(f"| **0.30** | **0.70** | {v.mean():.4f} ± {v.std(ddof=1):.4f} | — (default) | — | |")
        else:
            m, sd, da, dr, p = paired(v, ref)
            w(f"| {a:.2f} | {b_:.2f} | {m:.4f} ± {sd:.4f} | {da:+.4f} | {pf(p)} | {sig(p,alpha)} |")
    allm = [cell(a, b_).mean() for a, b_ in cfgs]
    w()
    w(f"*Correction family: {len(cfgs)-1} comparisons vs default, Bonferroni α = {alpha:.4f}.*")
    w()
    w(f"**Spread across all {len(cfgs)} configurations: {max(allm)-min(allm):.4f} F1** — smaller than the")
    w(f"seed-level SD at the default setting ({ref.std(ddof=1):.4f}).")


# ------------------------------------------------- 8. size match -----------

full = load_grid("results_crowdgleason/logs")
small = load_grid("results_cg_sizematch/logs")
if small:
    w("| Train size | non-CL F1 | CL F1 (mean of 4) | ΔF1 abs | ΔF1 rel | p (n=15) | positive cells |")
    w("|---|---|---|---|---|---|---|")
    for label, g, n in [("8,951 (full)", full, 8951), ("1,849 (matched)", small, 1849)]:
        cl_all, b_all, npos = [], [], 0
        for enc in ENCS:
            try:
                b = np.array([g[("non_cl", enc, "subsets", s)] for s in SEEDS])
                cl = np.mean([[g[(e, enc, "subsets", s)] for s in SEEDS] for e in EXPS], axis=0)
            except KeyError:
                continue
            cl_all.append(cl); b_all.append(b)
            for e in EXPS:
                v = np.array([g[(e, enc, "subsets", s)] for s in SEEDS])
                if v.mean() > b.mean():
                    npos += 1
        if not cl_all:
            continue
        C, B = np.concatenate(cl_all), np.concatenate(b_all)
        m, sd, da, dr, p = paired(C, B)
        w(f"| {label} | {B.mean():.3f} | {m:.3f} | {da:+.4f} | {dr:+.2f}% | {pf(p)} | {npos}/12 |")
    w()


# ------------------------------------------------- 9. noise (CrowdGleason) --

NOISE_DIR = "results_rerun_noise_crowdgleason/experiment3_{enc}_reorder_ordered"
noise_exps = ["cl_prior", "cl_adaptive_raw", "cl_combined_staged", "cl_combined_optimized"]


def noise_f1(enc, exp, pct, i):
    p = os.path.join(ROOT, NOISE_DIR.format(enc=enc), f"tmp_{exp}_noise{pct}_run{i}_test.json")
    return json.load(open(p))["f1"] if os.path.exists(p) else None


noise_levels = [5, 30]
if all(noise_f1(ENCS[0], "non_cl", pct, 0) is not None for pct in noise_levels):
    alpha = 0.05 / (len(noise_levels) * len(noise_exps))
    w("| Noise | non-CL F1 | Criterion | CL F1 | ΔF1 abs | ΔF1 rel | p (n=15) | sig |")
    w("|---|---|---|---|---|---|---|---|")
    for pct in noise_levels:
        base = np.array([noise_f1(enc, "non_cl", pct, i) for enc in ENCS for i in range(5)])
        cls = {}
        for exp in noise_exps:
            cls[exp] = np.array([noise_f1(enc, exp, pct, i) for enc in ENCS for i in range(5)])
            m, sd, da, dr, p = paired(cls[exp], base)
            w(f"| {pct}% | {base.mean():.3f} ± {base.std(ddof=1):.3f} | {exp} | "
              f"{m:.3f} ± {sd:.3f} | {da:+.4f} | {dr:+.2f}% | {pf(p)} | {sig(p,alpha)} |")
        mean_cl = np.mean([cls[e] for e in noise_exps], axis=0)
        m, sd, da, dr, p = paired(mean_cl, base)
        w(f"| {pct}% | {base.mean():.3f} ± {base.std(ddof=1):.3f} | **mean of 4** | "
          f"{m:.3f} ± {sd:.3f} | {da:+.4f} | {dr:+.2f}% | {pf(p)} | {sig(p,alpha)} |")
    w()
    w(f"*Correction family: {len(noise_levels)*len(noise_exps)} comparisons "
      f"({len(noise_levels)} levels × {len(noise_exps)} criteria), Bonferroni α = {alpha:.4f}.*")



SCARCITY_DIR = "results_rerun_scarcity_crowdgleason/experiment4_{enc}_subsets_ordered"
scarcity_exps = ["cl_prior", "cl_adaptive_raw", "cl_combined_staged", "cl_combined_optimized"]


def scarcity_f1(enc, exp, pct, i):
    p = os.path.join(ROOT, SCARCITY_DIR.format(enc=enc), f"tmp_{exp}_frac{pct}_run{i}_test.json")
    return json.load(open(p))["f1"] if os.path.exists(p) else None


fracs = [10, 25, 50, 100]
if all(scarcity_f1(ENCS[0], "non_cl", pct, 0) is not None for pct in fracs):
    alpha = 0.05 / (len(fracs) * len(scarcity_exps))
    w("| Fraction | non-CL F1 | Criterion | CL F1 | ΔF1 abs | ΔF1 rel | p (n=15) | sig |")
    w("|---|---|---|---|---|---|---|---|")
    for pct in fracs:
        base = np.array([scarcity_f1(enc, "non_cl", pct, i) for enc in ENCS for i in range(5)])
        cls = {}
        for exp in scarcity_exps:
            cls[exp] = np.array([scarcity_f1(enc, exp, pct, i) for enc in ENCS for i in range(5)])
            m, sd, da, dr, p = paired(cls[exp], base)
            w(f"| {pct}% | {base.mean():.3f} ± {base.std(ddof=1):.3f} | {exp} | "
              f"{m:.3f} ± {sd:.3f} | {da:+.4f} | {dr:+.2f}% | {pf(p)} | {sig(p,alpha)} |")
        mean_cl = np.mean([cls[e] for e in scarcity_exps], axis=0)
        m, sd, da, dr, p = paired(mean_cl, base)
        w(f"| {pct}% | {base.mean():.3f} ± {base.std(ddof=1):.3f} | **mean of 4** | "
          f"{m:.3f} ± {sd:.3f} | {da:+.4f} | {dr:+.2f}% | {pf(p)} | {sig(p,alpha)} |")
    w()
    w(f"*Correction family: {len(fracs)*len(scarcity_exps)} comparisons "
      f"({len(fracs)} fractions × {len(scarcity_exps)} criteria), Bonferroni α = {alpha:.4f}.*")



open(OUT_PATH, "w").write("\n".join(OUT) + "\n")
print(f"wrote {OUT_PATH} ({len(OUT)} lines)")
