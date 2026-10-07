"""Results layer of the v3 manuscript (RadKev v3 on the radiology benchmark, Addendum 2). Turns the job outputs into \\V{r3_*} macros,
figures, supplementary tables, a compiled preview and an editable flattened copy. Every number is ledgered (v3/numbers_v3.csv).

    python3 paper/rewrite/v3/build_v3.py            # real data; inputs not yet present are rendered as red [pending: ...] markers
    python3 paper/rewrite/v3/build_v3.py --mock     # PREDICTED inputs from predict_v3.py (banner on every page of Results)
    python3 paper/rewrite/v3/build_v3.py --strict   # fail unless every input is present and every claim holds (final build)

Outputs: v3/numbers_v3.{tex,csv}; figures/v3_*.{pdf,png} and generated/v3_*.tex (beside the manuscript's own, new names only);
v3/CLAIMS.md (every directional statement in the v3 text and whether the data support it); v3/main_v3.tex + main_v3.pdf
(main.tex with the v3 abstract, Results, Discussion, Conclusions and supplementary tables swapped in; main.tex is not modified);
v3/export/ + v3/RadKev_v3_latex.zip (self-contained .tex with every number written in).

The prose lives in v3/{abstract,results,discussion,conclusions,supp_tables}.tex and uses only \\V macros, so it can be pasted
into main.tex unchanged (figure paths figures/v3_*, tables generated/v3_*)."""
import csv
import json
import math
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent          # paper/rewrite/v3
RW = HERE.parent                                # paper/rewrite
PAPER = RW.parent
ROOT = PAPER.parent
sys.path.insert(0, str(RW))
import figures as F                             # noqa: E402  (the manuscript's figure style)

MOCK, STRICT = "--mock" in sys.argv, "--strict" in sys.argv

# ------------------------------------------------------------------------------------------------ inputs (the data contract)
# Each entry: the file the job writes, where it is expected locally, and the job that produces it. PLAN.md lists the same.
PUBLIC_V3 = (RW / "inputs").is_dir()   # public bundle (github.com/udiram/RadKev, paper/): job outputs under paper/inputs/artifacts
A_ = RW / "inputs" / "artifacts" if PUBLIC_V3 else PAPER / "artifacts"
INPUTS = {   # key: (candidate paths, first existing wins; producing job). Contract from the session that runs the jobs (2026-10-05).
    # RadCases panel prior correction is the default for RadKev (user 2026-10-06): jobs/eval_v3_prior.py, job ffe4c999, reruns eval_v3 ANALYSE
    # (+ openai_dec) and the reasoning-sample analysis with v3_27/v3_9 panel distributions divided by the training-split answer frequencies.
    "eval": ([A_ / "eval_v3_prior/artifacts/eval_v3_prior.json"], "jobs/eval_v3_prior.py (job ffe4c999; final rows of job 354b0d5e, RadCases panel prior-corrected for RadKev)"),
    "preread": ([A_ / "preread_v3/artifacts/preread_route.json", A_ / "preread_route/artifacts/preread_route.json"], "jobs/preread_route.py (v3f runs)"),
    "transfer": ([A_ / "transfer_v3/artifacts/transfer_paired_v3.json", A_ / "transfer_paired_v3/artifacts/transfer_paired_v3.json", A_ / "transfer_paired/artifacts/transfer_paired_v3.json"], "jobs/transfer_paired.py v3"),
    "latency": ([A_ / "latency_v3/artifacts/latency_v3.json"], "jobs/latency_bench.py --sample radbench"),
    "reasoning": ([A_ / "reasoning_v3_fix/artifacts/eval_v3_prior_reasoning.json"], "jobs/reasoning_v3_fix.py (reasoning-sample analysis of jobs/llm_reasoning.py v3, source-stratified bootstrap, examination tasks pooled, RadCases panel prior-corrected)"),
    "blind": ([A_ / "blind_v3_rerun/artifacts/blind_v3.json", A_ / "blind_v3/artifacts/blind_v3.json"], "jobs/blind_v3.py"),
    "answer_space": ([A_ / "answer_space3/artifacts/answer_space3.json"], "jobs/answer_space3.py (raw bits; needs a local analysis)"),
    "radgraph": ([A_ / "external_v3_analysis/artifacts/analysis_node.json", A_ / "external_v3/artifacts/analysis_node.json"], "jobs/external_tests.py v3 + jobs/external_analyse_node.py"),
    "rgxerr": ([A_ / "radgraph_xl_errors_v3/rgx_errors_v3.json"], "jobs/radgraph_xl_errors_v3.py (per-status RadGraph-XL breakdown, job 6164e2a4)"),
    "odec_eval": ([A_ / "eval_v3_prior/artifacts/eval_v3_prior.json"], "OpenAI Decisions API (gpt-6-luna) within jobs/eval_v3_prior.py"),
    "odec_lat": ([A_ / "openai_dec/artifacts/openai_dec_latency.json"], "OpenAI Decisions API, end-to-end latency from the node"),
    "odec_status": ([A_ / "openai_dec/artifacts/openai_dec_status.json"], "OpenAI Decisions API, refusals and cost"),
    "rcdiag": ([A_ / "radcases_panel_diag/radcases_panel_diag.json"], "jobs/radcases_panel_diag.py (RadCases panel errors by key type)"),
    "rcprior": ([A_ / "radcases_prior/artifacts/radcases_prior.json"], "jobs/radcases_prior.py (RadCases panel, as scored vs prior-corrected; job e6474944)"),
    "eval_raw": ([A_ / "eval_v3_rerun/artifacts/eval_v3.json"], "jobs/eval_v3.py (job 354b0d5e), RadKev as scored, without the RadCases panel prior correction"),
    "train27": ([A_ / "train_v3/artifacts/training_metrics_27b.json"], "Kev training_metrics of RadKev-27B v3 (kev-27b-dp)"),
    "train9": ([A_ / "train_v3/artifacts/training_metrics_9b.json"], "Kev training_metrics of RadKev-9B v3 (kev-9b-dp or kev-9b)"),
    "cal27": ([A_ / "train_v3/artifacts/dev_cal_27b.json"], "dev_cal summary of RadKev-27B v3 (fitted temperature)"),
    "cal9": ([A_ / "train_v3/artifacts/dev_cal_9b.json"], "dev_cal summary of RadKev-9B v3 (fitted temperature)"),
}


def locate(key):
    paths = INPUTS[key][0]
    if MOCK:
        p = HERE / "mock" / paths[0].name
        return p if p.exists() else None
    hits = [p for p in paths if p.exists()]
    return max(hits, key=lambda p: p.stat().st_mtime) if hits else None   # newest copy (a job may be recomputed in place)


SRC = {k: locate(k) for k in INPUTS}
J = {k: json.loads(p.read_text()) for k, p in SRC.items() if p}
if "odec_eval" in J and "eval" in J:   # merge only the OpenAI system, its pairs and calibration pairs; everything else stays from eval_v3
    _O, _E = J["odec_eval"], J["eval"]
    assert all(json.dumps(_O["systems"][s_], sort_keys=True) == json.dumps(_E["systems"][s_], sort_keys=True) for s_ in _E["systems"]), "openai_dec_eval differs from eval_v3"
    _E["systems"]["openai_dec"] = _O["systems"]["openai_dec"]; _E["calibration"]["openai_dec"] = _O["calibration"]["openai_dec"]
    for _t in [t for t in _E["systems"]["openai_dec"]["tasks"] if t.startswith("ctrate:")]:   # CT-RATE was not sent to the API (status plan.not_sent)
        del _E["systems"]["openai_dec"]["tasks"][_t]
    for _p in [p for p in _O["pairs"] if "openai_dec" in p]:
        for _t in [t for t in _O["pairs"][_p]["tasks"] if t.startswith("ctrate:")]: del _O["pairs"][_p]["tasks"][_t]
    _E["pairs"].update({k: v for k, v in _O["pairs"].items() if "openai_dec" in k})
    _E.setdefault("calibration_pairs", {}).update({k: v for k, v in _O.get("calibration_pairs", {}).items() if "openai_dec" in k})
if "reasoning" in J and "paired" not in J["reasoning"]: J["reasoning"] = {"paired": J["reasoning"]}   # eval_v3_prior writes the paired block itself
if "preread" in J:   # jobs/preread_route.py writes its status file with the analysis under "result" (absent until the analysis ran)
    J["preread"] = J["preread"].get("result") if "models" not in J["preread"] else J["preread"]
    if not J["preread"]: del J["preread"]
REL = lambda k: ("paper/artifacts/" + str(SRC[k].relative_to(A_))) if SRC.get(k) else "PENDING"   # same ledger source in both repositories

# ------------------------------------------------------------------------------------------------ ledger
LEDGER, PENDING = [], {}


def put(key, value, src, field, note):
    assert re.fullmatch(r"r3_[a-z0-9_]+", key), key
    assert key not in {r["key"] for r in LEDGER}, f"duplicate {key}"
    LEDGER.append({"key": key, "value": value, "source": src, "field": field, "note": note})


def pending(key, what):
    PENDING.setdefault(what, []).append(key)
    put(key, r"\missing{pending: " + what.replace("_", r"\_") + "}", "PENDING", "", what)


num = lambda x, d=1: f"{x:.{d}f}".replace("-", "\u2212")
pct = lambda x: num(100 * x)
pci = lambda c: f"{pct(c[0])} to {pct(c[1])}"
n_ = lambda x: f"{int(x):,}"
WORDS = dict(enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen".split()))
word = lambda k: WORDS.get(k, str(k))
sig = lambda c: 1 if c[0] > 0 else -1 if c[1] < 0 else 0


def cmp_words(c, more="more accurate than", less="less accurate than", same="not detectably different in accuracy from"):
    return {1: more, -1: less, 0: same}[sig(c)]


# ------------------------------------------------------------------------------------------------ names
# Comparison set (user decision 2026-10-06): RadKev v3 replaces the pilot (v2), so no v3-vs-v2 comparison is reported anywhere;
# eval_v3.json still holds v2_27/r9 and the smaller decision models (kev4, kev08, laya, laya_typed, gliner_decide, julia1),
# which are ignored. Add a model here only if the user asks for the smaller baselines.
NAME = {"v3_27": "RadKev-27B", "v3_9": "RadKev-9B", "stock27": "Kev-27B", "stock9": "Kev-9B", "qwen38": "Qwen3.8-27B", "medgemma_fix": "MedGemma-27B-text",
        "openai_dec": "OpenAI Decisions"}   # hosted decision model gpt-6-luna (amendment 10, user request 2026-10-06)
SHORT = {"v3_27": "RadKev-27B", "v3_9": "RadKev-9B", "stock27": "Kev-27B", "stock9": "Kev-9B", "qwen38": "Qwen", "medgemma_fix": "MedGemma", "openai_dec": "OpenAI"}
SYSTEMS = list(NAME)
MAIN = lambda E: {s: E["systems"][s] for s in SYSTEMS if s in E["systems"]}   # the ablation arms are reported in 3.3 only
LLMS = ["qwen38", "medgemma_fix"]
HUMAN = ["iu_finding", "iu_normal", "iu_which", "eurorad_dx", "eurorad_route", "rsna_radioqa", "radcases_panel", "radcases_topic",
         "medmcqa_rad", "medmcqa_other_rad", "medqa_rad", "medxpertqa_rad", "mmlu_rad", "pubmedqa_rad"]
CONSTR = ["rexerr_error"]
BENCH = HUMAN + CONSTR
CTRATE = ["ctrate:ctrate_finding", "ctrate:ctrate_normal", "ctrate:ctrate_which"]
NEW_SRC = ["rexerr_error", "radcases_panel", "radcases_topic"]          # training splits added for v3 (in-distribution)
EVAL_ONLY = ["iu_finding", "iu_normal", "iu_which", "rsna_radioqa", "mmlu_rad", "pubmedqa_rad", "medxpertqa_rad"]   # sources with no training split
KNOW = ["medmcqa_rad", "medmcqa_other_rad", "medqa_rad", "medxpertqa_rad", "mmlu_rad", "pubmedqa_rad"]
GROUPS = [("Report reading", ["iu_finding", "iu_normal", "iu_which", "rexerr_error"]),
          ("Case diagnosis and classification", ["eurorad_dx", "eurorad_route", "rsna_radioqa", "radcases_panel", "radcases_topic"]),
          ("Radiological knowledge", KNOW)]
TASK = {"iu_finding": "IU: finding present", "iu_normal": "IU: normal study", "iu_which": "IU: which finding",
        "rexerr_error": "ReXErr: report error", "eurorad_dx": "Eurorad: diagnosis", "eurorad_route": "Eurorad: subspecialty classification",
        "rsna_radioqa": "RSNA-RadioQA: diagnosis", "radcases_panel": "RadCases: ACR panel", "radcases_topic": "RadCases: ACR topic",
        "medmcqa_rad": "MedMCQA: radiology", "medmcqa_other_rad": "MedMCQA: other subjects", "medqa_rad": "MedQA",
        "medxpertqa_rad": "MedXpertQA", "mmlu_rad": "MMLU", "pubmedqa_rad": "PubMedQA",
        "ctrate:ctrate_finding": "CT-RATE: finding present", "ctrate:ctrate_normal": "CT-RATE: normal study", "ctrate:ctrate_which": "CT-RATE: which finding"}
TICK = {"iu_finding": "IU\nfinding", "iu_normal": "IU\nnormal", "iu_which": "IU\nwhich", "rexerr_error": "ReXErr\nerror",
        "eurorad_dx": "Eurorad\ndiagnosis", "eurorad_route": "Eurorad\nsubspec.", "rsna_radioqa": "RSNA\nRadioQA",
        "radcases_panel": "RadCases\npanel", "radcases_topic": "RadCases\ntopic", "medmcqa_rad": "MedMCQA\nradiology",
        "medmcqa_other_rad": "MedMCQA\nother", "medqa_rad": "MedQA", "medxpertqa_rad": "MedXpertQA", "mmlu_rad": "MMLU",
        "pubmedqa_rad": "PubMedQA"}
TID = {"iu_finding": "iufind", "iu_normal": "iunorm", "iu_which": "iuwhich", "rexerr_error": "rexerr", "eurorad_dx": "eurodx",
       "eurorad_route": "euroroute", "rsna_radioqa": "rsna", "radcases_panel": "rcpanel", "radcases_topic": "rctopic",
       "medmcqa_rad": "mcqrad", "medmcqa_other_rad": "mcqother", "medqa_rad": "medqa", "medxpertqa_rad": "medx", "mmlu_rad": "mmlu",
       "pubmedqa_rad": "pubmed", "ctrate:ctrate_finding": "ctfind", "ctrate:ctrate_normal": "ctnorm", "ctrate:ctrate_which": "ctwhich"}
KEYTASK = TID.__getitem__   # macro-safe task id
PAIRS = {"prim": ("v3_27", "stock27"), "spec9": ("v3_9", "stock9"),
         "q": ("v3_27", "qwen38"), "mg": ("v3_27", "medgemma_fix"), "q9": ("v3_9", "qwen38"), "size": ("v3_27", "v3_9"),
         "scale": ("stock27", "stock9"), "r9k27": ("v3_9", "stock27"), "qk": ("qwen38", "stock27"),
         "od": ("v3_27", "openai_dec"), "od9": ("v3_9", "openai_dec"), "odk": ("openai_dec", "stock27"), "odq": ("openai_dec", "qwen38"), "odmg": ("openai_dec", "medgemma_fix")}


# ------------------------------------------------------------------------------------------------ main evaluation
def build_eval():
    E = J.get("eval"); src = REL("eval")
    put("r3_banner", r"\missing{PREDICTED VALUES, NOT RESULTS: every v3 number in this document comes from predict\_v3.py and only shows the layout.}"
        if MOCK else "", src, "", "banner shown only for predicted inputs")
    put("r3_ntask", word(len(BENCH)), "Addendum 2, amendment 6", "BENCH", "benchmark tasks")
    put("r3_nhtask", word(len(HUMAN)), "Addendum 2, amendment 6", "HUMAN", "human-assigned tasks")
    put("r3_nnew", word(len(NEW_SRC)), "build_v3.NEW_SRC", "", "benchmark tasks whose training split was added in v3")
    put("r3_nold", word(len(BENCH) - len(NEW_SRC)), "build_v3", "", "benchmark tasks present in the pilot")
    if not E:
        for k in ("r3_prim_tm", "r3_prim_tm_ci", "r3_prim_htm", "r3_prim_htm_ci"): pending(k, "eval_v3")
        return False
    S, P, CAL = MAIN(E), E["pairs"], E["calibration"]
    missing_sys = [s for s in SYSTEMS if s not in S]
    if missing_sys and STRICT: raise SystemExit(f"eval_v3 lacks systems {missing_sys}")
    # benchmark size cross-check against the Methods ledger (Table 1 / build.py)
    nq = sum(E["n"][t] for t in BENCH if t in E["n"]); nh = sum(E["n"][t] for t in HUMAN if t in E["n"])
    M = {r["key"]: r["value"] for r in csv.DictReader(open(RW / "numbers.csv"))}
    if not MOCK:
        assert n_(nq) == M["v3_bench_q"] and n_(nh) == M["v3_bench_human_q"], ("benchmark counts differ from Methods", nq, nh)
    put("r3_nq", n_(nq), src, "n[BENCH]", "benchmark questions scored"); put("r3_nhq", n_(nh), src, "n[HUMAN]", "human-assigned questions scored")
    for t in BENCH + CTRATE:
        if t in E["n"]: put(f"r3_n_{KEYTASK(t)}", n_(E["n"][t]), src, f"n.{t}", f"test questions, {TASK[t]}")
    for s in SYSTEMS:
        if s not in S: continue
        for agg, a in (("bench", ""), ("human", "h")):
            r = S[s][agg]
            put(f"r3_a_{s}_{a}tm", pct(r["task_mean"]), src, f"systems.{s}.{agg}.task_mean", f"{NAME[s]}, task mean ({agg}, {r['tasks']} tasks)")
            put(f"r3_a_{s}_{a}tm_ci", pci(r["task_mean_ci"]), src, f"systems.{s}.{agg}.task_mean_ci", "95% CI")
            put(f"r3_a_{s}_{a}pool", pct(r["pooled"]), src, f"systems.{s}.{agg}.pooled", f"{NAME[s]}, pooled ({agg}, n={r['n']})")
            put(f"r3_a_{s}_{a}pool_ci", pci(r["pooled_ci"]), src, f"systems.{s}.{agg}.pooled_ci", "95% CI")
        for t, v in S[s]["tasks"].items():
            if t in TASK: put(f"r3_t_{s}_{KEYTASK(t)}", pct(v["acc"]), src, f"systems.{s}.tasks.{t}.acc", f"{NAME[s]}, {TASK[t]}")
    # the human-assigned tasks that every system answered (the LLMs skip RadCases topic: 225 options)
    common = [t for t in HUMAN if all(t in S[s]["tasks"] for s in S)]
    put("r3_ncommon", word(len(common)), src, "tasks in every system", "human-assigned tasks scored for every system")
    hc = {s: sum(S[s]["tasks"][t]["acc"] for t in common) / len(common) for s in S}
    for s in S: put(f"r3_c_{s}", pct(hc[s]), src, f"mean of systems.{s}.tasks[common].acc", f"{NAME[s]}, task mean over the common human-assigned tasks")
    rank = sorted(hc, key=lambda s: -hc[s])
    put("r3_c_best", NAME[rank[0]], src, "argmax", "most accurate system, common human-assigned tasks")
    # paired comparisons
    for al, (a, b) in PAIRS.items():
        key = f"{a}-{b}"
        if key not in P: continue
        o = P[key]
        for agg, x in (("bench", ""), ("human", "h")):
            r = o[agg]
            put(f"r3_{al}_{x}tm", pct(r["task_mean_d"]), src, f"pairs.{key}.{agg}.task_mean_d", f"{NAME[a]} minus {NAME[b]}, task mean (pp)")
            put(f"r3_{al}_{x}tm_ci", pci(r["task_mean_ci"]), src, f"pairs.{key}.{agg}.task_mean_ci", "95% CI")
            put(f"r3_{al}_{x}tm_cmp", cmp_words(r["task_mean_ci"]), src, f"sign of pairs.{key}.{agg}.task_mean_ci", "comparison words")
            put(f"r3_{al}_{x}ntask", word(r["tasks"]), src, f"pairs.{key}.{agg}.tasks", "tasks compared")
            put(f"r3_{al}_{x}pool", pct(r["pooled_d"]), src, f"pairs.{key}.{agg}.pooled_d", f"{NAME[a]} minus {NAME[b]}, pooled (pp)")
            put(f"r3_{al}_{x}pool_ci", pci(r["pooled_ci"]), src, f"pairs.{key}.{agg}.pooled_ci", "95% CI")
            put(f"r3_{al}_{x}pool_cmp", cmp_words(r["pooled_ci"]), src, f"sign of pairs.{key}.{agg}.pooled_ci", "comparison words")
            put(f"r3_{al}_{x}nq", n_(r["n"]), src, f"pairs.{key}.{agg}.n", "questions compared")
        for t, v in o["tasks"].items():
            if t not in TASK: continue
            put(f"r3_{al}_t_{KEYTASK(t)}", pct(v["d"]), src, f"pairs.{key}.tasks.{t}.d", f"{NAME[a]} minus {NAME[b]}, {TASK[t]} (pp)")
            put(f"r3_{al}_t_{KEYTASK(t)}_ci", pci(v["ci"]), src, f"pairs.{key}.tasks.{t}.ci", "95% CI")
    for al in ("prim", "q", "mg", "spec9", "od"):   # range of the per-task differences (subgroups before summaries)
        key = "-".join(PAIRS[al])
        if key not in P: continue
        tt = {t: v["d"] for t, v in P[key]["tasks"].items() if t in BENCH}
        lo_, hi_ = min(tt, key=tt.get), max(tt, key=tt.get)
        put(f"r3_{al}_tmin", pct(tt[lo_]), src, f"min pairs.{key}.tasks[BENCH].d", "smallest per-task difference (pp)")
        put(f"r3_{al}_tmin_task", TASK[lo_], src, "argmin", "task with the smallest difference")
        put(f"r3_{al}_tmax", pct(tt[hi_]), src, f"max pairs.{key}.tasks[BENCH].d", "largest per-task difference (pp)")
        put(f"r3_{al}_tmax_task", TASK[hi_], src, "argmax", "task with the largest difference")
        put(f"r3_{al}_npos", word(sum(v > 0 for v in tt.values())), src, "count d>0", "tasks with a positive point estimate")
    for agg, f, x in (("bench", "task_mean", ""), ("human", "task_mean", "h"), ("human", "pooled", "hpool")):
        a_, b_ = P["v3_27-stock27"][agg][f + "_d"], P["stock27-stock9"][agg][f + "_d"]
        dci = E["did"][{"": "r3_did27_tm_ci", "h": "r3_did27_htm_ci", "hpool": "r3_did27_hpool_ci"}[x]]   # difference of differences on shared resamples: wording follows its CI
        put(f"r3_specscale_{x or 'tm'}", "larger than" if dci[0] > 0 else "smaller than" if dci[1] < 0 else "not detectably different from", src,
            f"did.r3_did27_{x or 'tm'}_ci", "specialization (27B) vs scale")
    pr = P["v3_27-stock27"]["tasks"]
    up = [t for t in BENCH if pr[t].get("p_holm", 1) < 0.05 and pr[t]["d"] > 0]
    dn = [t for t in BENCH if pr[t].get("p_holm", 1) < 0.05 and pr[t]["d"] < 0]
    put("r3_prim_nup", word(len(up)), REL("eval"), "pairs.v3_27-stock27.tasks[*].p_holm<0.05, d>0", "tasks with a Holm-significant gain")
    put("r3_prim_ndown", word(len(dn)), REL("eval"), "pairs.v3_27-stock27.tasks[*].p_holm<0.05, d<0", "tasks with a Holm-significant loss")
    lst = lambda ts: ", ".join(TASK[t] for t in ts) if ts else "none"
    put("r3_prim_up_list", lst(up), src, "", "tasks with a Holm-significant gain"); put("r3_prim_down_list", lst(dn), src, "", "tasks with a Holm-significant loss")
    part = lambda ts, w: (f"{w} on {word(len(ts))} task{'s' if len(ts) != 1 else ''} ({lst(ts)})" if ts else f"{w} on none")
    put("r3_prim_holm", part(up, "more accurate") + (" and " + part(dn, "less accurate") if dn else " and less accurate on no task"), src, "p_holm < 0.05 by sign", "Holm summary clause")
    # decomposition of the primary point estimate (no interval: the task mean is linear in the per-task differences)
    d = {t: pr[t]["d"] for t in BENCH}; tot = sum(d.values())
    put("r3_prim_share_new", num(100 * sum(d[t] for t in NEW_SRC) / tot, 0) if tot else "--", src, "sum d[NEW_SRC] / sum d[BENCH]", "share of the task-mean gain from the tasks added in v3 (%)")
    put("r3_prim_old_tm", pct(sum(d[t] for t in BENCH if t not in NEW_SRC) / (len(BENCH) - len(NEW_SRC))), src, "mean d over pilot tasks", "task-mean difference without the v3 tasks (pp)")
    put("r3_prim_know_tm", pct(sum(d[t] for t in KNOW) / len(KNOW)), src, "mean d over knowledge tasks", "task-mean difference, knowledge tasks (pp)")
    TRAINED = [t for t in BENCH if t not in EVAL_ONLY]
    put("r3_nevalonly", word(len(EVAL_ONLY)), "build_v3.EVAL_ONLY", "", "benchmark tasks from sources used only for evaluation")
    put("r3_ntrained", word(len(TRAINED)), "build_v3.EVAL_ONLY", "", "benchmark tasks from sources with a training split")
    put("r3_prim_evalonly_tm", pct(sum(d[t] for t in EVAL_ONLY) / len(EVAL_ONLY)), src, "mean d over EVAL_ONLY", "task-mean difference, evaluation-only sources (pp)")
    put("r3_prim_trained_tm", pct(sum(d[t] for t in TRAINED) / len(TRAINED)), src, "mean d over the other tasks", "task-mean difference, sources with a training split (pp)")
    if "eval_raw" in J:   # prespecified primary outcome without the post hoc RadCases panel prior correction
        rb = J["eval_raw"]["pairs"]["v3_27-stock27"]["bench"]
        put("r3_prim_tm_raw", pct(rb["task_mean_d"]), REL("eval_raw"), "pairs.v3_27-stock27.bench.task_mean_d", "primary outcome without the prior correction (pp)")
        put("r3_prim_tm_raw_ci", pci(rb["task_mean_ci"]), REL("eval_raw"), "pairs.v3_27-stock27.bench.task_mean_ci", "95% CI")
    else:
        pending("r3_prim_tm_raw", "eval_raw"); pending("r3_prim_tm_raw_ci", "eval_raw")
    # specialization vs scale as a difference of differences (point estimates; intervals need the shared resamples -> eval job)
    for agg, x in (("bench", ""), ("human", "h")):
        sp27, sp9, sc = (P[k][agg]["task_mean_d"] for k in ("v3_27-stock27", "v3_9-stock9", "stock27-stock9"))
        put(f"r3_did27_{x}tm", pct(sp27 - sc), src, f"pairs: (v3_27-stock27) - (stock27-stock9), {agg} task mean", "specialization minus scale, 27B (pp)")
        put(f"r3_did9_{x}tm", pct(sp9 - sc), src, f"pairs: (v3_9-stock9) - (stock27-stock9), {agg} task mean", "specialization minus scale, 9B (pp)")
        sp27p, sp9p, scp = (P[k][agg]["pooled_d"] for k in ("v3_27-stock27", "v3_9-stock9", "stock27-stock9"))
        put(f"r3_did27_{x}pool", pct(sp27p - scp), src, f"pairs pooled, {agg}", "specialization minus scale, 27B, pooled (pp)")
        put(f"r3_did9_{x}pool", pct(sp9p - scp), src, f"pairs pooled, {agg}", "specialization minus scale, 9B, pooled (pp)")
    for k in ("r3_did27_tm_ci", "r3_did27_htm_ci", "r3_did9_tm_ci", "r3_did27_hpool_ci"):
        if "did" in E: put(k, pci(E["did"][k]), src, f"did.{k}", "95% CI")
        else: pending(k, "difference-of-differences intervals (add to eval_v3 ANALYSE)")
    # CT-RATE: agreement with the classifier
    for s in ("v3_27", "stock27", "v3_9", "stock9", "qwen38", "medgemma_fix"):
        if s in S and all(t in S[s]["tasks"] for t in CTRATE):
            put(f"r3_ct_{s}", pct(sum(S[s]["tasks"][t]["acc"] for t in CTRATE) / 3), src, f"mean systems.{s}.tasks[ctrate:*].acc", f"{NAME[s]}, CT-RATE agreement, mean of 3 tasks")
    # calibration (human-assigned questions)
    for s in SYSTEMS:
        if s not in CAL: continue
        for cond, p_ in (("as_scored", "cal"), ("recalibrated", "rc")):
            c = CAL[s]["human"][cond]
            put(f"r3_{p_}_{s}_ece", num(c["ece"], 3), src, f"calibration.{s}.human.{cond}.ece", "expected calibration error")
            put(f"r3_{p_}_{s}_brier", num(c["brier"], 3), src, f"calibration.{s}.human.{cond}.brier", "Brier score")
            put(f"r3_{p_}_{s}_ce", pct(c["conf_err"]), src, f"calibration.{s}.human.{cond}.conf_err", "confident errors (%)")
            put(f"r3_{p_}_{s}_cov", pct(c["cov5"]), src, f"calibration.{s}.human.{cond}.cov5", "coverage at 5% error (%)")
    for s in SYSTEMS:   # coverage over all benchmark questions each system answered (abstract)
        if s in CAL and "bench" in CAL[s]:
            put(f"r3_bcov_{s}", pct(CAL[s]["bench"]["as_scored"]["cov5"]), src, f"calibration.{s}.bench.as_scored.cov5", "coverage at 5% error, all benchmark questions (%)")
    c3, ck = CAL["v3_27"]["human"]["as_scored"], CAL["stock27"]["human"]["as_scored"]
    put("r3_calp_cov", pct(c3["cov5"] - ck["cov5"]), src, "cov5 v3_27 - stock27", "coverage difference (pp, point)")
    if "calibration_pairs" in E:
        put("r3_calp_cov_ci", pci(E["calibration_pairs"]["v3_27-stock27"]["cov5_ci"]), src, "calibration_pairs", "95% CI")
    else:
        pending("r3_calp_cov_ci", "calibration-difference intervals (add to eval_v3 ANALYSE)")
    # wording
    W = E.get("wording", {}).get("v3_27-stock27")
    if W and not all(k in W for k in ("heldout", "seen")): W = None   # both wordings needed (eval_v3 of 10-06 has "seen" only)
    if W:
        for k, x in (("heldout", "held"), ("seen", "seen")):
            put(f"r3_wd_{x}", pct(W[k]["d"]), src, f"wording.v3_27-stock27.{k}.d", f"gain on {k} wordings (pp)")
            put(f"r3_wd_{x}_ci", pci(W[k]["ci"]), src, f"wording.v3_27-stock27.{k}.ci", "95% CI")
            put(f"r3_wd_{x}_n", n_(W[k]["n"]), src, f"wording.v3_27-stock27.{k}.n", "questions")
        put("r3_wd_did", pct(W["did"]["d"]) if "did" in W else pct(W["heldout"]["d"] - W["seen"]["d"]), src, "wording.v3_27-stock27.did.d", "withheld minus seen (pp)")
        if "did" in W: put("r3_wd_did_ci", pci(W["did"]["ci"]), src, "wording.v3_27-stock27.did.ci", "95% CI")
        else: pending("r3_wd_did_ci", "wording difference interval")
        W9 = E["wording"].get("v3_9-stock9", {})
        if all(k in W9 for k in ("heldout", "seen", "did")):
            for k, x in (("heldout", "held"), ("seen", "seen")):
                put(f"r3_wd9_{x}", pct(W9[k]["d"]), src, f"wording.v3_9-stock9.{k}.d", f"9B gain on {k} wordings (pp)")
                put(f"r3_wd9_{x}_ci", pci(W9[k]["ci"]), src, f"wording.v3_9-stock9.{k}.ci", "95% CI")
            put("r3_wd9_did", pct(W9["did"]["d"]), src, "wording.v3_9-stock9.did.d", "9B withheld minus seen (pp)")
            put("r3_wd9_did_ci", pci(W9["did"]["ci"]), src, "wording.v3_9-stock9.did.ci", "95% CI")
    else:
        for k in ("r3_wd_held", "r3_wd_held_ci", "r3_wd_seen", "r3_wd_seen_ci", "r3_wd_did", "r3_wd_held_n", "r3_wd_seen_n"): pending(k, "wording split in eval_v3")
    return True


# ------------------------------------------------------------------------------------------------ pre-read subspecialty
PR_SYS = ["v3_27", "stock27", "v3_9", "stock9", "qwen38", "medgemma_fix"]


def build_preread():
    R = J.get("preread"); src = REL("preread")
    keys = ["r3_pr_n", "r3_pr_v3_27_full", "r3_pr_v3_27_pre", "r3_pr_stock27_full", "r3_pr_stock27_pre", "r3_pr_qwen38_full",
            "r3_pr_qwen38_pre", "r3_pr_medgemma_fix_full", "r3_pr_medgemma_fix_pre", "r3_pr_minchg", "r3_pr_maxchg",
            "r3_prp_prim_full", "r3_prp_prim_full_ci", "r3_prp_prim_pre", "r3_prp_prim_pre_ci", "r3_prp_q_pre", "r3_prp_q_pre_ci", "r3_pr_v3_27_chg", "r3_pr_v3_27_chg_ci"]
    if not R:
        for k in keys: pending(k, "pre-read job (preread_route.json)")
        return False
    put("r3_pr_n", n_(R["n_questions"]), src, "n_questions", "subspecialty questions, both conditions")
    for m, v in R["models"].items():
        if m not in NAME: continue
        put(f"r3_pr_{m}_full", pct(v["full"]), src, f"models.{m}.full", f"{NAME[m]}, full state")
        put(f"r3_pr_{m}_pre", pct(v["preread"]), src, f"models.{m}.preread", f"{NAME[m]}, imaging findings removed")
        put(f"r3_pr_{m}_chg", pct(v["change"]), src, f"models.{m}.change", "change (pp)")
        put(f"r3_pr_{m}_chg_ci", pci(v["change_ci"]), src, f"models.{m}.change_ci", "95% CI")
    ch = [v["change"] for m, v in R["models"].items() if m in NAME]
    put("r3_pr_minchg", pct(-max(ch)), src, "min |change|", "smallest loss (pp)"); put("r3_pr_maxchg", pct(-min(ch)), src, "max |change|", "largest loss (pp)")
    for al, key in (("prim", "v3_27-stock27"), ("q", "v3_27-qwen38"), ("mg", "v3_27-medgemma_fix"), ("spec9", "v3_9-stock9")):
        if key not in R["pairs"]: continue
        for c, x in (("full", "full"), ("preread", "pre")):
            put(f"r3_prp_{al}_{x}", pct(R["pairs"][key][c]["diff"]), src, f"pairs.{key}.{c}.diff", f"{key}, {c} (pp)")
            put(f"r3_prp_{al}_{x}_ci", pci(R["pairs"][key][c]["ci"]), src, f"pairs.{key}.{c}.ci", "95% CI")
    return True


# ------------------------------------------------------------------------------------------------ inputs still to come
def build_training():
    """Wall time (wall_seconds) and fitted temperature (dev_cal temperature). Peak memory stays pending: under data parallelism
    only rank 0 writes training_metrics, and peak_device_bytes may cover one device only (interpretation pending from the job owner)."""
    for m in ("27", "9"):
        T, C = J.get(f"train{m}"), J.get(f"cal{m}")
        if T:
            put(f"r3_hours_{m}", num(T["wall_seconds"] / 3600), REL(f"train{m}"), "wall_seconds / 3600", f"training wall time, {m}B (h)")
            M = {r["key"]: r["value"] for r in csv.DictReader(open(RW / "numbers.csv"))}
            if not MOCK and f"v3_hours_{m}b" in M: assert M[f"v3_hours_{m}b"] == num(T["wall_seconds"] / 3600), ("training hours differ from Methods", m)
        else: pending(f"r3_hours_{m}", f"training summary {m}B")
        if C: put(f"r3_temp_{m}", num(C["temperature"], 2), REL(f"cal{m}"), "temperature", f"fitted temperature, {m}B")
        else: pending(f"r3_temp_{m}", f"dev_cal summary {m}B")
        # peak_device_bytes is rank 0's primary device only (verified by the job owner, 2026-10-06): report it as such, never as "per GPU"
        if T and "peak_device_bytes" in T:
            put(f"r3_gib_{m}", num(T["peak_device_bytes"] / 2 ** 30), REL(f"train{m}"), "peak_device_bytes / 2^30", f"peak allocated memory on the first GPU of rank 0, {m}B (GiB)")
        else: pending(f"r3_gib_{m}", f"peak memory {m}B")
        if T and "requested_records" in T:
            put(f"r3_records_{m}", n_(T["requested_records"]), REL(f"train{m}"), "requested_records", f"training records per epoch, {m}B (not records_seen)")
            M = {r["key"]: r["value"] for r in csv.DictReader(open(RW / "numbers.csv"))}
            if not MOCK and "v3r_accepted" in M: assert n_(T["requested_records"]) == M["v3r_accepted"], ("training records differ from Methods", T["requested_records"], M["v3r_accepted"])


def build_odec():
    """OpenAI Decisions API: refusals, cost, end-to-end latency from the node (4 requests in flight; 27 sequential probes), cov5 pair."""
    keys = ["r3_od_refusals", "r3_od_cost", "r3_lt_od", "r3_lt_od_n", "r3_lt_od_seq", "r3_lt_od_seq_n", "r3_lt_od_server", "r3_od_cov_d", "r3_od_cov_ci"]
    S, L, E = J.get("odec_status"), J.get("odec_lat"), J.get("eval")
    if not (S and L and E and "openai_dec" in E["systems"]): return _pend_all(keys, "OpenAI Decisions API")
    put("r3_od_refusals", n_(S["refusals"]["questions"]), REL("odec_status"), "refusals.questions", "questions refused by the API (scored as uniform)")
    put("r3_od_cost", f"{S['spent']['usd']:.2f}", REL("odec_status"), "spent.usd", "API cost of the benchmark run (USD)")
    c, q = L["concurrent"], L["sequential"]; src = REL("odec_lat")
    put("r3_lt_od", f"{c['per_question_ms']['median']:.0f}", src, "concurrent.per_question_ms.median", "end-to-end ms per question, 4 requests in flight")
    put("r3_lt_od_n", n_(c["per_question_ms"]["n"]), src, "concurrent.per_question_ms.n", "requests")
    put("r3_lt_od_seq", f"{q['per_question_ms']['median']:.0f}", src, "sequential.per_question_ms.median", "end-to-end ms per question, one request at a time")
    put("r3_lt_od_seq_n", n_(q["per_question_ms"]["n"]), src, "sequential.per_question_ms.n", "sequential requests")
    put("r3_lt_od_server", f"{c['server_ms']['median']:.0f}", src, "concurrent.server_ms.median", "server-reported ms per request")
    cp = E["calibration_pairs"].get("v3_27-openai_dec")
    if cp:
        put("r3_od_cov_d", pct(cp["cov5_d"]), REL("odec_eval"), "calibration_pairs.v3_27-openai_dec.cov5_d", "coverage difference, human-assigned (pp)")
        put("r3_od_cov_ci", pci(cp["cov5_ci"]), REL("odec_eval"), "calibration_pairs.v3_27-openai_dec.cov5_ci", "95% CI")
    else: _pend_all(["r3_od_cov_d", "r3_od_cov_ci"], "OpenAI coverage pair")


def build_rcprior():
    """RadCases panel question before and after the prior correction (the corrected accuracy is the eval default)."""
    D = J.get("rcprior"); keys = ["r3_rcp_raw_rk27", "r3_rcp_raw_rk9", "r3_rcp_none_panel_rk27", "r3_rcp_none_panel_rk9", "r3_rcp_none_none_rk27", "r3_rcp_none_none_rk9"]
    if not D: return _pend_all(keys, "RadCases prior correction")
    src = REL("rcprior"); S = D["systems"]; E = J.get("eval")
    for m in ("v3_27", "v3_9"):   # the job's corrected accuracy must be the one the evaluation now uses
        assert abs(S[m + "_corrected"]["acc"] - E["systems"][m]["tasks"]["radcases_panel"]["acc"]) < 1e-9, f"{m}: radcases_prior and eval_v3_prior disagree"
    for k, m, f, d in (("r3_rcp_raw_rk27", "v3_27", "acc", "accuracy as scored (%)"), ("r3_rcp_raw_rk9", "v3_9", "acc", "accuracy as scored (%)"),
                       ("r3_rcp_none_panel_rk27", "v3_27_corrected", "none_picked_on_panel_keys", "% selecting 'no topic applies' when a panel is the key, corrected"),
                       ("r3_rcp_none_panel_rk9", "v3_9_corrected", "none_picked_on_panel_keys", "% selecting 'no topic applies' when a panel is the key, corrected"),
                       ("r3_rcp_none_none_rk27", "v3_27_corrected", "acc_on_none_keys", "accuracy when 'no topic applies' is the key, corrected (%)"),
                       ("r3_rcp_none_none_rk9", "v3_9_corrected", "acc_on_none_keys", "accuracy when 'no topic applies' is the key, corrected (%)")):
        put(k, pct(S[m][f]), src, f"systems.{m}.{f}", d)


def build_rcdiag():
    """RadCases panel question: how often each system selects the option that no ACR topic applies, by key type."""
    D = J.get("rcdiag"); keys = ["r3_rc_none_panel_rk27", "r3_rc_none_panel_k27", "r3_rc_none_none_rk27", "r3_rc_none_none_k27", "r3_rc_none_none_q",
                                 "r3_rc_none_none_mg", "r3_rc_npanel", "r3_rc_nnone", "r3_rc_train_none", "r3_rc_train_q", "r3_rc_train_none_pct",
                                 "r3_rc_none_panel_rk9", "r3_rc_none_panel_k9", "r3_rc_none_none_rk9"]
    if not D: return _pend_all(keys, "RadCases panel diagnosis")
    src = REL("rcdiag"); S = D["systems"]
    assert all(v["key_or_label_mismatch"] == 0 for v in D["fairness"].values())
    for k, m, kind in (("r3_rc_none_panel_rk27", "v3_27", "panel"), ("r3_rc_none_panel_k27", "stock27", "panel"), ("r3_rc_none_panel_rk9", "v3_9", "panel"),
                       ("r3_rc_none_panel_k9", "stock9", "panel"), ("r3_rc_none_none_rk27", "v3_27", "None"), ("r3_rc_none_none_k27", "stock27", "None"),
                       ("r3_rc_none_none_q", "qwen38", "None"), ("r3_rc_none_none_rk9", "v3_9", "None"), ("r3_rc_none_none_mg", "medgemma_fix", "None")):
        put(k, pct(S[m]["pick_none_rate"][kind]), src, f"systems.{m}.pick_none_rate.{kind}", f"% selecting 'no topic applies', key = {kind}")
    put("r3_rc_npanel", n_(S["v3_27"]["acc"]["panel"]["n"]), src, "systems.v3_27.acc.panel.n", "panel questions whose key is a panel")
    put("r3_rc_nnone", n_(S["v3_27"]["acc"]["None"]["n"]), src, "systems.v3_27.acc.None.n", "panel questions whose key is 'no topic applies'")
    tr = D["radcases_v3_splits"]["train"]; tn = sum(v for k, v in tr.items() if k.endswith("/None")); tq = sum(tr.values())
    put("r3_rc_train_none", n_(tn), src, "radcases_v3_splits.train.*/None", "training panel questions keyed 'no topic applies'")
    put("r3_rc_train_q", n_(tq), src, "radcases_v3_splits.train.*", "training panel questions")
    put("r3_rc_train_none_pct", num(100 * tn / tq, 0), src, "ratio", "% of training panel questions keyed 'no topic applies'")


def _pend_all(keys, what):
    for k in keys: pending(k, what)


def build_transfer():
    """Kev's out-of-domain transfer suite: each specialized model against the released model it started from (general decision skill)."""
    T = J.get("transfer"); tk = ["r3_tr_9", "r3_tr_9_ci", "r3_tr_27", "r3_tr_27_ci", "r3_transfer_n"]
    if not T: return _pend_all(tk, "transfer suite with v3")
    st = REL("transfer"); P = T["pairs"]
    for k, key in (("r3_tr_9", "v3_9-stock9"), ("r3_tr_27", "v3_27-stock27")):
        x = P[key]["acc_micro"]
        put(k, pct(x["delta"]), st, f"pairs.{key}.acc_micro.delta", "change on Kev's transfer suite (pp)")
        put(k + "_ci", pci(x["ci95"]), st, f"pairs.{key}.acc_micro.ci95", "95% CI")
    put("r3_transfer_n", n_(T["models"]["stock9"]["n"]), st, "models.stock9.n", "transfer-suite questions")


def build_latency():
    L = J.get("latency"); keys = ["r3_lt_rk27", "r3_lt_rk9", "r3_lt_k27", "r3_lt_q", "r3_lt_qgen", "r3_lt_qthink", "r3_lt_ratio_gen", "r3_lt_ratio_think",
                                  "r3_lat_records", "r3_thr_rk27_hr", "r3_thr_rk9_hr", "r3_lt_qthink_n"]
    if not L: return _pend_all(keys, "latency with v3")
    src = REL("latency"); M = L["models"]; med = lambda d: d["per_question_ms"]["median"]
    put("r3_lt_rk27", f"{med(M['v3_27']):.0f}", src, "models.v3_27.per_question_ms.median", "ms")
    put("r3_lt_k27", f"{med(M['stock27']):.0f}", src, "models.stock27.per_question_ms.median", "ms")
    if "v3_9" in M: put("r3_lt_rk9", f"{med(M['v3_9']):.0f}", src, "models.v3_9.per_question_ms.median", "ms")
    else: pending("r3_lt_rk9", "latency of RadKev-9B v3")
    q = M["qwen38"]
    put("r3_lt_q", f"{med(q['letter']):.0f}", src, "models.qwen38.letter.per_question_ms.median", "ms, letter logits")
    put("r3_lt_qgen", f"{med(q['direct']):.0f}", src, "models.qwen38.direct.per_question_ms.median", "ms, generated letter")
    put("r3_lt_qthink", num(med(q["reasoning"]) / 1000), src, "models.qwen38.reasoning.per_question_ms.median", "s, reasoning")
    put("r3_lt_qthink_n", n_(q["reasoning"]["per_question_ms"]["n"]), src, "models.qwen38.reasoning.per_question_ms.n", "questions timed with reasoning")
    put("r3_lt_ratio_gen", num(med(q["direct"]) / med(M["v3_27"])), src, "direct / v3_27 median", "x")
    put("r3_lt_ratio_think", f"{med(q['reasoning']) / med(M['v3_27']):.0f}", src, "reasoning / v3_27 median", "x")
    put("r3_lat_records", n_(L["sample"]["records"]), src, "sample.records", "records timed")
    put("r3_thr_rk27_hr", n_(round(3.6e6 / med(M["v3_27"]), -3)), src, "3.6e6 / v3_27 median", "questions per hour, sequential")
    if "v3_9" in M: put("r3_thr_rk9_hr", n_(round(3.6e6 / med(M["v3_9"]), -3)), src, "3.6e6 / v3_9 median", "questions per hour, sequential")
    else: pending("r3_thr_rk9_hr", "latency of RadKev-9B v3")
    # the rule of the manuscript: never claim a decision model is faster per question than the letter-scored LLM
    LAT_FACTS.update(letter_vs_rk27=med(q["letter"]) / med(M["v3_27"]))
    rows = [("RadKev-27B", med(M["v3_27"])), ("Kev-27B", med(M["stock27"]))] + ([("RadKev-9B", med(M["v3_9"]))] if "v3_9" in M else []) + \
           [("Qwen3.8-27B, letter logits", med(q["letter"])), ("Qwen3.8-27B, generated letter", med(q["direct"])), ("Qwen3.8-27B, reasoning", med(q["reasoning"]))]
    if J.get("odec_lat"): rows.insert(3, ("OpenAI Decisions (network)", J["odec_lat"]["sequential"]["per_question_ms"]["median"]))   # one request at a time, like the others
    L["rows"] = rows


LAT_FACTS = {}


def build_reasoning():
    R = (J.get("reasoning") or {}).get("paired"); keys = ["r3_rs_q", "r3_rs_qwen38", "r3_rs_qwen38_think", "r3_rs_v3_27", "r3_rs_medgemma_think",
                                                         "r3_rp_rk_qthink", "r3_rp_rk_qthink_ci", "r3_rp_rk_qthink_tm", "r3_rp_rk_qthink_tm_ci", "r3_rp_qthink", "r3_rp_qthink_ci", "r3_rs_ntk", "r3_rp_rk_qthink_tm14", "r3_rp_rk_qthink_tm14_ci"]
    if not R: return _pend_all(keys, "reasoning sample with v3")
    src = REL("reasoning"); S, P = R["systems"], R["pairs"]
    put("r3_rs_q", n_(S["v3_27"]["bench"]["n"]), src, "paired.systems.v3_27.bench.n", "questions in the reasoning sample")
    put("r3_rs_ntk", word(S["v3_27"]["bench"]["tasks_k"]), src, "paired.systems.v3_27.bench.tasks_k", "tasks of the reasoning sample, examination tasks pooled")
    for k, m in (("r3_rs_qwen38", "qwen38"), ("r3_rs_qwen38_think", "qwen38_think"), ("r3_rs_v3_27", "v3_27"), ("r3_rs_medgemma_think", "medgemma_think")):
        if m in S: put(k, pct(S[m]["bench"]["pooled"]), src, f"paired.systems.{m}.bench.pooled", "accuracy on the sample (%)")
        else: pending(k, f"reasoning sample: {m}")
    for k, key in (("r3_rp_rk_qthink", "v3_27-qwen38_think"), ("r3_rp_qthink", "qwen38_think-qwen38")):
        x = P[key]["bench"]
        put(k, pct(x["pooled_d"]), src, f"paired.pairs.{key}.bench.pooled_d", "difference, pooled (pp)")
        put(k + "_ci", pci(x["pooled_ci"]), src, f"paired.pairs.{key}.bench.pooled_ci", "95% CI")
        if k == "r3_rp_rk_qthink":
            put(k + "_tm", pct(x["task_mean_k_d"]), src, f"paired.pairs.{key}.bench.task_mean_k_d", "difference, task mean, examination tasks pooled (pp)")
            put(k + "_tm_ci", pci(x["task_mean_k_ci"]), src, f"paired.pairs.{key}.bench.task_mean_k_ci", "95% CI")
            put(k + "_tm14", pct(x["task_mean_d"]), src, f"paired.pairs.{key}.bench.task_mean_d", "difference, task mean over the 14 unpooled tasks (pp)")
            put(k + "_tm14_ci", pci(x["task_mean_ci"]), src, f"paired.pairs.{key}.bench.task_mean_ci", "95% CI")


def build_blind():
    B = J.get("blind"); keys = ["r3_bl_dx_rk27", "r3_bl_dx_k27", "r3_bl_dx_gain_blind", "r3_bl_dx_gain_blind_ci", "r3_bl_dx_gain_full", "r3_bl_dx_gain_full_ci",
                                "r3_bl_rsna_rk27", "r3_bl_rsna_k27", "r3_bl_rsna_q", "r3_bl_dx_q", "r3_bl_aw_flag", "r3_bl_aw_flag_ci", "r3_bl_aw_none", "r3_bl_aw_none_ci"]
    if not B: return _pend_all(keys, "options-only control with v3")
    src = REL("blind"); T = B["tasks"]
    for t, x in (("eurorad_dx", "dx"), ("rsna_radioqa", "rsna")):
        M = T[t]["models"]
        put(f"r3_bl_{x}_rk27", pct(M["v3_27"]["blind"]), src, f"tasks.{t}.models.v3_27.blind", "options-only accuracy (%)")
        put(f"r3_bl_{x}_k27", pct(M["stock27"]["blind"]), src, f"tasks.{t}.models.stock27.blind", "options-only accuracy (%)")
        if "qwen38" in M: put(f"r3_bl_{x}_q", pct(M["qwen38"]["blind"]), src, f"tasks.{t}.models.qwen38.blind", "options-only accuracy (%)")
        else: pending(f"r3_bl_{x}_q", "options-only control: Qwen3.8-27B rerun")
        for m, k in (("v3_27", "rk27"), ("stock27", "k27"), ("v3_9", "rk9"), ("stock9", "k9"), ("medgemma_fix", "mg"), ("qwen38", "q")):
            if m in M: put(f"r3_bl_{x}_{k}_full", pct(M[m]["full"]), src, f"tasks.{t}.models.{m}.full", "accuracy with the case (%)")
        if "medgemma_fix" in M: put(f"r3_bl_{x}_mg", pct(M["medgemma_fix"]["blind"]), src, f"tasks.{t}.models.medgemma_fix.blind", "options-only accuracy (%)")
        for m, k in (("v3_9", "rk9"), ("stock9", "k9")):
            if m in M: put(f"r3_bl_{x}_{k}", pct(M[m]["blind"]), src, f"tasks.{t}.models.{m}.blind", "options-only accuracy (%)")
        pr_ = T[t]["pairs"]["v3_27-stock27"]
        if t == "rsna_radioqa":
            for c in ("full", "blind"):
                put(f"r3_bl_rsna_gain_{c}", pct(pr_[c]), src, f"tasks.{t}.pairs.v3_27-stock27.{c}", f"gain, {c} (pp)")
                put(f"r3_bl_rsna_gain_{c}_ci", pci(pr_[c + "_ci"]), src, f"tasks.{t}.pairs.v3_27-stock27.{c}_ci", "95% CI")
    pr = T["eurorad_dx"]["pairs"]["v3_27-stock27"]
    put("r3_bl_dx_gain_blind", pct(pr["blind"]), src, "tasks.eurorad_dx.pairs.v3_27-stock27.blind", "gain without the case (pp)")
    put("r3_bl_dx_gain_blind_ci", pci(pr["blind_ci"]), src, "...blind_ci", "95% CI")
    put("r3_bl_dx_gain_full", pct(pr["full"]), src, "tasks.eurorad_dx.pairs.v3_27-stock27.full", "gain with the case (pp)")
    put("r3_bl_dx_gain_full_ci", pci(pr["full_ci"]), src, "...full_ci", "95% CI")
    put("r3_bl_dx_did", pct(pr["did"]), src, "tasks.eurorad_dx.pairs.v3_27-stock27.did", "gain with minus without the case (pp)")
    put("r3_bl_dx_did_ci", pci(pr["did_ci"]), src, "tasks.eurorad_dx.pairs.v3_27-stock27.did_ci", "95% CI")
    p9 = T["eurorad_dx"]["pairs"].get("v3_9-stock9")
    if p9:
        for c in ("full", "blind"):
            put(f"r3_bl_dx9_gain_{c}", pct(p9[c]), src, f"tasks.eurorad_dx.pairs.v3_9-stock9.{c}", f"9B gain, {c} (pp)")
    sp = T["eurorad_dx"]["split"]
    for k, nm in (("r3_bl_aw_flag", "answer_word_in_case"), ("r3_bl_aw_none", "no_answer_word")):
        x = sp[nm]["pairs"]["v3_27-stock27"]
        put(k, pct(x["full"]), src, f"tasks.eurorad_dx.split.{nm}.pairs.v3_27-stock27.full", "gain (pp)")
        put(k + "_ci", pci(x["full_ci"]), src, f"...{nm}...full_ci", "95% CI")
        put(k + "_n", n_(sp[nm]["n"]), src, f"tasks.eurorad_dx.split.{nm}.n", "questions")


def build_radgraph():
    G = (J.get("radgraph") or {}).get("radgraph_xl"); keys = ["r3_rg_n", "r3_rg_v3_27", "r3_rg_stock27", "r3_rg_qwen38", "r3_rg_medgemma_fix", "r3_rg_d", "r3_rg_d_ci",
                                                               "r3_rg_d9", "r3_rg_d9_ci", "r3_rg_dq", "r3_rg_dq_ci"]
    if not G: return _pend_all(keys, "RadGraph-XL with v3")
    src = REL("radgraph")
    put("r3_rg_n", n_(G["n"]["all"]), src, "radgraph_xl.n.all", "questions")
    for m in ("v3_27", "stock27", "qwen38", "medgemma_fix", "v3_9", "stock9"):
        if m in G["models"]: put(f"r3_rg_{m}", pct(G["models"][m]["all"]["acc"]), src, f"radgraph_xl.models.{m}.all.acc", "accuracy (%)")
    for k, key in (("r3_rg_d", "v3_27-stock27"), ("r3_rg_d9", "v3_9-stock9"), ("r3_rg_dq", "v3_27-qwen38")):
        put(k, pct(G["pairs"][key]["all"]["d"]), src, f"radgraph_xl.pairs.{key}.all.d", "difference (pp)")
        put(k + "_ci", pci(G["pairs"][key]["all"]["ci"]), src, f"radgraph_xl.pairs.{key}.all.ci", "95% CI")
    X = J.get("rgxerr")   # hedged findings (RadGraph-XL 'uncertain') and the status answers of the training data
    if X:
        sx = REL("rgxerr")
        for m in ("v3_27", "stock27", "qwen38"):
            put(f"r3_rgu_{m}", pct(X["models"][m]["acc_by_gold"]["uncertain"]), sx, f"models.{m}.acc_by_gold.uncertain", "accuracy on uncertain (hedged) findings (%)")
        put("r3_rgu_share", pct(X["gold_dist"]["uncertain"] / sum(X["gold_dist"].values())), sx, "gold_dist.uncertain / n", "share of uncertain keys (%)")
        T = X["train_status_labels"]["train"]["teacher_cxr_findings_finding_status"]
        put("r3_rgu_train_share", pct(T["uncertain"] / sum(T.values())), sx, "train_status_labels.train.uncertain / n", "share of uncertain among training status answers (%)")
        put("r3_rgu_qwin_unc", n_(X["pairs"]["v3_27-qwen38"]["only_b_right_by_gold"]["uncertain"]), sx, "pairs.v3_27-qwen38.only_b_right_by_gold.uncertain", "questions only Qwen answered correctly, uncertain key")
        put("r3_rgu_qwin", n_(X["pairs"]["v3_27-qwen38"]["only_b_right"]), sx, "pairs.v3_27-qwen38.only_b_right", "questions only Qwen answered correctly")
    else:
        _pend_all(["r3_rgu_v3_27", "r3_rgu_stock27", "r3_rgu_qwen38", "r3_rgu_share", "r3_rgu_train_share", "r3_rgu_qwin_unc", "r3_rgu_qwin"], "RadGraph-XL per-status job")


AS_SYS = [("v3_27", "RadKev-27B", "rk27"), ("stock27", "Kev-27B", "k27"), ("v3_9", "RadKev-9B", "rk9"), ("stock9", "Kev-9B", "k9"),
          ("qwen38_num", "Qwen3.8-27B", "q"), ("medgemma_fix_num", "MedGemma-27B-text", "mg")]
AS_K = [2, 4, 16, 64, 255]
AS_WARMUP = 5   # jobs/answer_space3.py CFG["lat_warmup"]: first 5 latency requests of each model excluded


def as_summary():
    """answer_space3.json (per-question bit strings per model and condition; latency [K, ms]) -> accuracy per source x condition x K,
    median latency per K. A numbered LLM answer that is not a valid option number ("x") counts as wrong; "-" = not asked."""
    A = J.get("answer_space")
    if not A: return None
    import statistics
    src = A["src"]; out = {"n": {s_: src.count(s_) for s_ in set(src)}, "acc": {}, "lat": {}, "lat_cases": None}
    for m, _, _ in AS_SYS:
        if m not in A["models"]: continue
        for c, bitstr in A["models"][m]["correct"].items():
            for s_ in set(src):
                xs = [b for b, ss in zip(bitstr, src) if ss == s_ and b != "-"]
                if xs: out["acc"].setdefault(m, {}).setdefault(s_, {})[c] = (sum(b == "1" for b in xs) / len(xs), len(xs))
    for m, rows in A.get("latency", {}).items():
        byk = {}
        for r in rows: byk.setdefault(int(r[0]), []).append(float(r[1]))
        out["lat"][m] = {k: statistics.median(v) for k, v in sorted(byk.items())}
        if m == "v3_27":   # requests = cases x sizes, of which the first AS_WARMUP (shuffled) were excluded as warm-up
            out["lat_requests"] = len(rows); out["lat_cases"] = round((len(rows) + AS_WARMUP) / len(byk))
    return out


def build_answer_space():
    keys = ["r3_as_n_dx", "r3_as_n_rsna", "r3_as_lat_cases", "r3_as_rk27_dx_sim16", "r3_as_k27_dx_sim16", "r3_as_q_dx_sim16", "r3_as_mg_dx_sim16",
            "r3_as_rk27_dx_rand255", "r3_as_rk27_dx_sim255", "r3_as_lat_rk27_2", "r3_as_lat_rk27_255", "r3_as_lat_requests"]   # final grid: K = 2, 4, 16, 64, 255
    S = as_summary()
    if not S:
        tab_pending("v3_answer_space", "answer space with v3"); return _pend_all(keys, "answer space with v3")
    src = REL("answer_space"); acc = lambda m, s_, c: S["acc"].get(m, {}).get(s_, {}).get(c, (None, 0))[0]
    put("r3_as_n_dx", n_(S["n"].get("eurorad_dx", 0)), src, "count(src == eurorad_dx)", "answer-space questions, Eurorad diagnosis")
    put("r3_as_n_rsna", n_(S["n"].get("rsna_radioqa", 0)), src, "count(src == rsna_radioqa)", "answer-space questions, RSNA-RadioQA")
    put("r3_as_lat_cases", n_(S["lat_cases"] or 0), src, "(len(latency.v3_27) + 5 warm-up) / sizes", "latency cases")
    put("r3_as_lat_requests", n_(S.get("lat_requests") or 0), src, "len(latency.v3_27)", "timed latency requests after warm-up, RadKev-27B")
    for m, _, k in AS_SYS:
        v = acc(m, "eurorad_dx", "sim_16")
        if k in ("rk27", "k27", "q", "mg"):
            if v is None: pending(f"r3_as_{k}_dx_sim16", f"answer space: {m} sim_16")
            else: put(f"r3_as_{k}_dx_sim16", pct(v), src, f"models.{m}.correct.sim_16 (eurorad_dx)", "accuracy, 16 most similar options (%)")
    for c in ("rand_255", "sim_255"):
        v = acc("v3_27", "eurorad_dx", c)
        key = f"r3_as_rk27_dx_{c.replace('_', '')}"
        if v is None: pending(key, f"answer space: {c}")
        else: put(key, pct(v), src, f"models.v3_27.correct.{c} (eurorad_dx)", "accuracy (%)")
    for m, _, k in AS_SYS:
        if k in ("rk27", "k27", "q", "mg"):
            v = acc(m, "rsna_radioqa", "sim_16")
            if v is None: pending(f"r3_as_{k}_rsna_sim16", f"answer space: {m} rsna sim_16")
            else: put(f"r3_as_{k}_rsna_sim16", pct(v), src, f"models.{m}.correct.sim_16 (rsna_radioqa)", "accuracy, 16 most similar options (%)")
    for m, key, Ks in (("qwen38_num", "q", (2, 255)), ("qwen38_letter", "qletter", (2, 16))):
        for K in Ks:
            v = S["lat"].get(m, {}).get(K)
            if v is None: pending(f"r3_as_lat_{key}_{K}", f"answer-space latency {m} K={K}")
            else: put(f"r3_as_lat_{key}_{K}", n_(round(v)), src, f"latency.{m} median at K={K}", "ms per request")
    for s_, sk in (("eurorad_dx", "dx"), ("rsna_radioqa", "rsna")):
        for c in ("rand_255", "sim_255", "llm_16"):
            if sk == "dx" and c != "llm_16": continue   # Eurorad rand/sim 255 are put above
            v = S["acc"].get("v3_27", {}).get(s_, {}).get(c)
            if v is None: pending(f"r3_as_rk27_{sk}_{c.replace('_', '')}", f"answer space: {c}"); continue
            put(f"r3_as_rk27_{sk}_{c.replace('_', '')}", pct(v[0]), src, f"models.v3_27.correct.{c} ({s_})", "accuracy (%)")
            if c == "llm_16": put(f"r3_as_n_{sk}_llm16", n_(v[1]), src, f"count of {s_} questions with 16 LLM-written options", "questions")
    # confidence of RadKev-27B by answer space, pooled over both sources (conf: integer percent per question)
    A3 = J["answer_space"]; CM, FM = A3["models"]["v3_27"]["correct"], A3["models"]["v3_27"]["conf"]
    def cstat(c):
        xs = [(b == "1", f / 100) for b, f in zip(CM[c], FM[c]) if b != "-" and f is not None]
        hi = [a for a, f in xs if f >= 0.9]
        return sum(a for a, _ in xs) / len(xs), sum(f for _, f in xs) / len(xs), (sum(hi) / len(hi) if hi else None), len(hi) / len(xs)
    CS = {c: cstat(c) for c in CM if c in FM}; ASC.update(CS)
    a_, m_, _, sh = CS["sim_255"]
    put("r3_asc_acc_sim255", pct(a_), src, "models.v3_27.correct.sim_255 (both sources)", "accuracy, 255 most similar options, both sources (%)")
    put("r3_asc_conf_sim255", pct(m_), src, "mean models.v3_27.conf.sim_255 / 100", "mean confidence (%)")
    put("r3_asc_gap_sim255", pct(m_ - a_), src, "mean conf - accuracy", "confidence minus accuracy (pp)")
    put("r3_asc_share_sim255", num(100 * sh, 0), src, "share conf >= 0.9", "questions answered with confidence >= 0.9 (%)")
    put("r3_asc_share_rand16", num(100 * CS["rand_16"][3], 0), src, "share conf >= 0.9, rand_16", "questions answered with confidence >= 0.9 (%)")
    his = [v[2] for v in CS.values() if v[2] is not None]
    put("r3_asc_hi_lo", num(100 * min(his), 0), src, "min over conditions of accuracy at conf >= 0.9", "%"); put("r3_asc_hi_hi", num(100 * max(his), 0), src, "max", "%")
    L = S["lat"].get("v3_27", {})
    for K in (2, 255):
        if K in L: put(f"r3_as_lat_rk27_{K}", n_(round(L[K])), src, f"latency.v3_27 median at K={K}", "ms per request")
        else: pending(f"r3_as_lat_rk27_{K}", f"answer-space latency K={K}")
    # Supplementary table: accuracy (%) by source, condition and K for every system
    rows, mids = [], []
    for s_, lab in (("eurorad_dx", "Eurorad diagnosis"), ("rsna_radioqa", "RSNA-RadioQA")):
        if rows: mids.append(len(rows))
        for fam, flab in (("orig", "Original options"), ("rand", "Random"), ("sim", "Most similar"), ("llm", "LLM-written"), ("orig_llm", "Original + LLM-written")):
            for K in ([None] if fam in ("orig", "orig_llm") else (2, 4, 8, 16) if fam == "llm" else AS_K):
                c = fam if fam in ("orig", "orig_llm") else f"{fam}_{K}"
                cells = []
                for m, _, _ in AS_SYS:
                    v = acc(m, s_, c); cells.append(pct(v) if v is not None else "--")
                rows.append([lab if (fam == "orig") else "", flab if (K in (None, 2)) else "", "--" if K is None else str(K)] + cells)
    tab("v3_answer_space", "@{}lllrrrrrr@{}", "Source & Alternatives & K & " + " & ".join(SHORT.get(m.replace("_num", ""), m) for m, _, _ in AS_SYS), rows, midrules=tuple(mids))
    AS_FIG["S"] = S


AS_FIG = {}
ASC = {}   # RadKev-27B accuracy, mean confidence, accuracy at confidence >= 0.9 and its share, per answer-space condition


# ------------------------------------------------------------------------------------------------ figures
def _placeholder(ax, text):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_linestyle((0, (3, 3))); sp.set_color("#BBBBBB")
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color="#B03030", **{**F.FONT, "size": 8})


def _ticks(ax):
    """_gbar picks 1/2/5-point steps; panels spanning a wide range get 10-point steps so the labels stay legible."""
    lo, hi = ax.get_ylim(); import numpy as np
    if hi - lo > 40: ax.set_yticks([t for t in np.arange(np.ceil(lo / 10) * 10, hi + 1e-9, 10)])


def _save(fig, name):
    out = RW / "figures"; fig.savefig(out / f"{name}.pdf"); fig.savefig(out / f"{name}.png", dpi=300); F.plt.close(fig)


def _head(fig, ax, letter, title, sub=None, dx=-26, y=1.20, ysub=1.11):
    from matplotlib.transforms import offset_copy
    tr = lambda d: offset_copy(ax.transAxes, fig=fig, x=d, y=0, units="points")
    ax.text(0.0, y, letter, transform=tr(dx), ha="left", va="bottom", fontweight="bold", color=F.WP_INK, **{**F.FONT, "size": 9})
    ax.text(0.0, y, title, transform=tr(dx + 11), ha="left", va="bottom", fontweight="bold", color=F.WP_INK, **F.FONT)
    if sub: ax.text(0.0, ysub, sub, transform=tr(dx + 11), ha="left", va="bottom", color=F.WP_SUB, **{**F.FONT, "size": 7.2})


def fig_primary():
    from matplotlib.patches import Patch
    E = J.get("eval"); fig = F.plt.figure(figsize=(180 * F.MM, 128 * F.MM))
    lay = [[0.085, 0.565, 0.27, 0.31], [0.45, 0.565, 0.53, 0.31], [0.085, 0.085, 0.38, 0.31], [0.525, 0.085, 0.455, 0.31]]
    axs = [fig.add_axes(r) for r in lay]
    heads = [("a", "Averages", "Task means and all questions pooled"), ("b", GROUPS[0][0], "IU/Open-i and ReXErr"),
             ("c", GROUPS[1][0], "Eurorad, RSNA-RadioQA and RadCases"), ("d", GROUPS[2][0], "Radiology questions of the examination collections")]
    if E:
        S, P = E["systems"], E["pairs"]["v3_27-stock27"]
        avg = [("Task mean,\n15 tasks", 100 * S["stock27"]["bench"]["task_mean"], 100 * S["v3_27"]["bench"]["task_mean"], 100 * P["bench"]["task_mean_d"]),
               ("Task mean,\n14 human-\nlabeled tasks", 100 * S["stock27"]["human"]["task_mean"], 100 * S["v3_27"]["human"]["task_mean"], 100 * P["human"]["task_mean_d"]),
               ("All questions\npooled", 100 * S["stock27"]["bench"]["pooled"], 100 * S["v3_27"]["bench"]["pooled"], 100 * P["bench"]["pooled_d"])]
        F._gbar(fig, axs[0], avg, (F.WP_BASE, F.WP_ACC), delta=False, tick_size=6.8)
        for ax, (_, ts) in zip(axs[1:], GROUPS):
            F._gbar(fig, ax, [(TICK[t], 100 * S["stock27"]["tasks"][t]["acc"], 100 * S["v3_27"]["tasks"][t]["acc"], 100 * P["tasks"][t]["d"]) for t in ts],
                    (F.WP_BASE, F.WP_ACC), delta=False, tick_size=6.8)
        for ax in (axs[0], axs[2]): ax.set_ylabel("Accuracy (%)", color=F.WP_SUB, **F.FONT)
        for ax in axs: _ticks(ax)
    else:
        for ax in axs: _placeholder(ax, "pending: eval_v3")
    for ax, h in zip(axs, heads): _head(fig, ax, *h, y=1.12, ysub=1.04)
    fig.legend(handles=[Patch(color=F.WP_BASE, label="Kev-27B (released)"), Patch(color=F.WP_ACC, label="RadKev-27B (specialized)")],
               loc="upper right", ncol=2, frameon=False, prop={**F.FONT, "size": 7.5}, bbox_to_anchor=(0.99, 1.0), handlelength=1.0, handleheight=0.8)
    _save(fig, "v3_fig_primary")


def _bars3(ax, groups, series, colors):
    """groups: [(label, [v per series])]; grouped bars with value labels (whitepaper style), axis zoomed to the data."""
    import numpy as np
    vals = [v for _, vs in groups for v in vs if v is not None]; lo, hi = max(0, min(vals) - 5), max(vals) + 5
    w = 0.8 / len(series)
    for i, (lab, vs) in enumerate(groups):
        for k, (v, c) in enumerate(zip(vs, colors)):
            if v is None: continue
            x = i - 0.4 + w * (k + 0.5)
            ax.bar(x, v - lo, w * 0.94, bottom=lo, color=c, zorder=2)
            ax.text(x, v + (hi - lo) * 0.012, f"{v:.1f}", ha="center", va="bottom", color=F.WP_INK, **{**F.FONT, "size": 6.2})
    ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups], **{**F.FONT, "size": 7.2}, linespacing=1.1)
    ax.set_ylim(lo, hi); ax.set_xlim(-0.6, len(groups) - 0.4)
    step = 2 if hi - lo < 16 else 5
    ax.set_yticks([t for t in np.arange(np.ceil(lo / step) * step, hi + 1e-9, step)])
    ax.grid(axis="y", color=F.WP_GRID, lw=0.6, zorder=0)
    for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color("#BDBDBD"); ax.tick_params(axis="x", length=0); ax.tick_params(axis="y", length=0, labelsize=7.2, colors=F.WP_SUB)


def fig_llm():
    """RadKev in blue, other decision models in greys (Kev light, OpenAI Decisions slate), LLMs in muted tans; same type and grids as Figure 4."""
    from matplotlib.patches import Patch
    import matplotlib.ticker as mt
    E = J.get("eval"); fig = F.plt.figure(figsize=(180 * F.MM, 128 * F.MM))
    axa = fig.add_axes([0.175, 0.535, 0.30, 0.33]); axb = fig.add_axes([0.60, 0.535, 0.38, 0.33]); axc = fig.add_axes([0.255, 0.075, 0.60, 0.30])
    sysc = lambda s: F.C_RAD if s.startswith("v3") else F.WP_BASE if s.startswith("stock") else F.C_ODEC if s == "openai_dec" else F.C_LLM if s == "qwen38" else F.C_LLMR if s == "medgemma_fix" else F.C_DMO
    if E:
        S = MAIN(E); common = [t for t in HUMAN if all(t in S[s]["tasks"] for s in S)]
        hc = {s: 100 * sum(S[s]["tasks"][t]["acc"] for t in common) / len(common) for s in S}
        rows = [(NAME[s], hc[s], sysc(s), s.startswith("v3")) for s in sorted(hc, key=lambda s: -hc[s])]
        lo = 5 * int((min(r[1] for r in rows) - 4) // 5); hi = max(r[1] for r in rows) + 4
        F._hbars(axa, rows, lambda v: f"{v:.1f}", hi, xmin=lo); axa.set_xlim(lo, hi)
        axa.xaxis.set_major_locator(mt.MultipleLocator(5)); axa.set_xlabel("Accuracy (%), task mean", color=F.WP_SUB, **F.FONT)
        P = E["pairs"]; trio = ["v3_27", "qwen38", "medgemma_fix"]
        groups = []   # on the questions the LLMs answered: Qwen and MedGemma as scored; RadKev-27B = Qwen + paired difference
        for lab, k, f in (("Task mean\n(all tasks)", "bench", "task_mean"), ("Task mean\n(human-labeled)", "human", "task_mean"), ("All questions\npooled", "bench", "pooled")):
            q = 100 * S["qwen38"][k][f]
            groups.append((lab, [q + 100 * P["v3_27-qwen38"][k][f + "_d"], q, 100 * S["medgemma_fix"][k][f]]))
        _bars3(axb, groups, trio, [F.C_RAD, F.C_LLM, F.C_LLMR]); axb.set_ylabel("Accuracy (%)", color=F.WP_SUB, **F.FONT)
    else:
        _placeholder(axa, "pending: eval_v3"); _placeholder(axb, "pending: eval_v3")
    L = J.get("latency")
    if L and "rows" in L:
        lc = lambda lab: F.C_RAD if "RadKev" in lab else F.WP_BASE if lab.startswith("Kev") else F.C_ODEC if "OpenAI" in lab else F.C_LLMQ if "reasoning" in lab else F.C_LLM
        F._hbars(axc, [(r[0], r[1], lc(r[0]), "RadKev" in r[0]) for r in L["rows"]],
                 lambda v: (f"{v / 1000:.1f} s" if v >= 1000 else f"{v:.0f} ms"), 1, log=True)
        axc.set_xscale("log"); axc.set_xlim(20, 60000)
        axc.xaxis.set_major_locator(mt.FixedLocator([10, 100, 1000, 10000])); axc.xaxis.set_minor_locator(mt.NullLocator())
        axc.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v / 1000:g} s" if v >= 1000 else f"{v:g} ms"))
        axc.set_xlabel("Median latency per question (logarithmic axis)", color=F.WP_SUB, **F.FONT)
    else:
        _placeholder(axc, "pending:\nlatency\nwith v3")
    _head(fig, axa, "a", "All systems", "Human-labeled tasks answered by every system", dx=-86, y=1.10, ysub=1.035)
    _head(fig, axb, "b", "RadKev-27B and the LLMs", "Questions with at most 16 options", dx=-30, y=1.10, ysub=1.035)
    _head(fig, axc, "c", "Latency", "One request at a time on two RTX A6000 GPUs; OpenAI Decisions over the network", dx=-123, y=1.10, ysub=1.035)
    fig.legend(handles=[Patch(color=F.C_RAD, label="RadKev"), Patch(color=F.WP_BASE, label="Kev"), Patch(color=F.C_ODEC, label="OpenAI Decisions"),
                        Patch(color=F.C_LLM, label="Qwen3.8-27B"), Patch(color=F.C_LLMQ, label="Qwen3.8-27B, reasoning"), Patch(color=F.C_LLMR, label="MedGemma-27B-text")],
               loc="upper right", ncol=6, frameon=False, prop={**F.FONT, "size": 7.0}, bbox_to_anchor=(0.99, 1.0), handlelength=1.0, handleheight=0.8, columnspacing=1.0)
    _save(fig, "v3_fig_llm")


def fig_spec():
    """Specialization vs scale (a, b) and general decision skill on Kev's transfer suite (c)."""
    from matplotlib.patches import Patch
    E = J.get("eval"); fig = F.plt.figure(figsize=(180 * F.MM, 78 * F.MM))
    axa = fig.add_axes([0.08, 0.14, 0.25, 0.64]); axb = fig.add_axes([0.42, 0.14, 0.25, 0.64]); axc = fig.add_axes([0.78, 0.14, 0.20, 0.64])
    if E and "v3_9" in E["systems"]:
        S, P = E["systems"], E["pairs"]
        for ax, k, f in ((axa, "bench", "task_mean"), (axb, "human", "pooled")):
            F._gbar(fig, ax, [("9B models", 100 * S["stock9"][k][f], 100 * S["v3_9"][k][f], 100 * P["v3_9-stock9"][k][f + "_d"]),
                              ("27B models", 100 * S["stock27"][k][f], 100 * S["v3_27"][k][f], 100 * P["v3_27-stock27"][k][f + "_d"])], (F.WP_BASE, F.WP_ACC), delta=False)
            ax.set_ylabel("Accuracy (%)", color=F.WP_SUB, **F.FONT)
    else:
        _placeholder(axa, "pending: eval_v3"); _placeholder(axb, "pending: eval_v3")
    T = J.get("transfer")
    if T and all(k in T["pairs"] for k in ("v3_9-stock9", "v3_27-stock27")):
        for i, key in enumerate(("v3_9-stock9", "v3_27-stock27")):
            x = T["pairs"][key]["acc_micro"]; d, lo, hi = 100 * x["delta"], 100 * x["ci95"][0], 100 * x["ci95"][1]
            axc.bar(i, d, 0.55, color=F.WP_ACC, zorder=2)
            axc.errorbar(i, d, yerr=[[d - lo], [hi - d]], fmt="none", ecolor="#4D4D4D", elinewidth=0.8, capsize=2.4, zorder=4)
            axc.text(i, (hi + 0.3) if d >= 0 else (lo - 0.3), f"{d:+.1f}".replace("-", "\u2212"), ha="center", va="bottom" if d >= 0 else "top", color=F.WP_INK, **{**F.FONT, "size": 6.8})
        axc.axhline(0, color="#9A9A9A", lw=0.7); axc.set_xticks([0, 1]); axc.set_xticklabels(["9B models", "27B models"], **{**F.FONT, "size": 7.2})
        axc.set_xlim(-0.6, 1.6); axc.set_ylabel("Change in accuracy (pp)", color=F.WP_SUB, **F.FONT)
        lo_, hi_ = axc.get_ylim(); axc.set_ylim(lo_ - 0.15 * (hi_ - lo_), hi_ + 0.1 * (hi_ - lo_))
        for sp in ("top", "right", "left", "bottom"): axc.spines[sp].set_visible(False)
        axc.grid(axis="y", color=F.WP_GRID, lw=0.6, zorder=0); axc.tick_params(length=0, labelsize=7.2, colors=F.WP_SUB)
    else:
        _placeholder(axc, "pending:\ntransfer suite")
    _head(fig, axa, "a", "Benchmark task mean", "Kev (grey) and RadKev (blue) at each size", dx=-30, y=1.12, ysub=1.04)
    _head(fig, axb, "b", "Human-labeled questions, pooled", "Kev (grey) and RadKev (blue) at each size", dx=-30, y=1.12, ysub=1.04)
    _head(fig, axc, "c", "Outside radiology", "Kev's transfer suite, RadKev − Kev", dx=-30, y=1.12, ysub=1.04)
    fig.legend(handles=[Patch(color=F.WP_BASE, label="Kev (released)"), Patch(color=F.WP_ACC, label="RadKev (specialized)")],
               loc="upper right", ncol=2, frameon=False, prop={**F.FONT, "size": 7.5}, bbox_to_anchor=(0.99, 1.0), handlelength=1.0, handleheight=0.8)
    _save(fig, "v3_fig_spec")


def fig_calib():
    import numpy as np
    from matplotlib.patches import Patch
    E = J.get("eval"); fig = F.plt.figure(figsize=(180 * F.MM, 70 * F.MM))
    axs = [fig.add_axes([0.16 + 0.285 * i, 0.17, 0.235, 0.58]) for i in range(3)]
    sys_ = ["v3_27", "stock27", "v3_9", "stock9", "qwen38", "medgemma_fix", "openai_dec"]   # the API: as scored only (probabilities rounded to 0.01)
    specs = [("cov5", 100, "{:.1f}", "a", "Coverage at 5% error", "Questions answered (%)"),
             ("ece", 1, "{:.3f}", "b", "Calibration error", "ECE, ten bins"),
             ("conf_err", 100, "{:.1f}", "c", "Confident errors", "Incorrect with confidence ≥ 0.9 (%)")]
    if E:
        C = E["calibration"]; ss = [s for s in sys_ if s in C]
        for ax, (k, mul, fmt, l, t, xl) in zip(axs, specs):
            a = [mul * C[s]["human"]["as_scored"][k] for s in ss]
            b = [mul * C[s]["human"]["recalibrated"][k] if s != "openai_dec" else None for s in ss]
            bb = [v for v in b if v is not None]
            y = np.arange(len(ss)); h = 0.38; lo = 0 if k != "cov5" else max(0, min(a + bb) - 5); hi = max(a + bb) * 1.12
            ax.barh(y - h / 2, np.array(a) - lo, h, left=lo, color="#5A6577", zorder=2)
            for i in range(len(ss)):
                ax.text(a[i] + (hi - lo) * 0.012, i - h / 2, fmt.format(a[i]), va="center", color=F.WP_INK, **{**F.FONT, "size": 6.2})
                if b[i] is None: continue
                ax.barh(y[i] + h / 2, b[i] - lo, h, left=lo, color="#B7C0CC", zorder=2)
                ax.text(b[i] + (hi - lo) * 0.012, i + h / 2, fmt.format(b[i]), va="center", color=F.WP_SUB, **{**F.FONT, "size": 6.2})
            ax.set_yticks(y); ax.set_yticklabels([NAME[s] for s in ss] if ax is axs[0] else [], **{**F.FONT, "size": 7}); ax.set_ylim(len(ss) - 0.45, -0.55)
            ax.set_xlim(lo, hi); ax.set_xlabel(xl, color=F.WP_SUB, **F.FONT)
            for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color("#BDBDBD"); ax.spines["bottom"].set_linewidth(0.6)   # light axis line, as in the other figures
            ax.grid(axis="x", color=F.WP_GRID, lw=0.6, zorder=0); ax.tick_params(length=0, labelsize=7, colors=F.WP_SUB)
            _head(fig, ax, l, t, None, dx=-11, y=1.06)
    else:
        for ax in axs: _placeholder(ax, "pending: eval_v3")
    fig.legend(handles=[Patch(color="#5A6577", label="as scored"), Patch(color="#B7C0CC", label="recalibrated (two-fold cross-fitted)")],
               loc="upper right", ncol=2, frameon=False, prop={**F.FONT, "size": 7.5}, bbox_to_anchor=(0.99, 1.0), handlelength=1.0, handleheight=0.8)
    _save(fig, "v3_fig_calib")


def fig_robust():
    import numpy as np
    from matplotlib.patches import Patch
    R, E = J.get("preread"), J.get("eval")
    fig = F.plt.figure(figsize=(180 * F.MM, 132 * F.MM))
    axa = fig.add_axes([0.09, 0.565, 0.56, 0.31]); axb = fig.add_axes([0.76, 0.565, 0.21, 0.31])
    axc = fig.add_axes([0.09, 0.085, 0.34, 0.31]); axd = fig.add_axes([0.60, 0.085, 0.32, 0.31])
    if R:
        ms = [m for m in PR_SYS if m in R["models"]]
        items = [(SHORT[m].replace("-", "-\n", 1) if m != "medgemma_fix" else "Med-\nGemma", 100 * R["models"][m]["full"], 100 * R["models"][m]["preread"], 100 * R["models"][m]["change"]) for m in ms]
        F._gbar(fig, axa, items, ("#5A6577", "#B7C0CC"), delta=False); axa.set_ylabel("Accuracy (%)", color=F.WP_SUB, **F.FONT)
    else:
        _placeholder(axa, "pending: pre-read job")
    W = (E or {}).get("wording", {}).get("v3_27-stock27")
    if W and all(k in W for k in ("heldout", "seen")):
        for i, k in enumerate(("heldout", "seen")):
            d, lo, hi = 100 * W[k]["d"], 100 * W[k]["ci"][0], 100 * W[k]["ci"][1]
            axb.bar(i, d, 0.55, color=F.WP_ACC, zorder=2)
            axb.errorbar(i, d, yerr=[[d - lo], [hi - d]], fmt="none", ecolor="#4D4D4D", elinewidth=0.8, capsize=2.4, zorder=4)
            axb.text(i, hi + 0.3, f"{d:+.1f}".replace("-", "−"), ha="center", va="bottom", color=F.WP_INK, **{**F.FONT, "size": 6.8})
        axb.axhline(0, color="#BDBDBD", lw=0.6); axb.set_xticks([0, 1]); axb.set_xticklabels(["Withheld\nwording", "Seen\nwording"], **{**F.FONT, "size": 7.2})
        axb.set_xlim(-0.6, 1.6); axb.set_ylabel("Gain (pp)", color=F.WP_SUB, **F.FONT); axb.set_ylim(0, 1.2 * max(100 * W[k]["ci"][1] for k in ("heldout", "seen")) + 0.3)
        for sp in ("top", "right", "left", "bottom"): axb.spines[sp].set_visible(False)
        axb.grid(axis="y", color=F.WP_GRID, lw=0.6, zorder=0); axb.tick_params(length=0, labelsize=7, colors=F.WP_SUB)
    else:
        _placeholder(axb, "pending: wording")
    B = J.get("blind")
    if B:
        pilot = {r["key"]: r["value"] for r in csv.DictReader(open(RW / "numbers.csv"))}
        for ax, t, ch in ((axc, "eurorad_dx", float(pilot.get("bl_chance_dx", "24"))), (axd, "rsna_radioqa", 25.0)):
            M, pr = B["tasks"][t]["models"], B["tasks"][t]["pairs"]["v3_27-stock27"]
            F._gbar(fig, ax, [("Full case", 100 * M["stock27"]["full"], 100 * M["v3_27"]["full"], 100 * pr["full"]),
                              ("Options only", 100 * M["stock27"]["blind"], 100 * M["v3_27"]["blind"], 100 * pr["blind"])], (F.WP_BASE, F.WP_ACC), extra=(ch,), delta=False)
            ax.axhline(ch, color=F.WP_SUB, lw=0.7, ls=(0, (3, 2)), zorder=4)
            ax.text(1.015, ch, f"chance\n{ch:.0f}%", transform=ax.get_yaxis_transform(), ha="left", va="center", color=F.WP_SUB, linespacing=1.0, **{**F.FONT, "size": 6.8})
            ax.set_ylabel("Accuracy (%)", color=F.WP_SUB, **F.FONT); _ticks(ax)
    else:
        _placeholder(axc, "pending: options-only\ncontrol with v3"); _placeholder(axd, "pending: options-only\ncontrol with v3")
    _head(fig, axa, "a", "Subspecialty classification before the read", "Full state and with the imaging findings removed", dx=-30, y=1.12, ysub=1.04)
    _head(fig, axb, "b", "Instruction wording", "RadKev-27B − Kev-27B", dx=-30, y=1.12, ysub=1.04)
    _head(fig, axc, "c", "Options only: Eurorad diagnosis", "Kev-27B (grey), RadKev-27B (blue)", dx=-30, y=1.12, ysub=1.04)
    _head(fig, axd, "d", "Options only: RSNA-RadioQA", "Kev-27B (grey), RadKev-27B (blue)", dx=-30, y=1.12, ysub=1.04)
    fig.legend(handles=[Patch(color="#5A6577", label="full state"), Patch(color="#B7C0CC", label="imaging findings removed (pre-read)")],
               loc="upper left", ncol=2, frameon=False, prop={**F.FONT, "size": 7.5}, bbox_to_anchor=(0.01, 1.0), handlelength=1.0, handleheight=0.8)
    _save(fig, "v3_fig_robust")


def fig_answer_space():
    """(a) Eurorad and (b) RSNA-RadioQA accuracy with the most similar alternatives, by number of options (log scale); (c) median latency."""
    S = AS_FIG.get("S")
    fig = F.plt.figure(figsize=(180 * F.MM, 76 * F.MM))
    axs = [fig.add_axes([0.07 + 0.33 * i, 0.16, 0.25, 0.54]) for i in range(3)]
    col = {"v3_27": F.C_RAD, "stock27": "#5A6577", "v3_9": "#8FB0EE", "stock9": "#B7C0CC", "qwen38_num": F.C_LLM, "medgemma_fix_num": F.C_LLMR,
           "qwen38_letter": F.C_LLM, "medgemma_fix_letter": F.C_LLMR}
    if not S:
        for ax in axs: _placeholder(ax, "pending: answer space")
    else:
        for ax, s_ in zip(axs[:2], ("eurorad_dx", "rsna_radioqa")):
            for m, lab, _ in AS_SYS:
                pts = [(K, 100 * S["acc"][m][s_][f"sim_{K}"][0]) for K in AS_K if f"sim_{K}" in S["acc"].get(m, {}).get(s_, {})]
                if pts: ax.plot(*zip(*pts), marker="o", ms=3, lw=1.4 if m.startswith("v3") else 1.0, color=col[m], label=lab)
            ax.set_xscale("log", base=2); ax.set_xticks(AS_K); ax.set_xticklabels([str(k) for k in AS_K]); ax.set_ylim(0, 100)
            ax.set_xlabel("Options offered", color=F.WP_SUB, **F.FONT); ax.set_ylabel("Accuracy (%)", color=F.WP_SUB, **F.FONT)
        for m, rows in S["lat"].items():
            name = {"v3_27": "RadKev-27B", "stock27": "Kev-27B", "v3_9": "RadKev-9B", "stock9": "Kev-9B", "qwen38_num": "Qwen3.8-27B, numbered",
                    "medgemma_fix_num": "MedGemma, numbered", "qwen38_letter": "Qwen3.8-27B, letter", "medgemma_fix_letter": "MedGemma, letter"}.get(m)
            if not name: continue
            ks = sorted(rows); kv = m.startswith("stock")   # RadKev and Kev of the same size coincide: Kev drawn wider underneath RadKev
            axs[2].plot(ks, [rows[k] for k in ks], marker="o", ms=4.5 if kv else 3, lw=2.6 if kv else 1.0, color=col.get(m, "#999"),
                        ls="--" if "letter" in m else "-", label=name, zorder=2 if kv else 4)
        axs[2].set_xscale("log", base=2); axs[2].set_yscale("log"); axs[2].set_xticks([2, 16, 64, 255]); axs[2].set_xticklabels(["2", "16", "64", "255"])
        axs[2].set_xlabel("Options offered", color=F.WP_SUB, **F.FONT); axs[2].set_ylabel("Median latency per request", color=F.WP_SUB, **F.FONT)
        import matplotlib.ticker as _mt
        axs[2].yaxis.set_major_locator(_mt.FixedLocator([100, 300, 1000, 3000])); axs[2].yaxis.set_minor_locator(_mt.NullLocator())
        axs[2].yaxis.set_major_formatter(_mt.FuncFormatter(lambda v, _: f"{v / 1000:g} s" if v >= 1000 else f"{v:g} ms"))
        for ax in axs:
            for sp in ("top", "right", "left"): ax.spines[sp].set_visible(False)
            ax.spines["bottom"].set_color("#BDBDBD"); ax.spines["bottom"].set_linewidth(0.6)   # light axis line, as in the other figures
            ax.grid(color=F.WP_GRID, lw=0.6); ax.tick_params(length=0, labelsize=7, colors=F.WP_SUB)
        from matplotlib.lines import Line2D
        fig.legend(handles=[Line2D([], [], color=col[m], marker="o", ms=3, lw=1.2, label=lab) for m, lab, _ in AS_SYS] +
                   [Line2D([], [], color="#777", ls="--", lw=1.0, label="LLM, letter scoring")],
                   loc="upper center", ncol=7, frameon=False, prop={**F.FONT, "size": 6.8}, bbox_to_anchor=(0.5, 1.0), handlelength=1.6, columnspacing=1.0)
    _head(fig, axs[0], "a", "Eurorad diagnosis", "Most similar alternatives", dx=-25)
    _head(fig, axs[1], "b", "RSNA-RadioQA", "Most similar alternatives", dx=-25)
    _head(fig, axs[2], "c", "Latency", "One question per request; logarithmic axes", dx=-25)
    _save(fig, "v3_fig_answer_space")


# ------------------------------------------------------------------------------------------------ tables
def tab(name, colspec, header, rows, midrules=()):
    body = []
    for i, r in enumerate(rows):
        if i in midrules: body.append(r"\midrule")
        body.append(" & ".join(r) + r" \\")
    (RW / "generated" / f"{name}.tex").write_text("\n".join([rf"\begin{{tabular}}{{{colspec}}}", r"\toprule", header + r" \\", r"\midrule", *body, r"\bottomrule", r"\end{tabular}"]) + "\n")


def tab_pending(name, what, ncol=2):
    (RW / "generated" / f"{name}.tex").write_text("\\begin{tabular}{l}\n\\toprule\n\\missing{pending: " + what.replace("_", r"\_") + "} \\\\\n\\bottomrule\n\\end{tabular}\n")


def build_tables():
    E = J.get("eval")
    if not E:
        for n in ("v3_primary_tasks", "v3_pertask", "v3_pairs", "v3_calib", "v3_ctrate"): tab_pending(n, "eval_v3")
    else:
        S, P = E["systems"], E["pairs"]; pr = P["v3_27-stock27"]
        fmt_p = lambda p: "<0.001" if p < 0.001 else f"{p:.3f}"
        rows = []
        for t in BENCH:
            v = pr["tasks"][t]
            rows.append([TASK[t], n_(E["n"][t]), pct(S["stock27"]["tasks"][t]["acc"]), pct(S["v3_27"]["tasks"][t]["acc"]),
                         f"{pct(v['d'])} ({pci(v['ci'])})", fmt_p(v["p_holm"])])
        for lab, k, f in (("Task mean, all tasks (primary)", "bench", "task_mean"), ("Task mean, human-labeled", "human", "task_mean"),
                          ("All questions pooled", "bench", "pooled"), ("Human-labeled questions pooled", "human", "pooled")):
            rows.append([lab, n_(S["v3_27"][k]["n"]), pct(S["stock27"][k][f]), pct(S["v3_27"][k][f]),
                         f"{pct(pr[k][f + '_d'])} ({pci(pr[k][f + '_ci'])})", "--"])
        tab("v3_primary_tasks", "@{}lrrrlr@{}", r"Task & n & Kev-27B & RadKev-27B & $\Delta$ (95\% CI) & Holm $p$", rows, midrules=(len(HUMAN), len(BENCH)))
        cols = [s for s in SYSTEMS if s in S]
        rows = [[TASK[t]] + [pct(S[s]["tasks"][t]["acc"]) if t in S[s]["tasks"] else "--" for s in cols] for t in BENCH]
        rows += [["Task mean, all tasks"] + [pct(S[s]["bench"]["task_mean"]) if S[s]["bench"]["tasks"] == len(BENCH) else "--" for s in cols],
                 ["Task mean, human-labeled"] + [pct(S[s]["human"]["task_mean"]) if S[s]["human"]["tasks"] == len(HUMAN) else "--" for s in cols]]
        rows += [[TASK[t]] + [pct(S[s]["tasks"][t]["acc"]) if t in S[s]["tasks"] else "--" for s in cols] for t in CTRATE]
        tab("v3_pertask", "@{}l" + "r" * len(cols) + "@{}", "Task & " + " & ".join(SHORT[s] for s in cols), rows, midrules=(len(HUMAN), len(BENCH), len(BENCH) + 2))
        rows = []
        for al, (a, b) in PAIRS.items():
            key = f"{a}-{b}"
            if key not in P: continue
            o = P[key]
            rows.append([f"{NAME[a]} $-$ {NAME[b]}"] + [f"{pct(o[k][f + '_d'])} ({pci(o[k][f + '_ci'])})" for k, f in (("bench", "task_mean"), ("human", "task_mean"), ("bench", "pooled"))]
                        + [str(o["bench"]["tasks"])])
        tab("v3_pairs", "@{}llllr@{}", r"Comparison & Task mean, all & Task mean, human & Pooled, all & Tasks", rows)
        C = E["calibration"]; rows = []
        for s in [s for s in SYSTEMS if s in C]:
            a, r = C[s]["human"]["as_scored"], C[s]["human"]["recalibrated"]
            rc_ = (lambda f: "--") if s == "openai_dec" else (lambda f: f())   # API probabilities are rounded to 2 decimals: no recalibration reported
            rows.append([NAME[s], num(a["ece"], 3), rc_(lambda: num(r["ece"], 3)), num(a["brier"], 3), pct(a["conf_err"]), rc_(lambda: pct(r["conf_err"])), pct(a["cov5"]), rc_(lambda: pct(r["cov5"]))])
        tab("v3_calib", "@{}lrrrrrrr@{}", r"System & ECE & ECE (recal.) & Brier & Conf.\ err. & Conf.\ err. (recal.) & Cov.\ 5\% & Cov.\ 5\% (recal.)", rows)
    R = J.get("preread")
    if R:
        rows = [[NAME[m], pct(R["models"][m]["full"]), pct(R["models"][m]["preread"]), f"{pct(R['models'][m]['change'])} ({pci(R['models'][m]['change_ci'])})"] for m in PR_SYS if m in R["models"]]
        tab("v3_preread", "@{}lrrl@{}", r"System & Full state & Findings removed & Change (95\% CI)", rows)
    else:
        tab_pending("v3_preread", "pre-read job")
    T = J.get("transfer")
    if T:
        rows = []
        for key, a_, b_ in (("v3_9-stock9", "v3_9", "stock9"), ("v3_27-stock27", "v3_27", "stock27")):
            x = T["pairs"][key]["acc_micro"]; M = T["models"]
            rows.append([f"{NAME[a_]} vs {NAME[b_]}", pct(M[a_]["acc"]) if a_ in M else "--", pct(M[b_]["acc"]), f"{pct(x['delta'])} ({pci(x['ci95'])})"])
        tab("v3_transfer", "@{}lrrl@{}", r"Comparison & RadKev & Kev & $\Delta$ (95\% CI)", rows)
    else:
        tab_pending("v3_transfer", "transfer suite with v3")
    B = J.get("blind")
    if B:
        rows, mids = [], []
        for t, lab in (("eurorad_dx", "Eurorad diagnosis"), ("rsna_radioqa", "RSNA-RadioQA")):
            if rows: mids.append(len(rows))
            M = B["tasks"][t]["models"]
            for m in ("v3_27", "stock27", "v3_9", "stock9", "qwen38", "medgemma_fix"):
                if m in M:
                    v = M[m]
                    rows.append([lab if m == "v3_27" else "", NAME[m], f"{pct(v['full'])} ({pci(v['full_ci'])})", f"{pct(v['blind'])} ({pci(v['blind_ci'])})"])
        tab("v3_blind", "@{}lllr@{}".replace("r@{}", "l@{}"), r"Task & System & Full case & Options only", rows, midrules=tuple(mids))
    else:
        tab_pending("v3_blind", "options-only control with v3")
    L = J.get("latency")
    if L:
        M = L["models"]; rows = []
        for m, lab in (("v3_27", "RadKev-27B"), ("stock27", "Kev-27B"), ("v3_9", "RadKev-9B"), ("stock9", "Kev-9B")):
            if m in M: rows.append([lab, "one pass per record", f"{M[m]['per_question_ms']['median']:.0f}", f"{M[m]['per_question_ms']['p95']:.0f}", n_(M[m]['per_question_ms']['n'])])
        if J.get("odec_lat"):
            for k_, mlab in (("concurrent", "end to end, 4 in flight"), ("sequential", "end to end, one at a time")):
                x = J["odec_lat"][k_]["per_question_ms"]; rows.append(["OpenAI Decisions", mlab, f"{x['median']:.0f}", f"{x['p95']:.0f}", n_(x["n"])])
        for m, lab in (("qwen38", "Qwen3.8-27B"), ("medgemma_fix", "MedGemma-27B-text")):
            for mode, mlab in (("letter", "option-letter logits"), ("direct", "generated letter"), ("reasoning", "reasoning")):
                if m in M and mode in M[m]:
                    x = M[m][mode]["per_question_ms"]; rows.append([lab, mlab, f"{x['median']:.0f}", f"{x.get('p95', float('nan')):.0f}", n_(x["n"])])
        tab("v3_latency", "@{}llrrr@{}", r"System & Mode & Median (ms) & 95th percentile (ms) & $n$", rows)
    else:
        tab_pending("v3_latency", "latency with v3")
    R = (J.get("reasoning") or {}).get("paired")
    if R:
        rows = []
        for m, lab in (("v3_27", "RadKev-27B"), ("stock27", "Kev-27B"), ("v3_9", "RadKev-9B"), ("qwen38", "Qwen3.8-27B"), ("qwen38_think", "Qwen3.8-27B, reasoning"),
                       ("medgemma_fix", "MedGemma-27B-text"), ("medgemma_think", "MedGemma-27B-text, reasoning")):
            if m in R["systems"]:
                b_ = R["systems"][m]["bench"]
                rows.append([lab, f"{pct(b_['pooled'])} ({pci(b_['pooled_ci'])})", f"{pct(b_['task_mean_k'])} ({pci(b_['task_mean_k_ci'])})"])
        tab("v3_reasoning", "@{}lll@{}", r"System & Pooled over questions & Averaged over tasks", rows)
    else:
        tab_pending("v3_reasoning", "reasoning sample with v3")
    G = (J.get("radgraph") or {}).get("radgraph_xl")
    if G:
        rows = [[NAME[m], f"{pct(G['models'][m]['all']['acc'])} ({pci(G['models'][m]['all']['acc_ci'])})"] for m in SYSTEMS if m in G["models"]]
        rows += [[f"{NAME[a_]} $-$ {NAME[b_]}", f"{pct(G['pairs'][f'{a_}-{b_}']['all']['d'])} ({pci(G['pairs'][f'{a_}-{b_}']['all']['ci'])})"]
                 for a_, b_ in (("v3_27", "stock27"), ("v3_9", "stock9"), ("v3_27", "qwen38"), ("v3_27", "medgemma_fix")) if f"{a_}-{b_}" in G["pairs"]]
        tab("v3_radgraph", "@{}ll@{}", r"System or comparison & Accuracy or difference (95\% CI)", rows, midrules=(sum(1 for m in SYSTEMS if m in G["models"]),))
    else:
        tab_pending("v3_radgraph", "RadGraph-XL with v3")


# ------------------------------------------------------------------------------------------------ claims
def claims():
    """Directional statements of the v3 prose (marked % [Cn] in the .tex). Each holds or the sentence must be rewritten."""
    E = J.get("eval")
    if not E: return []
    P, S = E["pairs"], MAIN(E); R = J.get("preread")
    c = lambda k, agg="bench", f="task_mean": P[k][agg][f + "_ci"]
    out = [
        ("C1", "abstract, results 3.1, conclusions", "Specialization increased the primary outcome (task mean, all tasks)", sig(c("v3_27-stock27")) == 1),
        ("C2", "abstract, results 3.1", "...and the human-assigned task mean", sig(c("v3_27-stock27", "human")) == 1),
        ("C3", "results 3.1, discussion, conclusions", "RadCases panel: as scored RadKev-27B defaulted to the catch-all option (below Kev-27B); prior-corrected it does not differ detectably from Kev-27B",
         J.get("rcprior") is not None and J["rcprior"]["pairs"]["v3_27-stock27"]["ci"][1] < 0 and P["v3_27-stock27"]["tasks"]["radcases_panel"]["p_holm"] >= 0.05
         and P["v3_27-stock27"]["tasks"]["radcases_panel"]["ci"][0] < 0 < P["v3_27-stock27"]["tasks"]["radcases_panel"]["ci"][1]),
        ("C4", "abstract, results 3.2, discussion, conclusions", "RadKev-27B more accurate than Qwen3.8-27B (task mean on shared tasks)", sig(c("v3_27-qwen38")) == 1),
        ("C5", "abstract, results 3.2, conclusions", "RadKev-27B more accurate than MedGemma-27B-text", sig(c("v3_27-medgemma_fix")) == 1),
        ("C6", "results 3.2, discussion", "Kev-27B and Qwen3.8-27B not detectably different (benchmark task mean)", sig(c("qwen38-stock27")) == 0),
        ("C16", "results 3.2, discussion, conclusions", "RadKev-27B more accurate than Qwen3.8-27B over the human-assigned tasks", sig(c("v3_27-qwen38", "human")) == 1),
        ("C8", "results 3.3, discussion, conclusions", "27B: specialization vs scale not detectably different on the task means (CIs include 0), larger pooled over human-assigned questions",
         E["did"]["r3_did27_tm_ci"][0] < 0 < E["did"]["r3_did27_tm_ci"][1] and E["did"]["r3_did27_htm_ci"][0] < 0 < E["did"]["r3_did27_htm_ci"][1]
         and E["did"]["r3_did27_hpool_ci"][0] > 0),
        ("C9", "results 3.3, discussion", "Specialization increased the task mean at 9B too", "v3_9-stock9" in P and sig(c("v3_9-stock9")) == 1),
        ("C11", "results 3.4, discussion", "RadKev-27B covers more questions at 5% error than Kev-27B (point estimate)",
         E["calibration"]["v3_27"]["human"]["as_scored"]["cov5"] > E["calibration"]["stock27"]["human"]["as_scored"]["cov5"]),
        ("C12", "results 3.4, discussion, conclusions", "RadKev-27B ECE within 0.005 of Kev-27B as scored, but more confident errors",
         abs(E["calibration"]["v3_27"]["human"]["as_scored"]["ece"] - E["calibration"]["stock27"]["human"]["as_scored"]["ece"]) < 0.005
         and E["calibration"]["v3_27"]["human"]["as_scored"]["conf_err"] > E["calibration"]["stock27"]["human"]["as_scored"]["conf_err"]),
        ("C15", "discussion, conclusions", "Both specialized models cover more questions at 5% error than their starting points and both LLMs",
         min(E["calibration"][m]["human"]["as_scored"]["cov5"] for m in ("v3_27", "v3_9")) > max(E["calibration"][m]["human"]["as_scored"]["cov5"] for m in ("stock27", "stock9", "qwen38", "medgemma_fix"))),
    ]
    if ASC:
        his = [v[2] for v in ASC.values() if v[2] is not None]
        out.append(("C25", "results 3.6, discussion", "RadKev-27B: mean confidence exceeds accuracy by > 10 pp with the 255 most similar options; accuracy at confidence >= 0.9 at least 90% in every answer space; share at >= 0.9 lower for sim_255 than rand_16",
                    ASC["sim_255"][1] - ASC["sim_255"][0] > 0.10 and min(his) >= 0.90 and ASC["sim_255"][3] < ASC["rand_16"][3]))
    if "eval_raw" in J:
        out.append(("C24", "results 3.1", "Primary outcome positive without the RadCases prior correction (CI above 0)",
                    sig(J["eval_raw"]["pairs"]["v3_27-stock27"]["bench"]["task_mean_ci"]) == 1))
    if "v3_27-openai_dec" in P:
        out.append(("C23", "results 3.2", "RadKev-27B more accurate than the OpenAI Decisions API (benchmark task mean)", sig(c("v3_27-openai_dec")) == 1))
    B = J.get("blind")
    if B:
        pr = B["tasks"]["eurorad_dx"]["pairs"]["v3_27-stock27"]; M = B["tasks"]["eurorad_dx"]["models"]
        out.append(("C18", "abstract, results 3.5, discussion, conclusions", "Eurorad gain larger without the case (did CI < 0) and RadKev-27B options-only accuracy above 50%",
                    pr["did_ci"][1] < 0 and M["v3_27"]["blind"] > 0.5))
    RS = (J.get("reasoning") or {}).get("paired")
    if RS:   # task-mean rule agreed 2026-10-06; 2026-10-07 (user: fix as seen fit): examination tasks pooled (1-69 questions each) and source-stratified bootstrap
        out.append(("C19", "abstract, discussion, conclusions", "RadKev-27B more accurate than Qwen3.8-27B with reasoning: task mean (examination tasks pooled) and pooled CIs above 0",
                    sig(RS["pairs"]["v3_27-qwen38_think"]["bench"]["task_mean_k_ci"]) == 1 and sig(RS["pairs"]["v3_27-qwen38_think"]["bench"]["pooled_ci"]) == 1))
    out.append(("C20", "abstract", "RadKev-27B covers more of all benchmark questions at 5% error than Kev-27B (point estimate)",
                E["calibration"]["v3_27"]["bench"]["as_scored"]["cov5"] > E["calibration"]["stock27"]["bench"]["as_scored"]["cov5"]))
    A3 = as_summary()
    if A3 and J.get("latency"):
        lat = A3["lat"]; Lq = J["latency"]["models"]["qwen38"]
        out.append(("C21", "results 3.6, discussion", "Letter-scored LLM within 25% of RadKev-27B per question, and numbered LLM faster than RadKev-27B at 255 options",
                    Lq["letter"]["per_question_ms"]["median"] < 1.25 * J["latency"]["models"]["v3_27"]["per_question_ms"]["median"]
                    and lat["qwen38_num"][255] < lat["v3_27"][255]))
    G = (J.get("radgraph") or {}).get("radgraph_xl")
    if G:
        out.append(("C22", "conclusions, results 3.7", "RadGraph-XL: 27B specialization CI below 0, 9B specialization CI above 0",
                    G["pairs"]["v3_27-stock27"]["all"]["ci"][1] < 0 and G["pairs"]["v3_9-stock9"]["all"]["ci"][0] > 0))
    W = E.get("wording", {})
    if all(k in W.get(p_, {}) for p_ in ("v3_27-stock27", "v3_9-stock9") for k in ("heldout", "seen", "did")):
        out.append(("C17", "discussion", "27B: held-out gain > 0 and smaller than seen (did CI < 0); 9B: did CI includes 0",
                    W["v3_27-stock27"]["heldout"]["ci"][0] > 0 and W["v3_27-stock27"]["did"]["ci"][1] < 0 and sig(W["v3_9-stock9"]["did"]["ci"]) == 0))
    if R:
        out.append(("C13", "results 3.5, discussion", "Every system lost accuracy (CI below 0) when the imaging findings were removed",
                    all(v["change_ci"][1] < 0 for v in R["models"].values())))
        out.append(("C14", "results 3.5", "RadKev-27B and Kev-27B did not differ detectably before the read",
                    "v3_27-stock27" in R["pairs"] and sig(R["pairs"]["v3_27-stock27"]["preread"]["ci"]) == 0))
    return out


# ------------------------------------------------------------------------------------------------ manuscript assembly
def write_numbers():
    lines = ["% Generated by v3/build_v3.py. Do not edit; the ledger is v3/numbers_v3.csv."]
    lines += [rf"\expandafter\def\csname V@{r['key']}\endcsname{{{r['value']}}}" for r in LEDGER]
    (HERE / "numbers_v3.tex").write_text("\n".join(lines) + "\n")
    with open(HERE / "numbers_v3.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["key", "value", "source", "field", "note"]); w.writeheader(); w.writerows(LEDGER)


V3_TEX = ["abstract.tex", "results.tex", "discussion.tex", "conclusions.tex", "supp_tables.tex"]


def assemble():
    tex = (RW / "main.tex").read_text()
    tex = re.sub(r"(\\begin\{abstract\}).*?(\\end\{abstract\})", lambda m: m.group(1) + "\n\\input{v3/abstract.tex}\n" + m.group(2), tex, count=1, flags=re.S)
    i, j = tex.index("\\section{Results}"), tex.index("\\section*{Declarations}")
    tex = tex[:i] + "\\input{v3/results.tex}\n\n\\input{v3/discussion.tex}\n\n\\input{v3/conclusions.tex}\n\n" + tex[j:]
    i, j = tex.index("\\subsection*{Supplementary Tables}"), tex.index("\\end{document}")
    tex = tex[:i] + "\\input{v3/supp_tables.tex}\n\n" + tex[j:]
    tex = tex.replace("\\input{numbers.tex}", "\\input{numbers.tex}\\input{v3/numbers_v3.tex}", 1)
    tex = re.sub(r"(\\documentclass[^\n]*\n)", lambda m: m.group(1) + "\\makeatletter\\def\\input@path{{../}}\\makeatother\n", tex, count=1)
    tex = tex.replace("\\begin{document}", "\\graphicspath{{../}}\n\\begin{document}", 1)
    shutil.copy(RW / "refs.bib", HERE / "refs.bib")   # bibtex does not follow ../ paths; a copy, regenerated on every build
    head = "% GENERATED by v3/build_v3.py from main.tex: the v3 abstract, Results, Discussion, Conclusions and supplementary tables are swapped in.\n"
    (HERE / "main_v3.tex").write_text(head + tex)
    return tex


def check_macros(tex):
    known = {r["key"] for r in csv.DictReader(open(RW / "numbers.csv"))} | {r["key"] for r in LEDGER}
    used = set()
    strip = lambda t: re.sub(r"(?<!\\)%.*", "", t)
    for f in V3_TEX: used |= set(re.findall(r"\\V\{([^}]+)\}", strip((HERE / f).read_text())))
    used |= set(re.findall(r"\\V\{([^}]+)\}", strip(tex)))
    return sorted(used - known)


def abstract_chars():
    vals = {r["key"]: r["value"] for r in csv.DictReader(open(RW / "numbers.csv"))} | {r["key"]: r["value"] for r in LEDGER}
    body = re.sub(r"(?<!\\)%.*", "", (HERE / "abstract.tex").read_text())
    body = re.sub(r"\\V\{([^}]*)\}", lambda k: re.sub(r"\\missing\{([^}]*)\}", r"[\1]", vals.get(k.group(1), "??")) if k.group(1) != "r3_banner" else "", body)
    body = re.sub(r"\\missing\{[^}]*\}", "", body)
    body = body.replace("\\%", "%").replace("--", "\u2013").replace("~", " ")
    body = re.sub(r"\\[a-zA-Z]+\*?", " ", body)
    return " ".join(re.sub(r"[{}\\$]", "", body).split())


def compile_and_export(tex):
    exe = shutil.which("tectonic")
    if not exe: print("tectonic not found: PDF not compiled"); return
    p = subprocess.run([exe, "--keep-logs", "main_v3.tex"], cwd=HERE, capture_output=True, text=True)
    bad = [l for l in (p.stdout + p.stderr).splitlines() if re.search(r"error|undefined|\?\?", l, re.I)]
    print("compile:", "ok" if p.returncode == 0 else "FAILED", "\n  " + "\n  ".join(bad[:20]) if bad else "")
    if p.returncode: return
    # flattened, editable copy (every number written in), as export_latex does for main.tex
    out = HERE / "export"; shutil.rmtree(out, ignore_errors=True); (out / "figures").mkdir(parents=True)
    vals = {r["key"]: r["value"] for r in csv.DictReader(open(RW / "numbers.csv"))} | {r["key"]: r["value"] for r in LEDGER}
    t = (RW / "main.tex").read_text()
    t = re.sub(r"(\\begin\{abstract\}).*?(\\end\{abstract\})", lambda m: m.group(1) + "\n\\input{v3/abstract.tex}\n" + m.group(2), t, count=1, flags=re.S)
    i, j = t.index("\\section{Results}"), t.index("\\section*{Declarations}")
    t = t[:i] + "\\input{v3/results.tex}\n\n\\input{v3/discussion.tex}\n\n\\input{v3/conclusions.tex}\n\n" + t[j:]
    i, j = t.index("\\subsection*{Supplementary Tables}"), t.index("\\end{document}")
    t = t[:i] + "\\input{v3/supp_tables.tex}\n\n" + t[j:]
    t = re.sub(r"^\\input\{numbers\.tex\}.*\n", "", t, flags=re.M)
    for _ in range(4):
        t = re.sub(r"\\input\{((?:generated|v3)/[^}]+\.tex)\}", lambda m: (RW / m.group(1)).read_text().rstrip("\n"), t)
    t = re.sub(r"\\V\{([^}]+)\}", lambda m: vals.get(m.group(1), m.group(0)), t)   # function replacement: inserted verbatim
    (out / "RadKev_v3_manuscript.tex").write_text("% RadKev v3 manuscript: self-contained copy exported by v3/build_v3.py; numbers written in.\n" + t)
    shutil.copy(RW / "refs.bib", out / "refs.bib")
    for f in sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", t))): shutil.copy(RW / f, out / f)
    q = subprocess.run([exe, "RadKev_v3_manuscript.tex"], cwd=out, capture_output=True, text=True)
    print("export compile:", "ok" if q.returncode == 0 else "FAILED " + q.stderr[-800:])
    shutil.copy(HERE / "main_v3.pdf", HERE / ("RadKev_v3_PREDICTED.pdf" if MOCK else "RadKev_v3.pdf"))
    zp = HERE / ("RadKev_v3_PREDICTED_latex.zip" if MOCK else "RadKev_v3_latex.zip"); zp.unlink(missing_ok=True); shutil.make_archive(str(zp.with_suffix("")), "zip", out)


def main():
    have = build_eval(); build_odec(); build_rcdiag(); build_rcprior(); build_preread(); build_training(); build_transfer(); build_latency(); build_reasoning(); build_blind(); build_radgraph(); build_answer_space()
    for f in (fig_primary, fig_llm, fig_spec, fig_calib, fig_robust, fig_answer_space): f()
    build_tables()
    WIRED = "\\input{v3/results.tex}" in (RW / "main.tex").read_text()   # since 2026-10-06 main.tex inputs these sections; build.py compiles
    tex = "" if WIRED else assemble()
    for k in check_macros(tex):          # results macros whose input has not arrived: red marker, named after the missing input
        if k.startswith("r3_"): pending(k, "eval_v3" if not J.get("eval") else "input not yet available")
    write_numbers(); tex = "" if WIRED else assemble()
    miss = check_macros(tex)
    if miss: raise SystemExit(f"undefined \\V keys in the v3 text: {miss}")
    C = claims()
    lines = ["# Claims check (generated by build_v3.py" + (", PREDICTED inputs" if MOCK else "") + ")", "",
             "Each directional statement in the v3 prose is marked `% [Cn]` in the .tex. FAILS = rewrite that sentence before release.", "",
             "| id | holds | where | statement |", "|---|---|---|---|"]
    lines += [f"| {i} | {'yes' if ok else '**FAILS**'} | {w} | {s} |" for i, w, s, ok in C]
    lines += ["", "## Pending inputs", ""] + [f"- **{w}**: {len(k)} macros ({', '.join(k[:6])}{', ...' if len(k) > 6 else ''})" for w, k in PENDING.items()]
    (HERE / ("CLAIMS_PREDICTED.md" if MOCK else "CLAIMS.md")).write_text("\n".join(lines) + "\n")
    a = abstract_chars()
    print(f"inputs: " + ", ".join(f"{k}={'yes' if SRC[k] else 'no'}" for k in INPUTS))
    print(f"{len(LEDGER)} r3 macros; {sum(len(v) for v in PENDING.values())} pending in {len(PENDING)} groups; claims failing: {[i for i, _, _, ok in C if not ok]}")
    print(f"abstract: {len(a)} characters (arXiv limit 1,920)")
    if STRICT:
        assert not PENDING, f"pending inputs: {list(PENDING)}"
        assert all(ok for *_, ok in C), "claims fail: see CLAIMS.md"
        assert len(a) <= 1920, "abstract too long"
    if WIRED: print("main.tex already inputs v3/: numbers, figures and tables regenerated; compile with python3 paper/rewrite/build.py")
    elif "--no-pdf" not in sys.argv: compile_and_export(tex)


def export_public_v3(dest):
    """Copy the v3 sources and the job outputs they read (scrubbed like build.py's inputs) into the public paper/ directory."""
    import json as _j, shutil as _sh
    sys.path.insert(0, str(RW)); import build as B
    dest = Path(dest).resolve(); (dest / "v3").mkdir(parents=True, exist_ok=True)
    for f in ("build_v3.py", "abstract.tex", "results.tex", "discussion.tex", "conclusions.tex", "supp_tables.tex", "numbers_v3.tex", "numbers_v3.csv", "CLAIMS.md"):
        _sh.copy(HERE / f, dest / "v3" / f)
    n = 0; PRIV = _j.loads((HERE / "private_names.json").read_text())
    for k, p in SRC.items():
        if not p: continue
        out = dest / "inputs" / "artifacts" / p.relative_to(A_); out.parent.mkdir(parents=True, exist_ok=True); raw = p.read_bytes()
        if p.suffix == ".json":
            x = _j.loads(raw); y = B._scrub(x); raw = raw if y == x else (_j.dumps(y, ensure_ascii=False) + "\n").encode()
        s_ = raw.decode()
        for k_, v_ in PRIV["replace"].items(): s_ = s_.replace(k_, v_)   # internal host and account names (v3/private_names.json, never exported)
        raw = s_.encode(); hit = B.ABS_PATH.search(s_) or re.search("|".join(map(re.escape, PRIV["forbidden"])), s_)
        assert not hit, f"{p.name}: absolute path or internal name {hit.group(0)[:60]}"
        out.write_bytes(raw); n += 1
    print(f"public v3: {n} inputs and the v3 sources in {dest}")


if __name__ == "__main__":
    main()
    if "--export-public" in sys.argv: export_public_v3(sys.argv[sys.argv.index("--export-public") + 1])
