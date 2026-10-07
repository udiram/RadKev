"""Build the rewritten RadKev manuscript (paper/rewrite/).

Every number in main.tex is a \\V{key} macro. This script reads each value from a
result file and writes:
  numbers.tex  the macros main.tex inputs
  numbers.csv  the ledger: key, printed value, source file, field, what it is
then compiles main.tex with tectonic and prints the abstract length. The asserts
stop the build if a claim made in the text stops holding.

Usage: python3 build.py [--no-pdf] [--export-public DEST]

The same script builds the private repository's copy and the public bundle (RadKev/paper/). Every input is resolved
by inp(): from inputs/ when that directory exists (public mode), else from its location in the private repository.
--export-public DEST writes the public bundle (this script, figures.py, main.tex, refs.bib, README.md and scrubbed
copies of every input the build read) into DEST.
"""
import ast
import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parent
PUBLIC = (HERE / "inputs").is_dir()   # public bundle: every input under inputs/, scrubbed by export_public
# Inputs whose private location differs from paper/<name> (glob, relative to the private repository root).
PRIVATE = {"environment/setup.json": "deploy/*/results/fb1f2edb-b9e3-497d-b369-1ad77ea45b63/artifacts/setup.json",
           "results/final_test/comparisons.json": "paper/results/final_test/final-b71c9617/comparisons.json",
           "results/final_test/test_v2_27.json": "paper/results/final_test/final-b71c9617/test_v2_27.json",
           "results/transfer_paired/transfer_paired.json": "paper/results/transfer_paired/c52e3709/transfer_paired.json"}
USED = set()   # every input name resolved during the build (export_public publishes exactly these)


def inp(rel):
    """Path of the input named rel: inputs/<rel> in the public bundle, else its location in the private repository."""
    USED.add(rel)
    if PUBLIC:
        return HERE / "inputs" / rel
    return next(ROOT.glob(PRIVATE[rel])) if rel in PRIVATE else PAPER / rel


def job_constants():
    """Constants read from the job scripts that produced the data (published as inputs/constants.json; keys are ledger fields)."""
    if PUBLIC:
        return json.loads(inp("constants.json").read_text())
    rd = lambda f: (ROOT / f).read_text()
    return {"teacher_job --per-source": int(re.search(r'"--per-source", "(\d+)"', rd("jobs/pipeline.py")).group(1)),
            "--cap ctrate": int(re.search(r"ctrate=(\d+)", rd("jobs/build_gated.py")).group(1)),
            "NB": int(re.search(r"NB = min\(B, (\d+)\)", rd("jobs/robustness2.py")).group(1)),
            "docstring": re.search(r"vLLM \d+\.\d+", rd("jobs/llm_reasoning.py")).group(0),
            "CFG": ast.literal_eval(re.search(r"^CFG = (\{.*?\})$", rd("jobs/answer_space2.py"), re.S | re.M).group(1)),
            "answer_space3 CFG": (lambda s: s[s.index("CFG = {"):s.index("}", s.index('"sim_dup"')) + 1])(rd("jobs/answer_space3.py"))}


JOB_CONSTANTS = {"teacher_job --per-source": "jobs/pipeline.py: --per-source argument of the teacher job",
                 "--cap ctrate": "jobs/build_gated.py: --cap ctrate=N", "NB": "jobs/robustness2.py: NB = min(B, N)",
                 "docstring": "jobs/llm_reasoning.py: vLLM version in the module docstring", "CFG": "jobs/answer_space2.py: CFG",
                 "answer_space3 CFG": "jobs/answer_space3.py: CFG (source text)"}


def code_modules():
    """The data-construction modules whose constants the text quotes (identical in the private and public code)."""
    if PUBLIC:
        sys.path.insert(0, str(HERE.parent)); from radkev import data as bd, teacher as te
    else:
        sys.path.insert(0, str(ROOT)); import build_data as bd, teacher as te
    return bd, te

# Final 21-model comparison on the held-out test split (job b71c9617).
COMP = "results/final_test/comparisons.json"
TRANSFER = "results/transfer_paired/transfer_paired.json"  # Kev's out-of-domain suite, paired
DATA_COUNTS = "handoff/tables/data_counts.csv"
LANDSCAPE = "handoff/tables/landscape_test.csv"  # parameter counts

# Released general-purpose decision models scored zero-shot on the test split.
GENERALIST = ["stock27", "stock9", "kev4", "kev08", "laya", "laya_typed", "gliner_decide", "julia1"]
WORDS = dict(enumerate("zero one two three four five six seven eight nine".split()))

LEDGER = []


def put(key, value, source, field, note):
    LEDGER.append({"key": key, "value": value, "source": source, "field": field, "note": note})


def num(x):
    return f"{x:.1f}".replace("-", "\u2212")  # typographic minus


def pct(x):
    return num(100 * x)


def ci(lo, hi):
    return f"{pct(lo)} to {pct(hi)}"


def flip(x):
    """comparisons.json stores vs.<a>.<b> as b minus a; return a minus b."""
    return {"n": x["n"], "delta": -x["delta"], "ci95": [-x["ci95"][1], -x["ci95"][0]]}


def diff(key, x, field, note, sign=None):
    """A paired difference and its 95% CI; sign=+1/-1 asserts the CI excludes zero in that direction."""
    if sign is not None:
        assert (x["ci95"][0] > 0) if sign > 0 else (x["ci95"][1] < 0), f"{key}: CI no longer excludes zero"
    put(key, pct(x["delta"]), COMP, field, note + " (pp)")
    put(key + "_ci", ci(*x["ci95"]), COMP, field + " ci95", "95% CI, paired cluster bootstrap")


def build_numbers():
    C = json.loads(inp(COMP).read_text())
    M, ref, vs = C["models"], C["vs_reference"], C["vs"]
    hk = lambda m: M[m]["subsets"]["human_keys"]

    # Data and test set.
    rows = list(csv.DictReader(open(inp(DATA_COUNTS))))
    train = sum(int(r["records"]) for r in rows if r["split"] == "train")
    put("train_records", f"{train:,}", DATA_COUNTS, "sum of records where split == train", "training records, all sources")
    put("test_questions", f"{M['v2_27']['overall']['n']:,}", COMP, "models.v2_27.overall.n", "held-out test questions")
    put("hk_questions", f"{hk('v2_27')['n']:,}", COMP, "models.v2_27.subsets.human_keys.n",
        "test questions with human answer keys (IU, Eurorad, exam boards)")
    assert all(m in M for m in GENERALIST)
    put("n_generalist", WORDS[len(GENERALIST)], COMP, "models: " + ", ".join(GENERALIST), "released general-purpose decision models")
    put("n_other_generalist", WORDS[len(GENERALIST) - 1], COMP, "models: " + ", ".join(GENERALIST[1:]), "decision models other than Kev-27B")
    land = [r for r in csv.DictReader(open(inp(LANDSCAPE))) if r["Kind"].startswith("decision, generalist")]
    assert len(land) == len(GENERALIST)
    size = [float(r["Params"][:-1]) * (1e-3 if r["Params"].endswith("M") else 1) for r in land]
    put("params_min", f"{min(size):.2g}B", LANDSCAPE, "min Params where Kind starts 'decision, generalist'", "smallest general-purpose decision model")
    put("params_max", f"{max(size):.2g}B", LANDSCAPE, "max Params where Kind starts 'decision, generalist'", "largest general-purpose decision model")

    # Primary comparison (prespecified): RadKev-27B minus Kev-27B, accuracy averaged over tasks.
    x = ref["v2_27"]["overall_macro_acc"]
    tasks = M["v2_27"]["tasks"]
    assert tasks.keys() == M["stock27"]["tasks"].keys()
    macro = lambda m: sum(M[m]["tasks"][t]["acc"] for t in tasks) / len(tasks)
    assert abs(macro("v2_27") - macro("stock27") - x["delta"]) < 1e-9, "macro is not the unweighted mean over tasks"
    put("n_tasks", str(len(tasks)), COMP, "len(models.v2_27.tasks)", "test tasks; macro accuracy is their unweighted mean")
    diff("d_macro", x, "vs_reference.v2_27.overall_macro_acc", "RadKev-27B minus Kev-27B, macro accuracy", +1)
    diff("d_macro_hk", ref["v2_27"]["subsets"]["human_keys"]["macro_acc"], "vs_reference.v2_27.subsets.human_keys.macro_acc",
         "same, human-key tasks only", +1)

    # Eurorad case diagnosis.
    f = "case_diagnosis"
    put("dx_kev27", pct(M["stock27"]["families"][f]["acc"]), COMP, f"models.stock27.families.{f}.acc", "Kev-27B, Eurorad diagnosis (%)")
    put("dx_rk27", pct(M["v2_27"]["families"][f]["acc"]), COMP, f"models.v2_27.families.{f}.acc", "RadKev-27B, Eurorad diagnosis (%)")
    put("dx_n", f"{M['v2_27']['families'][f]['n']:,}", COMP, f"models.v2_27.families.{f}.n", "Eurorad diagnosis questions (one per case)")

    # Human-key accuracy of every system; RadKev-27B must be the highest.
    acc = {m: hk(m)["acc"] for m in M}
    assert max(acc, key=acc.get) == "v2_27", "RadKev-27B is no longer the most accurate system on human-key questions"
    for m in ["v2_27", "stock27", "qwen38", "medgemma_brief", "r9", "stock9"]:
        put(f"hk_acc_{m}", pct(acc[m]), COMP, f"models.{m}.subsets.human_keys.acc", f"{m}, human-key accuracy (%)")

    # Same backbone (Qwen3.8-27B, revision 1d4bf0f), three uses, on human-key questions (micro accuracy).
    sub = lambda x: x["subsets"]["human_keys"]["micro_acc"]
    diff("d_hk_kev27_qwen", flip(sub(ref["qwen38"])), "-vs_reference.qwen38.subsets.human_keys.micro_acc",
         "Kev-27B minus Qwen3.8-27B, human-key accuracy", -1)
    diff("d_hk_rk27_qwen", flip(sub(vs["v2_27"]["qwen38"])), "-vs.v2_27.qwen38.subsets.human_keys.micro_acc",
         "RadKev-27B minus Qwen3.8-27B, human-key accuracy", +1)
    diff("d_hk_rk27_mg", flip(sub(vs["v2_27"]["medgemma_brief"])), "-vs.v2_27.medgemma_brief.subsets.human_keys.micro_acc",
         "RadKev-27B minus MedGemma-27B-text (corrected scoring), human-key accuracy", +1)

    # Specialization versus scale, human-key questions.
    spec27, spec9 = sub(ref["v2_27"]), flip(sub(vs["r9"]["stock9"]))
    scale = flip(sub(ref["stock9"]))  # Kev-27B minus Kev-9B
    assert min(spec27["delta"], spec9["delta"]) > scale["delta"], "specialization no longer adds more than tripling size"
    diff("d_hk_spec27", spec27, "vs_reference.v2_27.subsets.human_keys.micro_acc", "RadKev-27B minus Kev-27B, human-key accuracy", +1)
    diff("d_hk_spec9", spec9, "-vs.r9.stock9.subsets.human_keys.micro_acc", "RadKev-9B minus Kev-9B, human-key accuracy", +1)
    diff("d_hk_scale", scale, "-vs_reference.stock9.subsets.human_keys.micro_acc", "Kev-27B minus Kev-9B, human-key accuracy", +1)
    diff("d_rhk_r9_kev27", flip(vs["r9"]["stock27"]["subsets"]["radiology_human_keys"]["micro_acc"]),
         "-vs.r9.stock27.subsets.radiology_human_keys.micro_acc", "RadKev-9B minus Kev-27B, radiology human-key accuracy", +1)
    diff("d_dx_r9_kev27", flip(vs["r9"]["stock27"]["families"]["case_diagnosis"]["acc"]),
         "-vs.r9.stock27.families.case_diagnosis.acc", "RadKev-9B minus Kev-27B, Eurorad diagnosis", +1)

    # Initialisation ablation (9B, post hoc): Kev-init (RadKev-9B) minus base-init, full data.
    diff("d_hk_init", flip(sub(vs["r9"]["b9"])), "-vs.r9.b9.subsets.human_keys.micro_acc", "Kev-init minus base-init, human-key accuracy", +1)
    T = json.loads(inp(TRANSFER).read_text())["pairs"]["r9-b9"]["acc_micro"]
    assert T["ci95"][0] > 0
    put("d_tr_init", pct(T["delta"]), TRANSFER, "pairs.r9-b9.acc_micro.delta", "Kev-init minus base-init, Kev transfer suite (pp)")
    put("d_tr_init_ci", ci(*T["ci95"]), TRANSFER, "pairs.r9-b9.acc_micro.ci95", "95% CI, Kev's paired bootstrap")


# ======================================================================== Methods: data, training, testing
RUN27, RUN9, RUNB9 = "results/train/v2mg/kev-27b", "results/train/v2x9/kev-9b", "results/train/v2x9/base-9b"
RUN9F10, RUNV1 = "results/train/v2x9f10/kev-9b", "results/train/v1med/kev-27b"
OPEN_MAN = "artifacts/build_open/artifacts/rad_open_manifest.json"
GATED_MAN = "artifacts/gated/artifacts/rad_gated_manifest.json"
TEACHER_MAN = "artifacts/teacher/artifacts/teacher_manifest.json"
TEACHER_V2 = "artifacts/teacher_rescore_v2/artifacts/teacher_rescore.json"   # regenerated labels (MedGemma with the pre-filled segment)
BAKEOFF = "artifacts/bakeoff/artifacts/bakeoff.json"              # split-GPU parity check (Kev-9B)
ENV = "environment/setup.json"                                      # Kev environment probe (see PRIVATE)
TRAIN_RUN = "results/train/v2mg/train.json"
PLAYGROUND = "results/playground/playground_00.json"               # the 420 open-licence test records shown early
LAT27, LAT9 = "artifacts/latency/artifacts/latency_bench.json", "artifacts/latency_9b/artifacts/latency_9b.json"
ROB, ROB2 = "artifacts/robustness/artifacts/robustness.json", "artifacts/robustness2/artifacts/robustness2.json"
AS_LLM = "artifacts/answer_space_llm/artifacts/answer_space_llm.json"
# The Eurorad example (playground item t10) was matched to its case in the public wanglab/eurorad-reasoning release by
# its imaging-findings text: case 14824. Its licence requires this attribution.
EURORAD_EXAMPLE = ("t14", "14824", "Abdominal wall endometriosis: MRI findings and the “Gorgon sign”")
TASK_SOURCE = [("iu_", "iu"), ("eurorad_", "eurorad"), ("medmcqa_", "medmcqa"), ("medqa", "medqa"), ("mmlu_", "mmlu_med"),
               ("pubmedqa", "pubmedqa"), ("medxpertqa", "medxpertqa"), ("ctrate_", "ctrate")]


def val(key):
    return next(x["value"] for x in LEDGER if x["key"] == key)


def rj(rel):
    return json.loads(inp(rel).read_text())


def sci(x):
    m, e = f"{x:.0e}".split("e")
    return rf"${int(m)}\times10^{{{int(e)}}}$"


def n(x):
    return f"{x:,}"


def src_of(task):
    return next(s for p, s in TASK_SOURCE if task.startswith(p))


def build_methods():
    C = json.loads(inp(COMP).read_text()); M = C["models"]
    o, g, tm = rj(OPEN_MAN)["splits"], rj(GATED_MAN)["splits"], rj(TEACHER_MAN)
    dev_cal = rj(RUN27 + "_dev_cal.json")

    # ---- records and questions per split (open + gated + teacher-labelled)
    kept = sum(v["agree"] for v in tm["questions"].values())
    cand = kept + sum(v["disagree"] for v in tm["questions"].values())
    t_dev = sum(v["n"] for k, v in dev_cal["tasks"].items() if k.startswith("teacher_"))
    t_test = sum(v["n"] for k, v in M["v2_27"]["tasks"].items() if k.startswith("teacher_"))
    t_train = kept - t_dev - t_test
    q = {"train": o["train"]["questions"] + g["train"]["questions"] + t_train,
         "dev": o["dev"]["questions"] + g["dev"]["questions"] + t_dev,
         "test": o["test"]["questions"] + g["test"]["questions"] + t_test}
    r = {s: o[s]["records"] + g[s]["records"] + tm["records"][s] for s in ("train", "dev", "test")}
    assert q["dev"] == dev_cal["coverage"]["evaluated_questions"] and q["test"] == M["v2_27"]["overall"]["n"]
    test_cov = rj(COMP.replace("comparisons.json", "test_v2_27.json"))["coverage"]
    assert r["train"] == int(val("train_records").replace(",", "")) and r["test"] == test_cov["evaluated_records"] == test_cov["requested_records"]
    for s in ("train", "dev", "test"):
        put(f"{s}_questions", n(q[s]), f"{OPEN_MAN} + {GATED_MAN} + {TEACHER_MAN}", f"splits.{s}.questions (teacher: kept minus dev and test)", f"{s} questions")
    put("dev_records", n(r["dev"]), f"{OPEN_MAN} + {GATED_MAN} + {TEACHER_MAN}", "splits.dev.records", "development records")
    put("test_records", n(r["test"]), f"{OPEN_MAN} + {GATED_MAN} + {TEACHER_MAN}", "splits.test.records", "test records")

    # ---- teacher-labelled questions. Two runs: the first (training, selection and main-evaluation labels) and the
    # regenerated run with MedGemma's reasoning segment pre-filled (TEACHER_V2), which is the procedure described.
    T2 = rj(TEACHER_V2); tq2 = T2["manifest_new"]["questions"]; lc = T2["label_changes"]
    kept2 = sum(v["agree"] for v in tq2.values()); cand2 = kept2 + sum(v["disagree"] for v in tq2.values())
    assert cand2 == cand, "both runs must score the same candidates"
    assert T2["manifest_old"]["questions"] == tm["questions"], "teacher_rescore's old manifest must be the training run"
    agree = {k: v["agree"] / (v["agree"] + v["disagree"]) for k, v in tq2.items()}
    lo, hi = min(agree, key=agree.get), max(agree, key=agree.get)
    put("teacher_cand", n(cand), TEACHER_V2, "manifest_new sum agree + disagree", "candidate teacher questions")
    put("teacher_kept", n(kept2), TEACHER_V2, "manifest_new sum agree", "retained, regenerated run")
    put("teacher_kept_pct", pct(kept2 / cand), TEACHER_V2, "kept / candidates", "%")
    put("teacher_kept_first", n(kept), TEACHER_MAN, "sum agree", "retained, first run (training labels)")
    put("teacher_kept_first_pct", pct(kept / cand), TEACHER_MAN, "kept / candidates", "%")
    tr = lc["train"]["ALL"]; assert all(lc[s_]["ALL"]["different_label"] == 0 for s_ in ("train", "dev", "test", "ALL"))
    put("teacher_train_first", n(tr["kept_old"]), TEACHER_V2, "label_changes.train.ALL.kept_old", "training questions, first run (deduplicated)")
    put("teacher_train_same", n(tr["same_label"]), TEACHER_V2, "label_changes.train.ALL.same_label", "also retained, same label")
    put("teacher_train_same_pct", pct(tr["same_label"] / tr["kept_old"]), TEACHER_V2, "same / kept_old", "%")
    put("teacher_test_v2", n(T2["test_split"]["ALL"]), TEACHER_V2, "test_split.ALL", "regenerated teacher-labeled test questions")
    put("agree_min", pct(agree[lo]), TEACHER_V2, f"agreement.{lo}", "lowest agreement (%)")
    put("agree_max", pct(agree[hi]), TEACHER_V2, f"agreement.{hi}", "highest agreement (%)")
    assert (lo, hi) == ("contrast", "finding_status")
    agree1 = {k: v["agree"] / (v["agree"] + v["disagree"]) for k, v in tm["questions"].items()}   # first run (training labels)
    lo1, hi1 = min(agree1, key=agree1.get), max(agree1, key=agree1.get)
    assert (lo1, hi1) == ("contrast", "finding_status")
    put("agree_first_min", pct(agree1[lo1]), TEACHER_MAN, f"questions.{lo1}", "lowest agreement, first run (%)")
    put("agree_first_max", pct(agree1[hi1]), TEACHER_MAN, f"questions.{hi1}", "highest agreement, first run (%)")
    JC = job_constants()
    put("teacher_per_source", n(JC["teacher_job --per-source"]), "constants.json", "teacher_job --per-source", "reports and indications per source")

    # ---- source construction constants (read from the code that built the data)
    bd, te = code_modules()
    put("iu_vocab", str(len(bd.IU_VOCAB)), "radkev/data.py", "len(IU_VOCAB)", "IU findings")
    put("n_sections", str(len(bd.SECTIONS)), "radkev/data.py", "len(SECTIONS)", "Eurorad subspecialty sections")
    put("medmcqa_cap", n(bd.MEDMCQA_CAP), "radkev/data.py", "MEDMCQA_CAP", "MedMCQA training questions per subject")
    cap = JC["--cap ctrate"]
    assert cap == g["train"]["records"]
    put("ctrate_cap", n(cap), "constants.json", "--cap ctrate", "CT-RATE training records")

    # ---- task groups on the test split
    hk_tasks = [k for k in M["v2_27"]["tasks"] if not k.startswith(("ctrate_", "teacher_"))]
    put("n_hk_tasks", str(len(hk_tasks)), COMP, "tasks not ctrate_/teacher_", "human-key tasks")
    put("n_machine_tasks", str(len(M["v2_27"]["tasks"]) - len(hk_tasks)), COMP, "ctrate_ and teacher_ tasks", "machine-label tasks")
    hkq = sum(M["v2_27"]["tasks"][k]["n"] for k in hk_tasks)
    assert hkq == M["v2_27"]["subsets"]["human_keys"]["n"]
    put("machine_questions", n(M["v2_27"]["overall"]["n"] - hkq), COMP, "overall.n minus human_keys.n", "machine-label test questions")
    put("rhk_questions", n(M["v2_27"]["subsets"]["radiology_human_keys"]["n"]), COMP, "subsets.radiology_human_keys.n", "radiology human-key questions")

    # ---- training recipe (Kev's trainer arguments as run)
    a27, a9, ab9 = (rj(x + "_training_config.json")["args"] for x in (RUN27, RUN9, RUNB9))
    m27, m9, mf10 = (rj(x + "_training_metrics.json") for x in (RUN27, RUN9, RUN9F10))
    assert a27["lora"] == a9["lora"] and a27["accum"] == a9["accum"] and a27["replay"] == a9["replay"] and a27["epochs"] == 1
    assert m27["truncated_records"] == 0 and m9["truncated_records"] == 0 and a27["perm_kl"] == 0 and a27["ord_w"] == 0
    for k, v, f in [("lora_rank", a27["lora"], "args.lora"), ("head_dim", a27["head_dim"], "args.head_dim"), ("accum", a27["accum"], "args.accum"),
                    ("replay", n(a27["replay"]), "args.replay"), ("max_state", n(a27["max_state"]), "args.max_state"),
                    ("weight_decay", a27["weight_decay"], "args.weight_decay"), ("p_none", f"{a27['p_none']:.2f}", "args.p_none"),
                    ("p_none_distract", f"{a27['p_none_distract']:.2f}", "args.p_none_distract"), ("p_distract", f"{a27['p_distract']:.2f}", "args.p_distract"),
                    ("p_none_pair", f"{a27['p_none_pair']:.2f}", "args.p_none_pair"), ("lr_27", sci(a27["lr"]), "args.lr")]:
        put(k, str(v), RUN27 + "_training_config.json", f, "RadKev-27B training argument")
    put("lr_9", sci(a9["lr"]), RUN9 + "_training_config.json", "args.lr", "RadKev-9B learning rate")
    put("lr_base", sci(ab9["lr"]), RUNB9 + "_training_config.json", "args.lr", "base-init learning rate")
    put("requested_records", n(m27["requested_records"]), RUN27 + "_training_metrics.json", "requested_records", "training records incl. replay")
    put("examples_seen", n(m27["records_seen"]), RUN27 + "_training_metrics.json", "records_seen", "examples incl. augmentation pairs")
    put("steps_27", n(m27["optimizer_steps"]), RUN27 + "_training_metrics.json", "optimizer_steps", "RadKev-27B steps")
    assert m9["optimizer_steps"] == m27["optimizer_steps"]
    put("hours_27", f"{m27['wall_seconds'] / 3600:.1f}", RUN27 + "_training_metrics.json", "wall_seconds/3600", "h")
    put("hours_9", f"{m9['wall_seconds'] / 3600:.1f}", RUN9 + "_training_metrics.json", "wall_seconds/3600", "h")
    put("peak_27", f"{m27['peak_device_bytes'] / 2**30:.1f}", RUN27 + "_training_metrics.json", "peak_device_bytes/2^30", "GiB per GPU")
    put("peak_9", f"{m9['peak_device_bytes'] / 2**30:.1f}", RUN9 + "_training_metrics.json", "peak_device_bytes/2^30", "GiB")
    put("T_27", f"{rj(RUN27 + '_dev_cal.json')['temperature']:.2f}", RUN27 + "_dev_cal.json", "temperature", "fitted temperature")
    put("T_9", f"{rj(RUN9 + '_dev_cal.json')['temperature']:.2f}", RUN9 + "_dev_cal.json", "temperature", "fitted temperature")
    put("f10_records", n(mf10["requested_records"] - a27["replay"]), RUN9F10 + "_training_metrics.json", "requested_records - replay", "10% sample")
    put("f10_steps", n(mf10["optimizer_steps"]), RUN9F10 + "_training_metrics.json", "optimizer_steps", "10% run steps")
    c27 = rj(RUN27 + "_training_config.json"); c9 = rj(RUN9 + "_training_config.json")
    put("rev_kev27", re.split(r"snapshots/|@", c27["init_source"]["resolved"])[1][:7], RUN27 + "_training_config.json", "init_source.resolved", "Kev-27B revision")
    put("rev_kev9", re.split(r"snapshots/|@", c9["init_source"]["resolved"])[1][:7], RUN9 + "_training_config.json", "init_source.resolved", "Kev-9B revision")
    put("rev_qwen27", c27["base_revision"][:7], RUN27 + "_training_config.json", "base_revision", "Qwen3.8-27B revision")
    put("rev_qwen9", c9["base_revision"][:7], RUN9 + "_training_config.json", "base_revision", "Qwen3.5-9B-Base revision")

    # ---- environment and the split-GPU parity check
    env = rj(ENV)["steps"]["kev_env"]
    put("kev_commit", env["commit"][:7], ENV, "steps.kev_env.commit", "Kev commit")
    for k in ("torch", "cuda", "transformers", "peft", "fla"):
        put(f"ver_{k}", env[k].split("+")[0], ENV, f"steps.kev_env.{k}", "library version")
    put("ver_cc1d", rj(TRAIN_RUN)["steps"]["prepare"]["causal_conv1d"], TRAIN_RUN, "steps.prepare.causal_conv1d", "library version")
    par = rj(BAKEOFF)["steps"]["parity"]
    assert par["ok"] and par["max_abs_dp"] == 0.0 and par["argmax_flips"] == 0
    put("parity_q", str(par["questions"]), BAKEOFF, "steps.parity.questions", "questions in the split-GPU parity check")

    # ---- prespecified model selection on the development split
    v2t, v1t = dev_cal["tasks"], rj(RUNV1 + "_dev_cal.json")["tasks"]
    shared = sorted(set(v2t) & set(v1t))
    mac = lambda T: sum(T[k]["acc"] for k in shared) / len(shared)
    put("sel_tasks", str(len(shared)), f"{RUN27}_dev_cal.json + {RUNV1}_dev_cal.json", "shared tasks", "tasks in the selection rule")
    put("sel_primary", f"{100 * mac(v2t):.2f}", RUN27 + "_dev_cal.json", "mean acc over shared tasks", "%")
    put("sel_ablation", f"{100 * mac(v1t):.2f}", RUNV1 + "_dev_cal.json", "mean acc over shared tasks", "%")
    put("sel_diff", f"{100 * (mac(v2t) - mac(v1t)):.2f}".replace("-", "\u2212"),
        RUN27 + "_dev_cal.json + " + RUNV1 + "_dev_cal.json", "primary minus ablation", "pp")
    assert mac(v2t) - mac(v1t) > -0.01, "the prespecified rule would have switched the primary model"

    # ---- testing
    put("transfer_n", n(json.loads(inp(TRANSFER).read_text())["models"]["stock9"]["n"]), TRANSFER, "models.stock9.n", "transfer-suite questions")
    R1 = rj(ROB)
    put("n_boot", n(R1["B"]), ROB, "B", "shared bootstrap resamples (records, stratified by source)")
    put("n_boot_cal", n(min(R1["B"], JC["NB"])), "constants.json", "NB", "resamples for paired calibration differences")
    put("demo_records", n(R1["demo420"]["records_matched"]), ROB, "demo420.records_matched", "test records matched to the demonstration sample")
    put("demo_questions", n(R1["demo420"]["questions_matched"]), ROB, "demo420.questions_matched", "their questions")
    RS = rj(RSN); rcfg = RS["cfg"]; smp = RS["sample"]
    rad_all = [t for t in ("eurorad_dx", "eurorad_route", "medmcqa_rad") if smp["tasks"][t]["sampled"] == smp["tasks"][t]["population"]]
    assert rad_all == ["eurorad_dx", "eurorad_route", "medmcqa_rad"], "claim: every Eurorad and MedMCQA-radiology question is sampled"
    iu_cap = {smp["tasks"][t]["sampled"] for t in smp["tasks"] if t.startswith("iu_")}; assert len(iu_cap) == 1
    put("reason_iu_cap", str(iu_cap.pop()), RSN, "sample.tasks.iu_*.sampled", "IU questions per task")
    put("reason_cap", str(rcfg["cap"]), RSN, "cfg.cap", "questions per other human-key task")
    put("reason_think_cap", n(rcfg["think_cap"]), RSN, "cfg.think_cap", "maximum reasoning tokens")
    put("reason_q", n(smp["questions"]), RSN, "sample.questions", "reasoning sample size")
    put("reason_records", n(smp["records"]), RSN, "sample.records", "reasoning sample records")
    put("ver_vllm", re.search(r"vLLM (\d+\.\d+)", JC["docstring"]).group(1), "constants.json", "docstring", "vLLM version")
    LB = rj(LAT27)
    put("lat_records", n(LB["sample"]["records"]), LAT27, "sample.records", "latency sample records")
    put("lat_q", n(LB["sample"]["questions"]), LAT27, "sample.questions", "latency sample questions")
    put("lat_warmup", str(LB["warmup_excluded"]), LAT27, "warmup_excluded", "warm-up requests excluded")
    pg = rj(PLAYGROUND)
    put("playground_n", str(pg["meta"]["n"]), PLAYGROUND, "meta.n", "test records scored early for a demonstration")
    put("playground_pct", pct(pg["meta"]["n"] / r["test"]), PLAYGROUND, "meta.n / test records", "%")
    put("grid_points", "121", "kev/metrics.py @ f2bb629", "TEMPERATURE_FIT points", "temperature grid")

    # ---- answer-space analysis (post hoc): configuration read from the job script itself
    AS = JC["CFG"]
    lst = lambda xs: ", ".join(map(str, xs[:-1])) + f" or {xs[-1]}"
    J = "constants.json"
    put("as_n_dx", n(M["v2_27"]["families"]["case_diagnosis"]["n"]), COMP, "families.case_diagnosis.n", "Eurorad diagnosis test questions")
    put("as_n_medqa", n(AS["medqa_n"]), J, "CFG.medqa_n", "MedQA sample")
    put("as_n_alt", str(AS["n_alt"]), J, "CFG.n_alt", "LLM distractors requested")
    put("as_dup", f"{AS['sim_dup']:.2f}", J, "CFG.sim_dup", "near-duplicate cosine")
    put("as_sizes", lst(AS["sizes"]), J, "CFG.sizes", "answer-space sizes")
    put("as_llm_sizes", lst(AS["llm_sizes"]), J, "CFG.llm_sizes", "LLM answer-space sizes")
    put("as_add", str(AS["add"]), J, "CFG.add", "LLM distractors added to the original options")
    put("as_lat_sizes", lst(AS["lat_sizes"]), J, "CFG.lat_sizes", "latency sizes")
    put("as_lat_cases", str(AS["lat_cases"]), J, "CFG.lat_cases", "latency cases")
    put("as_lat_warmup", str(AS["lat_warmup"]), J, "CFG.lat_warmup", "warm-up requests excluded")

    write_tables(o, g, tm, M, t_test, te)
    write_data_v3(o, g, tm, t_test)
    write_train_v3()
    write_teacher_agreement(M)
    write_examples(pg, bd, te)
    import figures
    figures.fig_pipeline({x["key"]: x["value"] for x in LEDGER}, HERE / "figures")


V3R = "artifacts/build_v3r/artifacts/build_v3r.json"  # Amendment 3: radiology-only training and development records (data/mix-v3r)
V3 = "artifacts/build_v3/artifacts/build_v3.json"   # RadCases and ReXErr splits, radiology filter of the knowledge test questions


def write_data_v3(o, g, tm, t_test):
    """Table 1 for the v3 design: every source with its records per split and the questions it contributes to the radiology
    benchmark (Addendum 2). Knowledge sources contribute only the questions kept by build_external.is_radiology()."""
    B = rj(V3); rc, rx, rf = B["radcases"], B["rexerr"], B["radiology_filter"]
    put("rc_label_rows", n(rc["build"]["stats"]["label_rows"]), V3, "radcases.build.stats.label_rows", "RadCases label rows (open subsets)")
    put("rc_matched", n(rc["build"]["stats"]["matched_cases"]), V3, "radcases.build.stats.matched_cases", "RadCases cases matched to their text")
    put("rc_multi_panel", n(rc["build"]["stats"]["multi_panel_excluded"]), V3, "radcases.build.stats.multi_panel_excluded", "RadCases cases with several panels (no panel question)")
    put("rc_overlap", n(rc["build"]["stats"]["overlap_with_radkev_splits"]), V3, "radcases.build.stats.overlap_with_radkev_splits", "excluded: text in a RadKev split")
    for s_ in ("train", "dev", "test"):
        put(f"rc_{s_}", n(rc["split"][s_]), V3, f"radcases.split.{s_}", f"RadCases {s_} cases")
    put("rc_test_q", n(rc["split"]["test_q"]), V3, "radcases.split.test_q", "RadCases test questions")
    put("rc_topics", n(rc["build"]["n_topics"]), V3, "radcases.build.n_topics", "ACR Appropriateness Criteria topics")
    put("rc_panel_options", n(rc["build"]["n_panels"]), V3, "radcases.build.n_panels", "panel question options (panels and none)")
    for s_ in ("train", "dev", "test"):
        put(f"rx_{s_}", n(rx[s_]), V3, f"rexerr.{s_}", f"ReXErr {s_} reports")
    put("rx_test_err", n(rx["test_error"]), V3, "rexerr.test_error", "ReXErr test reports with an injected error")
    kept = {"medmcqa": rf["medmcqa_med"]["kept"] + rf["medmcqa_rad"]["kept"], "medqa": rf["medqa"]["kept"], "pubmedqa": rf["pubmedqa"]["kept"],
            "medxpertqa": rf["medxpertqa"]["kept"], "mmlu_med": sum(v["kept"] for k, v in rf.items() if k.startswith("mmlu_"))}
    FL = rj("artifacts/radfilter_final/artifacts/radfilter.json")   # final word list (adds "radiopaque"); identical counts
    assert FL["test"] == {k: {"n": v["n"], "kept": v["kept"]} for k, v in rf.items()} and not FL["test_kept_only_by_radiopaque"]
    assert {k: v["kept"] for k, v in FL["train_final_list"].items()} == {k: rj(V3R)["train"]["kept"][k] for k in ("medmcqa", "medqa")}
    tot_k = sum(kept.values()); tot_n = sum(v["n"] for v in rf.values())
    put("rf_kept", n(tot_k), V3, "radiology_filter.*.kept", "knowledge test questions kept by the radiology filter")
    put("rf_n", n(tot_n), V3, "radiology_filter.*.n", "knowledge test questions screened")
    put("rf_medmcqa_rad", n(rf["medmcqa_rad"]["kept"]), V3, "radiology_filter.medmcqa_rad.kept", "MedMCQA radiology questions (all kept)")
    recs = {s_: {**o[s_]["by_source"], **g[s_]["by_source"], "teacher": tm["records"][s_], "radcases": rc["split"][s_], "rexerr": rx[s_]}
            for s_ in ("train", "dev", "test")}
    R3 = rj(V3R)
    for s_ in ("train", "dev"):   # Amendment 3: MedMCQA and MedQA training and development records restricted to radiology
        k3 = R3[s_]["kept"]
        for key, v in recs[s_].items():
            if key not in ("medmcqa", "medqa"): assert k3.get({"mmlu_med": "mmlu"}.get(key, key), 0) == v, (s_, key, v)
        recs[s_]["medmcqa"], recs[s_]["medqa"] = k3["medmcqa"], k3["medqa"]
        assert sum(recs[s_].values()) == R3[s_]["total"], (s_, sum(recs[s_].values()), R3[s_]["total"])
        for key in ("medmcqa", "medqa"):
            put(f"v3r_{key}_{s_}", n(k3[key]), V3R, f"{s_}.kept.{key}", f"{key} {s_} records kept (radiology)")
            put(f"v3r_{key}_{s_}_all", n(k3[key] + R3[s_]["dropped"][key]), V3R, f"{s_}.kept+dropped.{key}", f"{key} {s_} records before the filter")
    RQ = "artifacts/rsna_radioqa/rsna_radioqa_manifest.json"; MQ = rj(RQ)
    assert MQ["missing_in_appendix"] == [44] and len(MQ["radsafe_answer_index_discrepancies"]) == 2 and not MQ["weak_matches"]
    put("rsna_q", n(MQ["records"]), RQ, "records", "RSNA-RadioQA questions (Appendix S1 of RadioRAG; Q44 absent)")
    put("rsna_published", n(MQ["records"] + len(MQ["missing_in_appendix"])), RQ, "records + missing_in_appendix", "RSNA-RadioQA questions in the original study")
    put("rsna_keyfix", n(len(MQ["radsafe_answer_index_discrepancies"])), RQ, "radsafe_answer_index_discrepancies", "questions whose RadSaFE answer index differs from the published reference answer")
    recs["test"]["rsna"] = MQ["records"]
    bq = {"rsna": MQ["records"], "iu": sum(v for k, v in o["test"]["by_task"].items() if src_of(k) == "iu"),
          "eurorad": sum(v for k, v in o["test"]["by_task"].items() if src_of(k) == "eurorad"),
          **kept, "radcases": rc["split"]["test_q"], "rexerr": rx["test"]}
    rows_ = [("iu", r"IU/Open-i \citep{demnerfushman2016iu}", "Structured label extraction (report)", "Human"),
             ("eurorad", r"Eurorad \citep{eurorad}", "Diagnosis; subspecialty", "Human"),
             ("medmcqa", r"MedMCQA \citep{pal2022medmcqa}", "Radiology knowledge", "Human"),
             ("medqa", r"MedQA \citep{jin2021medqa}", "Radiology knowledge", "Human"),
             ("mmlu_med", r"MMLU \citep{hendrycks2021mmlu}", "Radiology knowledge", "Human"),
             ("pubmedqa", r"PubMedQA \citep{jin2019pubmedqa}", "Radiology knowledge", "Human"),
             ("medxpertqa", r"MedXpertQA \citep{zuo2025medxpertqa}", "Radiology knowledge", "Human"),
             ("rsna", r"RSNA-RadioQA \citep{tayebiarasteh2025radiorag}", "Diagnosis", "Human"),
             ("radcases", r"RadCases \citep{yao2025radcases}", "Imaging appropriateness (ACR)", "Human"),
             ("rexerr", r"ReXErr \citep{rao2025rexerr}", "Error detection (report)", "Construction"),
             ("ctrate", r"CT-RATE \citep{hamamci2026ctrate}", "Structured label extraction (report)", "Classifier"),
             ("teacher", "Teacher-labeled", "Orders, triage, follow-up", "LLM agreement")]
    cell = lambda x: n(x) if x else "–"
    body = []
    for key, name, task, keysrc in rows_:
        b = cell(bq.get(key, 0)) if key not in ("ctrate", "teacher") else "–"
        body.append(" & ".join([name, task, keysrc] + [cell(recs[s_].get(key, 0)) for s_ in ("train", "dev", "test")] + [b]) + r" \\")
    tot = [n(sum(recs[s_].values())) for s_ in ("train", "dev", "test")]
    bench = sum(bq.values())
    put("v3r_steps", "4,620", "runs/v3f-kev-27b-dp.log, job 8d6337a4 (final RadKev-27B v3 run; same count in the cancelled v3r run 1b065b97)", "step N/4620", "optimizer steps, RadKev-27B v3 and RadKev-9B v3")
    acc = 36954   # kev.train rank-0 log, "36954 training requests", identical for 27B and 9B (runs/v3r-kev-{27b,9b}.log, job 1b065b97)
    # same log: "dropped 155 of 37109 records that exceed the training context (1536 state / 2176 branch / 3200 packed tokens)"
    fed = sum(recs["train"].values()) + 1000
    assert (acc + 7) // 8 == 4620 and 0 < fed - acc < 1000, (fed, acc)
    put("v3r_accepted", n(acc), "runs/v3r-kev-27b.log and runs/v3r-kev-9b.log, job 1b065b97", "N training requests", "training records accepted by kev.train, incl. replay")
    put("v3r_fed", n(fed), f"{V3R} + replay", "train total + 1,000", "training records supplied, incl. replay")
    put("v3r_rejected", n(fed - acc), "v3r_fed - v3r_accepted", "difference", "records not accepted by kev.train at loading")
    put("v3_train_records", tot[0], f"{V3} + manifests", "sum of training records", "training records, v3")
    put("v3_bench_q", n(bench), f"{V3} + manifests", "benchmark questions", "radiology benchmark test questions")
    put("v3_bench_human_q", n(bench - rx["test"]), f"{V3} + manifests", "benchmark questions minus ReXErr", "human-assigned benchmark questions")
    RG = "artifacts/radgraph_xl_build/artifacts/radgraph_xl_build.json"; R = rj(RG); st = R["stats"]
    put("rg_reports", n(R["reports"]), RG, "reports", "RadGraph-XL reports (Stanford release)")
    put("rg_per_mod", n(R["by_modality"]["cxr"]), RG, "by_modality.cxr", "reports per modality")
    assert len(set(R["by_modality"].values())) == 1
    put("rg_excluded", n(R["excluded"]), RG, "excluded", "reports excluded for overlap")
    put("rg_records", n(st["records"]), RG, "stats.records", "RadGraph-XL records used")
    rgq = sum(v for k, v in st.items() if k.startswith("q_"))
    assert rgq == st["key_present"] + st["key_absent"] + st["key_uncertain"]
    put("rg_q", n(rgq), RG, "stats.q_*", "RadGraph-XL questions")
    put("rg_q_cxr", n(st["q_cxr"]), RG, "stats.q_cxr", "RadGraph-XL chest radiograph questions remaining")
    (HERE / "generated" / "tab_data.tex").write_text("\n".join([
        r"\begin{tabular}{@{}lllrrrr@{}}", r"\toprule",
        r"Source & Task & Ground truth & \multicolumn{3}{c}{Records} & Benchmark \\ \cmidrule(lr){4-6}",
        r" & & & Train & Dev & Test & questions \\", r"\midrule", *body[:10], r"\midrule", *body[10:], r"\midrule",
        r"RadGraph-XL \citep{delbrouck2024radgraphxl} & Structured label extraction (report) & Human & – & – & " + n(st["records"]) + r" & – \\",   # external test set
        r"\midrule",
        " & ".join(["Total", "", ""] + tot[:2] + [n(int(tot[2].replace(",", "")) + st["records"])] + [n(bench)]) + r" \\", r"\bottomrule", r"\end{tabular}"]) + "\n")

def write_train_v3():
    """Training of the present models (v3f runs: RadKev-27B two data-parallel processes over NVLink pairs, RadKev-9B four
    single-GPU processes), from the training metrics, configs and development calibration copied from the runs."""
    T = "artifacts/train_v3/artifacts/"
    for m, ws in (("27b", 2), ("9b", 4)):
        M = rj(T + f"training_metrics_{m}.json"); C = rj(T + f"kev-{m}-dp_training_config.json")["args"]; D = rj(T + f"dev_cal_{m}.json")
        assert M["optimizer_steps"] == 4620 and M["requested_records"] == 36954 and M["world_size"] == ws
        assert C["batch"] * C["accum"] * ws == 8 and C["replay"] == 1000 and C["seed"] == 0
        put(f"v3_hours_{m}", f"{M['wall_seconds'] / 3600:.1f}", T + f"training_metrics_{m}.json", "wall_seconds / 3600", f"training hours, RadKev-{m.upper()}")
        put(f"v3_examples_{m}", n(M["records_seen"]), T + f"training_metrics_{m}.json", "records_seen", "examples seen incl. minimal pairs")
        put(f"v3_peak_{m}", f"{M['peak_device_bytes'] / 2**30:.1f}", T + f"training_metrics_{m}.json", "peak_device_bytes / 2^30",
            "peak memory, first GPU of the first process (GiB)")
        put(f"v3_T_{m}", f"{D['temperature']:.2f}", T + f"dev_cal_{m}.json", "temperature", "fitted temperature")
        put(f"v3_accum_{m}", n(C["accum"]), T + f"kev-{m}-dp_training_config.json", "args.accum", "accumulation steps per process")
        put(f"v3_wdtype_{m}", {"bf16": "bfloat16", "fp32": "32-bit"}[C["weights_dtype"]], T + f"kev-{m}-dp_training_config.json", "args.weights_dtype", "frozen weight precision")


def tex(s):
    s = str(s)
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("$", r"\$"), ("#", r"\#"), ("_", r"\_"),
                 ("{", r"\{"), ("}", r"\}"), ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}")):
        s = s.replace(a, b)
    return s.replace(r"\textbackslash\{\}", r"\textbackslash{}")


DATA_ROWS = [  # key, source (with citation), text, questions, answer key
    ("iu", r"IU/Open-i \citep{demnerfushman2016iu}", "Chest radiograph reports",
     "Finding present (yes/no); normal study (yes/no); which of four findings is present", "MeSH indexing by human indexers"),
    ("ctrate", r"CT-RATE \citep{hamamci2026ctrate}", "Chest CT reports", "As for IU", "Classifier labels for 18 abnormalities"),
    ("eurorad", r"Eurorad \citep{eurorad}", "Case reports",
     r"Diagnosis among the case's differential; subspecialty section (\V{n_sections} options)", "Case authors"),
    ("medmcqa", r"MedMCQA \citep{pal2022medmcqa}", "Examination questions, all subjects", "Four options", "Examination key"),
    ("medqa", r"MedQA \citep{jin2021medqa}", "USMLE-style clinical vignettes", "Four options", "Examination key"),
    ("mmlu_med", r"MMLU \citep{hendrycks2021mmlu}", "Medical examination questions", "Four options", "Examination key"),
    ("pubmedqa", r"PubMedQA \citep{jin2019pubmedqa}", "Research abstracts", "Yes, no or maybe", "Expert annotation"),
    ("medxpertqa", r"MedXpertQA \citep{zuo2025medxpertqa}", "Expert examination questions", "Up to ten options", "Examination key"),
    ("teacher", "Teacher-labeled", "Reports and clinical indications", "Ten question kinds (Supplementary Table~S1)", "Agreement of two LLM teachers"),
]

TEACHER_ROWS = [  # kind, label, input, answer type, number of admissible answers (from teacher.py)
    ("critical", "Critical finding", "Report", "Yes/no", lambda te: 2),
    ("urgency", "Urgency", "Report", "Ordinal", lambda te: len(te.URGENCY)),
    ("follow_up", "Follow-up", "Report", "Choice", lambda te: len(te.FOLLOW)),
    ("change", "Interval change", "Report with a comparison", "Choice", lambda te: len(te.CHANGE)),
    ("incidental", "Incidental finding", "Report", "Yes/no", lambda te: 2),
    ("finding_status", "Finding status", "Chest radiograph report", "Choice", lambda te: len(te.STATUS)),
    ("exam", "First examination", "Clinical indication", "Choice", lambda te: len(te.EXAMS)),
    ("contrast", "Contrast", "Clinical indication", "Choice", lambda te: len(te.CONTRAST)),
    ("priority", "Priority", "Clinical indication", "Ordinal", lambda te: len(te.PRIORITY)),
    ("appropriate", "Appropriateness", "Indication and ordered examination", "Ordinal", lambda te: len(te.APPROPRIATE)),
]


def write_tables(o, g, tm, M, t_test, te):
    gen = HERE / "generated"; gen.mkdir(exist_ok=True)
    recs = {s: {**o[s]["by_source"], **g[s]["by_source"], "teacher": tm["records"][s]} for s in ("train", "dev", "test")}
    tq = {}
    for k, v in {**o["test"]["by_task"], **g["test"]["by_task"]}.items():
        tq[src_of(k)] = tq.get(src_of(k), 0) + v
    tq["teacher"] = t_test
    cell = lambda x: n(x) if x else "–"
    rows = []
    for key, name, text, qs, keysrc in DATA_ROWS:
        rows.append(" & ".join([name, keysrc] + [cell(recs[s].get(key, 0)) for s in ("train", "dev", "test")] + [n(tq[key])]) + r" \\")
    tot = [n(sum(recs[s].values())) for s in ("train", "dev", "test")] + [n(sum(tq.values()))]
    assert sum(tq.values()) == M["v2_27"]["overall"]["n"]
    (gen / "tab_data.tex").write_text("\n".join([
        r"\begin{tabular}{@{}llrrrr@{}}",
        r"\toprule", r"Source & Answer key & \multicolumn{3}{c}{Records} & Test questions \\ \cmidrule(lr){3-5}",
        r" & & Train & Dev & Test & \\", r"\midrule", *rows,
        r"\midrule", " & ".join(["Total", ""] + tot) + r" \\", r"\bottomrule", r"\end{tabular}"]) + "\n")
    T2 = rj(TEACHER_V2)["manifest_new"]["questions"]
    trows = []
    for kind, label, inp, typ, k in TEACHER_ROWS:
        v, w = tm["questions"][kind], T2[kind]; c = v["agree"] + v["disagree"]
        trows.append(f"{label} & {inp} & {typ} & {k(te)} & {n(c)} & {n(v['agree'])} ({pct(v['agree'] / c)}) & {n(w['agree'])} ({pct(w['agree'] / c)}) " + r"\\")
    kept = sum(v["agree"] for v in tm["questions"].values()); cand = kept + sum(v["disagree"] for v in tm["questions"].values())
    kept2 = sum(v["agree"] for v in T2.values())
    (gen / "tab_teacher.tex").write_text("\n".join([
        r"\begin{tabular}{@{}llllrrr@{}}", r"\toprule",
        r" & & & & & \multicolumn{2}{c}{Retained (\%)} \\ \cmidrule(l){6-7}",
        r"Question kind & Input & Answer type & Answers & Candidates & First run & Regenerated \\", r"\midrule", *trows, r"\midrule",
        f"Total & & & & {n(cand)} & {n(kept)} ({pct(kept / cand)}) & {n(kept2)} ({pct(kept2 / cand)}) " + r"\\", r"\bottomrule", r"\end{tabular}"]) + "\n")


TEACHER_TASKS = [("teacher_report_critical", "Critical finding"), ("teacher_report_urgency", "Urgency"), ("teacher_report_follow_up", "Follow-up"),
                 ("teacher_report_change", "Interval change"), ("teacher_report_incidental", "Incidental finding"),
                 ("teacher_cxr_findings_finding_status", "Finding status"), ("teacher_order_exam", "First examination"),
                 ("teacher_order_contrast", "Contrast"), ("teacher_order_priority", "Priority"), ("teacher_order_appropriate", "Appropriateness")]


def write_teacher_agreement(M):
    """Agreement of the decision models with the regenerated teacher-labeled test labels (TEACHER_V2)."""
    T = rj(TEACHER_V2)["models"]
    cols = [("kev27", "Kev-27B"), ("radkev27", "RadKev-27B"), ("kev9", "Kev-9B"), ("radkev9", "RadKev-9B")]
    rows = []
    for task, label in TEACHER_TASKS:
        rows.append(f"{label} & {n(T['radkev27']['tasks'][task]['n'])} & " + " & ".join(pct(T[m]["tasks"][task]["acc"]) for m, _ in cols) + r" \\")
    tot = T["radkev27"]["overall"]["n"]
    assert tot == sum(T["radkev27"]["tasks"][k]["n"] for k, _ in TEACHER_TASKS)
    rows.append(r"\midrule" + f" All & {n(tot)} & " + " & ".join(pct(T[m]["overall"]["acc"]) for m, _ in cols) + r" \\")
    for m, nm in (("radkev27", "rk27"), ("kev27", "k27"), ("radkev9", "rk9"), ("kev9", "k9")):
        put(f"tagree_{nm}", pct(T[m]["overall"]["acc"]), TEACHER_V2, f"models.{m}.overall.acc", "agreement with regenerated teacher test labels (%)")
    (HERE / "generated" / "tab_teacher_agreement.tex").write_text("\n".join([
        r"\begin{tabular}{@{}lrrrrr@{}}", r"\toprule",
        r" & & \multicolumn{4}{c}{Agreement with the teacher label (\%)} \\ \cmidrule(l){3-6}",
        r"Question kind & Test questions & " + " & ".join(c for _, c in cols) + r" \\", r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}"]) + "\n")


EXAMPLES = [  # source, heading, licence note
    ("iu", "IU/Open-i chest radiograph report", "development and test only; CC BY-NC-ND 4.0"),
    ("eurorad", "Eurorad case report", None),
    ("medmcqa", "MedMCQA examination question", "Apache-2.0"),
    ("medqa", "MedQA examination question", "CC BY 4.0"),
    ("mmlu_med", "MMLU examination question", "development and test only; MIT"),
    ("pubmedqa", "PubMedQA research question", "development and test only; MIT"),
    ("medxpertqa", "MedXpertQA examination question", "development and test only; MIT"),
]


def state_text(s):
    return s if isinstance(s, str) else "\n".join(f"{k}: {v}" for k, v in s.items())


def state_tex(s):
    if isinstance(s, str):
        return r"\\ ".join(tex(line) for line in s.splitlines() if line.strip())
    return r"\\ ".join(rf"\textit{{{tex(k[:1].upper() + k[1:])}:}} {tex(v)}" for k, v in s.items())


def question_tex(q):
    if q["type"] == "noul":
        return rf"{tex(q['instructions'])} \textit{{(yes/no)}} \textit{{Answer:}} {'yes' if q['label'] else 'no'}."
    if q["type"] == "choice":
        key = lambda k: tex(k).replace(r"\_", r"\_\allowbreak{}")          # long section keys may break at underscores
        show = lambda k, v: key(k) if v is None else f"{key(k)}: {tex(v)}"   # options may carry no description
        opts = "; ".join((rf"\textbf{{{show(k, v)}}}" if k == q["label"] else show(k, v)) for k, v in q["criteria"].items())
        return rf"{tex(q['instructions'])} \textit{{Options:}} {opts}. \textit{{Answer:}} {tex(q['label'])}."
    levels = "; ".join(f"{i}: {tex(v)}" for i, v in enumerate(q["criteria"]))
    return rf"{tex(q['instructions'])} \textit{{Levels:}} {levels}. \textit{{Answer:}} {q['label']}."


def write_examples(pg, bd, te):
    if PUBLIC:   # the 420 records are not redistributed; the public bundle carries the text generated from them
        shutil.copy(inp("examples.tex"), HERE / "generated" / "examples.tex"); return
    by = {}
    for it in pg["items"]:
        by.setdefault(it["source"], []).append(it)
    out = []
    for src, head, lic in EXAMPLES:
        items = sorted(by[src], key=lambda i: (len(state_text(i["state"])), i["id"]))
        it = items[len(items) // 4]   # the record at the first quartile of state length (selection rule stated in the text)
        if src == "eurorad":
            assert it["id"] == EURORAD_EXAMPLE[0], "Eurorad example changed; re-identify its case for attribution"
            lic = (rf"case {EURORAD_EXAMPLE[1]}, \emph{{{EURORAD_EXAMPLE[2]}}}, \url{{https://www.eurorad.org/case/{EURORAD_EXAMPLE[1]}}}; "
                   r"\textcopyright{} European Society of Radiology, CC BY-NC-SA 4.0")
        qs = "\n".join(rf"  \item {question_tex(q)}" for q in it["questions"].values())
        out.append(rf"""\paragraph{{{head}}} ({lic}).
\begin{{quote}}\footnotesize\raggedright
\textit{{State.}} {state_tex(it['state'])}
\begin{{enumerate}}[label=Q\arabic*.,leftmargin=2.2em,itemsep=1pt,topsep=2pt]
{qs}
\end{{enumerate}}
\end{{quote}}""")
    present, normal, which = bd.PRESENT_T[0].format(f="a pleural effusion"), bd.NORMAL_T[0], bd.WHICH_T[0]
    out.append(rf"""\paragraph{{CT-RATE chest CT report}} (CC BY-NC-SA 4.0; access-controlled, report text not reproduced).
\begin{{quote}}\footnotesize\raggedright
\textit{{State.}} A CT-RATE chest CT report with clinical information, technique, findings and impression.
\begin{{enumerate}}[label=Q\arabic*.,leftmargin=2.2em,itemsep=1pt,topsep=2pt]
  \item {tex(present)} \textit{{(yes/no; three such questions per report)}}
  \item {tex(normal)} \textit{{(yes/no)}}
  \item {tex(which)} \textit{{(one finding present and three absent)}}
\end{{enumerate}}
Answers follow the dataset's classifier labels.
\end{{quote}}""")
    fq = next(x for x in te.REPORT_Q if x[0] == "follow_up"); eq = next(x for x in te.ORDER_Q if x[0] == "exam")
    fol = "; ".join(tex(v) for v in te.FOLLOW.values()); ex = "; ".join(tex(v) for v in te.EXAMS.values())
    out.append(rf"""\paragraph{{Teacher-labeled questions}} (states from CheXpert Plus, ReXGradient-160K and CT-RATE reports and from clinical indications, including Eurorad case presentations).
\begin{{quote}}\footnotesize\raggedright
\textit{{Report question.}} State: a radiology report. {tex(fq[2][0])} \textit{{Options:}} {fol}.\\
\textit{{Order question.}} State: the clinical indication only. {tex(eq[2][0])} \textit{{Options:}} {ex}.\\
The label is the option that both teachers ranked first; the training target is the mean of the two teachers' distributions.
\end{{quote}}""")
    (HERE / "generated" / "examples.tex").write_text("\n\n".join(out) + "\n")

AS_OUT = "artifacts/answer_space2/artifacts/answer_space2.json"
AS_MODELS = [("radkev27", "RadKev-27B"), ("kev27", "Kev-27B"), ("radkev9", "RadKev-9B"), ("kev9", "Kev-9B")]


def answer_space_summary(path):
    """Accuracy with bootstrap CIs per model, source and answer space; paired contrasts; latency quartiles."""
    import numpy as np
    D = json.loads(Path(path).read_text()); src = np.array(D["src"]); rng = np.random.default_rng(0); B = 2000
    def vec(m, c):
        s = D["models"][m]["correct"][c]; return np.array([1.0 if ch == "1" else 0.0 if ch == "0" else np.nan for ch in s])
    def boot(x):
        x = x[~np.isnan(x)]; idx = rng.integers(0, len(x), (B, len(x))); bs = x[idx].mean(1)
        return {"acc": float(x.mean()), "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "n": int(len(x))}
    def pboot(a, b):
        ok = ~np.isnan(a) & ~np.isnan(b); d = a[ok] - b[ok]; idx = rng.integers(0, len(d), (B, len(d))); bs = d[idx].mean(1)
        return {"d": float(d.mean()), "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))], "n": int(ok.sum())}
    S = {"acc": {}, "lat": {}, "diff": {}, "gen": D["gen"], "conds": D["conditions"], "raw": D}
    for m, _ in AS_MODELS:
        if m not in D["models"]: continue
        for c in D["conditions"]:
            kind, K = (c.rsplit("_", 1)[0], int(c.rsplit("_", 1)[1])) if c[-1].isdigit() else (c, None)
            for s in ("eurorad_dx", "medqa"):
                x = vec(m, c)[src == s]
                if np.isfinite(x).any(): S["acc"][(m, s, kind, K)] = boot(x)
    for s in ("eurorad_dx", "medqa"):
        sel = src == s
        for c in D["conditions"]:
            for a, b_ in (("radkev27", "kev27"), ("radkev9", "kev9")):
                if a in D["models"] and b_ in D["models"]: S["diff"][(s, c, a, b_)] = pboot(vec(a, c)[sel], vec(b_, c)[sel])
        for m, _ in AS_MODELS:
            if m not in D["models"]: continue
            if "orig_llm" in D["conditions"]: S["diff"][(s, "orig_llm-orig", m)] = pboot(vec(m, "orig_llm")[sel], vec(m, "orig")[sel])
            for K in (2, 4, 8, 16):
                if f"llm_{K}" in D["conditions"] and f"rand_{K}" in D["conditions"]:
                    S["diff"][(s, f"llm_{K}-rand_{K}", m)] = pboot(vec(m, f"llm_{K}")[sel], vec(m, f"rand_{K}")[sel])
                if f"llm_{K}" in D["conditions"] and f"sim_{K}" in D["conditions"]:
                    S["diff"][(s, f"llm_{K}-sim_{K}", m)] = pboot(vec(m, f"llm_{K}")[sel], vec(m, f"sim_{K}")[sel])
    for m, rows in D["latency"].items():
        by = {}
        for K, ms in rows: by.setdefault(K, []).append(ms)
        for K, v in by.items():
            v = np.array(v); S["lat"][(m, K)] = {"median": float(np.median(v)), "q1": float(np.quantile(v, .25)), "q3": float(np.quantile(v, .75)), "n": len(v)}
    return S

def build_answer_space():
    """Section 3 answer-space results: macros, Supplementary Table S3 and Figure 2, from jobs/answer_space2.py."""
    import numpy as np
    S = answer_space_summary(inp(AS_OUT)); D = S["raw"]; src = AS_OUT
    A = lambda m, s, kind, K=None: S["acc"][(m, s, kind, K)]
    # the original-option condition must reproduce the main evaluation (same questions, same models)
    put("as_q", n(len(D["rids"])), src, "rids", "questions in the answer-space study")
    for m, nm in (("radkev27", "rk27"), ("kev27", "k27"), ("radkev9", "rk9"), ("kev9", "k9")):
        for s_, sn in (("eurorad_dx", "dx"), ("medqa", "mq")):
            for kind, K in (("orig", None), ("orig_llm", None), ("rand", 2), ("rand", 64), ("sim", 2), ("sim", 8), ("sim", 16), ("sim", 64), ("llm", 2), ("llm", 16)):
                v = A(m, s_, kind, K); key = f"as_{nm}_{sn}_{kind}{K or ''}"
                put(key, pct(v["acc"]), src, f"models.{m}.correct.{kind}{'_' + str(K) if K else ''}[{s_}]", "accuracy %")
                put(key + "_ci", ci(*v["ci"]), src, "bootstrap", "95% CI, bootstrap over questions")
    def d(key, k, sign=None, note=""):
        x = S["diff"][k]
        if sign is not None:
            assert (x["ci"][0] > 0) if sign > 0 else (x["ci"][1] < 0), f"{key}: CI no longer excludes zero"
        put(key, pct(x["d"]), src, "paired bootstrap " + "/".join(map(str, k)), note + " (pp)")
        put(key + "_ci", ci(*x["ci"]), src, "paired bootstrap", "95% CI"); put(key + "_n", n(x["n"]), src, "n", "paired questions")
    # specialization effect by answer space (Eurorad, 27B and 9B)
    for c, sign in (("orig", 1), ("llm_8", 1), ("llm_16", 1), ("orig_llm", 1), ("sim_8", 1), ("rand_64", None), ("rand_2", None)):
        d(f"as_spec27_dx_{c.replace('_', '')}", ("eurorad_dx", c, "radkev27", "kev27"), sign, f"RadKev-27B minus Kev-27B, Eurorad, {c}")
    for c, sign in (("orig", 1), ("llm_8", 1), ("rand_64", None), ("sim_8", 1)):
        d(f"as_spec9_dx_{c.replace('_', '')}", ("eurorad_dx", c, "radkev9", "kev9"), sign, f"RadKev-9B minus Kev-9B, Eurorad, {c}")
    for c in ("orig", "orig_llm", "llm_8"):
        d(f"as_spec27_mq_{c.replace('_', '')}", ("medqa", c, "radkev27", "kev27"), None, f"RadKev-27B minus Kev-27B, MedQA, {c}")
    # LLM distractors against the pools, and added to the original options
    for m, nm in (("radkev27", "rk27"), ("kev27", "k27")):
        for s_, sn in (("eurorad_dx", "dx"), ("medqa", "mq")):
            d(f"as_add_{nm}_{sn}", (s_, "orig_llm-orig", m), -1, f"{m}: original + LLM distractors minus original, {s_}")
            d(f"as_llmrand_{nm}_{sn}", (s_, "llm_8-rand_8", m), -1, f"{m}: LLM minus random pool at K=8, {s_}")
    d("as_llmsim_rk27_dx", ("eurorad_dx", "llm_8-sim_8", "radkev27"), 1, "RadKev-27B: LLM minus similar pool at K=8, Eurorad")
    # claim: for every model and source, 64 random options are answered more accurately than 2 similar ones
    for m, _ in AS_MODELS:
        for s_ in ("eurorad_dx", "medqa"):
            assert A(m, s_, "rand", 64)["acc"] > A(m, s_, "sim", 2)["acc"], (m, s_)
    for m, nm in (("radkev27", "rk27"), ("kev27", "k27")):
        for s_, sn in (("eurorad_dx", "dx"), ("medqa", "mq")):
            x = S["diff"][(s_, "orig_llm-orig", m)]
            put(f"as_drop_{nm}_{sn}", pct(-x["d"]), src, "paired bootstrap", "accuracy reduction from adding LLM distractors (pp)")
            put(f"as_drop_{nm}_{sn}_ci", ci(-x["ci"][1], -x["ci"][0]), src, "paired bootstrap", "95% CI")
    # the original-option condition reproduces the main evaluation (same questions, models and options)
    main = {(r["model"], r["task"]): float(r["acc %"]) for r in csv.DictReader(open(inp("handoff/tables/per_task_test.csv")))}
    for m, lab in (("radkev27", "RadKev-27B (final)"), ("kev27", "Kev-27B (stock)")):
        assert abs(100 * A(m, "eurorad_dx", "orig")["acc"] - main[(lab, "eurorad_dx")]) < 0.05, f"{m}: answer-space orig != main eval"
    # distractor similarity to the key (PubMedBERT cosine), mean over questions
    for c in ("orig", "rand_8", "sim_8", "llm_8"):
        x = [v for v in D["dcos"][c] if v is not None]
        put(f"as_cos_{c.replace('_', '')}", f"{np.mean(x):.2f}", src, f"dcos.{c}", "mean cosine of distractors to the key")
    # generation
    kept = np.array(D["gen"]["kept"]); put("as_gen_full", n(int((kept >= 15).sum())), src, "gen.kept", "questions with 15 usable LLM distractors")
    put("as_gen_med", n(int(np.median(kept))), src, "gen.kept", "median usable LLM distractors per question")
    put("as_gen_dropkey", n(D["gen"]["dup_key"]), src, "gen.dup_key", "LLM candidates dropped as near-duplicates of the key")
    for c in ("llm_8", "llm_16"):
        put(f"as_n_{c.replace('_', '')}", n(sum(1 for k in D["K"][c] if k)), src, f"K.{c}", "questions with this answer space")
    # latency
    for m, nm in (("radkev27", "rk27"), ("radkev9", "rk9")):
        for K in (2, 16, 64, 128):
            put(f"as_lat_{nm}_{K}", f"{S['lat'][(m, K)]['median']:.0f}", src, f"latency.{m}", f"median ms, K={K}")
    put("as_lat_n", n(min(v["n"] for v in S["lat"].values())), src, "latency", "minimum timed requests per size")
    write_answer_space_table(S)
    # simplified Figure 9: most similar alternatives; decision models (answer_space2) and LLMs (answer_space_llm, <= 16 options)
    acc = {s: {} for s in ("eurorad_dx", "medqa")}
    for (m, s, k, K), v in S["acc"].items():
        if k == "sim" and m in ("radkev27", "kev27") and K: acc[s].setdefault(m, []).append((K, 100 * v["acc"]))
    lat = {m: [(K, v["median"], v["q1"], v["q3"]) for (mm, K), v in S["lat"].items() if mm == m] for m in ("radkev27", "radkev9")}
    LL = inp(AS_LLM)
    if LL.exists():
        import numpy as np
        L = json.loads(LL.read_text()); src = dict(zip(S["raw"]["rids"], S["raw"]["src"]))
        for m in ("qwen38", "medgemma_fix"):
            if m not in L: continue
            for c, d in L[m]["correct"].items():
                if not c.startswith("sim_"): continue
                K = int(c.split("_")[1])
                for s in ("eurorad_dx", "medqa"):
                    xs = [v for rid, v in d.items() if src.get(rid) == s]
                    if xs: acc[s].setdefault(m, []).append((K, 100 * sum(xs) / len(xs)))
            by = {}
            for K, ms in L[m]["latency"]: by.setdefault(K, []).append(ms)
            lat[m] = [(K, float(np.median(v)), float(np.quantile(v, .25)), float(np.quantile(v, .75))) for K, v in sorted(by.items())]
    if LL.exists():
        L = json.loads(LL.read_text()); srcm = dict(zip(S["raw"]["rids"], S["raw"]["src"]))
        def la(m, c, sname):
            xs = [v for rid, v in L[m]["correct"].get(c, {}).items() if srcm.get(rid) == sname]
            return 100 * sum(xs) / len(xs)
        for m, nm in (("qwen38", "q"), ("medgemma_fix", "mg")):
            for sname, sn in (("eurorad_dx", "dx"), ("medqa", "mq")):
                for c in ("orig", "sim_2", "sim_8", "sim_16", "rand_16", "llm_8"):
                    put(f"asl_{nm}_{sn}_{c.replace('_', '')}", f"{la(m, c, sname):.1f}", AS_LLM, f"{m}.correct.{c}[{sname}]", "accuracy (%)")
            lm = {K: v for K, v, *_ in lat[m]}
            for K in (2, 16): put(f"asl_lat_{nm}_{K}", f"{lm[K]:.0f}", AS_LLM, f"{m}.latency median K={K}", "ms per request")
        main_q = 100 * float(next(r for r in csv.DictReader(open(inp(RB_BLIND))) if r["task"] == "eurorad_dx" and r["system"] == "qwen38")["full"])
        main_m = 100 * float(next(r for r in csv.DictReader(open(inp(RB_BLIND))) if r["task"] == "eurorad_dx" and r["system"] == "medgemma_fix")["full"])
        assert abs(la("qwen38", "orig", "eurorad_dx") - main_q) < 0.05, "Qwen original options must reproduce the main evaluation"
        put("asl_mg_orig_main", f"{main_m:.1f}", RB_BLIND, "eurorad_dx.medgemma_fix.full", "MedGemma Eurorad accuracy, main evaluation (%)")
        assert abs(la("medgemma_fix", "orig", "eurorad_dx") - main_m) < 1.5
        for K in (2, 8, 16):
            assert la("qwen38", f"sim_{K}", "eurorad_dx") < 100 * S["acc"][("radkev27", "eurorad_dx", "sim", K)]["acc"], "claim: RadKev-27B > Qwen with similar alternatives (Eurorad)"
        # Supplementary table: LLM answer spaces
        conds = [("Original options", "orig"), ("Original + LLM", "orig_llm")] + [(f"Random pool, K = {K}", f"rand_{K}") for K in (2, 4, 8, 16)] + \
                [(f"Similar pool, K = {K}", f"sim_{K}") for K in (2, 4, 8, 16)] + [(f"LLM, K = {K}", f"llm_{K}") for K in (2, 4, 8, 16)]
        rows = [f"{lab} & " + " & ".join(f"{la(m, c, sname):.1f}" for sname in ("eurorad_dx", "medqa") for m in ("qwen38", "medgemma_fix")) for lab, c in conds]
        _tab("supp_answer_space_llm", "@{}lrrrr@{}", r"Answer space & \multicolumn{2}{c}{Eurorad diagnosis} & \multicolumn{2}{c}{MedQA} \\ & Qwen3.8-27B & MedGemma & Qwen3.8-27B & MedGemma", rows, midrules=(2, 6, 10))
    import figures
    figures.fig_answerspace_simple(acc, lat, HERE / "figures")


def write_answer_space_table(S):
    rows = [("Original options", "orig", None), ("Original + LLM", "orig_llm", None)]
    rows += [(f"Random pool, K = {K}", "rand", K) for K in (2, 4, 8, 16, 32, 64)]
    rows += [(f"Similar pool, K = {K}", "sim", K) for K in (2, 4, 8, 16, 32, 64)]
    rows += [(f"LLM, K = {K}", "llm", K) for K in (2, 4, 8, 16)]
    out = [r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
           r" & \multicolumn{4}{c}{Eurorad diagnosis} & & \multicolumn{4}{c}{MedQA} \\ \cmidrule{2-5}\cmidrule{7-10}",
           r"Answer space & RK-27B & Kev-27B & RK-9B & Kev-9B & & RK-27B & Kev-27B & RK-9B & Kev-9B \\", r"\midrule"]
    for lab, kind, K in rows:
        cells = []
        for s_ in ("eurorad_dx", "medqa"):
            cells += [pct(S["acc"][(m, s_, kind, K)]["acc"]) for m, _ in AS_MODELS] + [""]
        out.append(f"{lab} & " + " & ".join(cells[:-1]) + r" \\")
        if kind in ("orig_llm",) or (kind == "rand" and K == 64) or (kind == "sim" and K == 64): out.append(r"\addlinespace")
    out += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "generated" / "tab_answer_space.tex").write_text("\n".join(out) + "\n")


RB = "results_book/numbers.json"   # the definitive results package (shared 2,000-resample bootstrap)


def build_rb():
    """Abstract and Introduction numbers from the results book (fractions there; percentage points here)."""
    R = rj(RB)
    def acc(key, k):
        put(key, pct(R[k]), RB, k, "accuracy %"); put(key + "_ci", ci(R[k + "_lo"], R[k + "_hi"]), RB, k + "_lo/_hi", "95% CI")
    def dif(key, k, sign=None):
        lo, hi = R[k + "_lo"], R[k + "_hi"]
        if sign is not None:
            assert (lo > 0) if sign > 0 else (hi < 0), f"{key}: CI no longer excludes zero"
        put(key, pct(R[k]), RB, k, "difference (pp)"); put(key + "_ci", ci(lo, hi), RB, k + "_lo/_hi", "95% CI")
    dif("rb_prim", "prim_macro_overall_nodemo", 1)          # prespecified: demo records excluded
    dif("rb_prim_hk", "prim_macro_human_keys_nodemo", 1)
    dif("rb_spec27", "d_v2_27_stock27_human_keys", 1)        # per question
    dif("rb_scale", "d_stock27_stock9_human_keys", 1)
    dif("rb_did", "did_specialisation27_minus_scale_hk", 1)
    dif("rb_spec27_tm", "dm_v2_27_stock27_human_keys", 1)    # task mean
    dif("rb_scale_tm", "dm_stock27_stock9_human_keys", 1)
    assert R["dm_stock27_stock9_human_keys"] > R["dm_v2_27_stock27_human_keys"], "claim: scale exceeds specialization as a task mean"
    mg = "medgemma_fix" if "acc_medgemma_fix_human_keys" in R else "medgemma_brief"   # single-BOS rescoring supersedes
    for m in ("stock27", "qwen38", "v2_27", mg):
        acc(f"rb_acc_{'medgemma_brief' if m == mg else m}", f"acc_{m}_human_keys")
    assert R["acc_v2_27_human_keys"] == max(v for k, v in R.items() if k.startswith("acc_") and k.endswith("_human_keys") and "radiology" not in k), "RadKev-27B must be the most accurate system"
    dif("rb_rk27_qwen", "d_v2_27_qwen38_human_keys", 1)
    dif("rb_qwen_k27", "d_qwen38_stock27_human_keys", 1)
    dif("rb_init", "d_r9_b9_human_keys", 1)
    dif("rb_r9_k27", "d_r9_stock27_human_keys", 1)
    put("rb_blind_rk27", pct(R["blind_eurorad_dx_v2_27"]), RB, "blind_eurorad_dx_v2_27", "Eurorad dx accuracy, case withheld (%)")
    put("rb_blind_k27", pct(R["blind_eurorad_dx_stock27"]), RB, "blind_eurorad_dx_stock27", "Eurorad dx accuracy, case withheld (%)")
    assert R["blind_gain_v2_27_stock27_eurorad_dx_blind"] > R["blind_gain_v2_27_stock27_eurorad_dx_full"], "claim: gain without the case exceeds the gain with it"


def build_fig_compare():
    """Figure 2: accuracy on human-key questions (results book) and latency per question (latency_bench, latency_9b)."""
    R = rj(RB)
    L27 = rj(LAT27); L9 = rj(LAT9)
    assert L27["sample"] == L9["sample"], "both latency jobs must time the same records"
    lat = {"v2_27": L27["models"]["radkev27"]["per_question_ms"], "stock27": L27["models"]["stock27"]["per_question_ms"],
           "r9": L9["models"]["radkev9"]["per_question_ms"], "stock9": L9["models"]["stock9"]["per_question_ms"],
           "qwen38": L27["models"]["qwen38"]["letter"]["per_question_ms"], "medgemma_brief": L27["models"]["medgemma"]["letter"]["per_question_ms"]}
    spec = [("dm", "RadKev-27B", "v2_27", True), ("dm", "Kev-27B", "stock27", False), ("dm", "RadKev-9B", "r9", True),
            ("dm", "Kev-9B", "stock9", False), ("llm", "Qwen3.8-27B", "qwen38", False), ("llm", "MedGemma-27B-text", "medgemma_brief", False)]
    mg = "medgemma_fix" if "acc_medgemma_fix_human_keys" in R else "medgemma_brief"
    rows = []
    for g, label, k, sp in spec:
        a = f"acc_{mg if k == 'medgemma_brief' else k}_human_keys"
        rows.append({"group": g, "label": label, "specialised": sp, "acc": R[a], "lo": R[a + "_lo"], "hi": R[a + "_hi"],
                     "lat_med": lat[k]["median"], "lat_p95": lat[k]["p95"]})
        put(f"f2_lat_{k}", f"{lat[k]['median']:.0f}", LAT27 + " + " + LAT9, "per_question_ms.median", "median latency per question (ms)")
        put(f"f2_lat_{k}_p95", f"{lat[k]['p95']:.0f}", LAT27 + " + " + LAT9, "per_question_ms.p95", "95th percentile (ms)")
    put("f2_lat_records", n(L27["sample"]["records"]), LAT27, "sample.records", "records timed")
    put("f2_lat_warmup", str(L27["warmup_excluded"]), LAT27, "warmup_excluded", "warm-up requests excluded")
    put("f2_qwen_direct", f"{L27['models']['qwen38']['direct']['per_question_ms']['median']:.0f}", LAT27, "qwen38.direct.per_question_ms.median", "generated letter (ms)")
    import figures
    figures.fig_compare(rows, HERE / "figures")


def write_design_v3():
    """Design constants of the answer-space and latency studies as run (jobs/answer_space3.py CFG, jobs latency_v3 arguments)."""
    import re
    cfg = job_constants()["answer_space3 CFG"]   # read from the job script here, from inputs/constants.json in the public bundle
    assert '"sizes": [2, 4, 16, 64, 255]' in cfg and '"eurorad_dx": 30, "rsna_radioqa": 30' in cfg and '"second_draw": False' in cfg
    assert '"lat_sizes": [2, 16, 64, 255], "lat_cases": 15, "lat_warmup": 5' in cfg and '"llm_sizes": [2, 4, 8, 16]' in cfg
    J = "jobs/answer_space3.py CFG"
    for k, v, note in (("as3_n_dx", "30", "Eurorad diagnosis questions"), ("as3_n_rsna", "30", "RSNA-RadioQA questions"),
                       ("as3_sizes", "2, 4, 16, 64 and 255", "answer-space sizes"), ("as3_llm_sizes", "2, 4, 8 and 16", "LLM-distractor sizes"),
                       ("as3_lat_cases", "15", "latency cases"), ("as3_lat_sizes", "2, 16, 64 and 255", "latency sizes"), ("as3_lat_warmup", "5", "warm-up excluded")):
        put(k, v, J, k, note)
    put("lat3_records", "60", "monitor queue commit 3bbfb94, latency_bench --n_records (latency_v3.json sample.records)", "n_records", "latency records")
    V3N = HERE / "v3" / "numbers_v3.csv"   # once the jobs land, the measured n must equal the design constants stated in the Methods
    if V3N.exists():
        R3 = {r["key"]: r["value"] for r in csv.DictReader(open(V3N))}
        for mk, rk in (("lat3_records", "r3_lat_records"), ("lat3_reason_q", "r3_lt_qthink_n"), ("as3_n_dx", "r3_as_n_dx"),
                       ("as3_n_rsna", "r3_as_n_rsna"), ("as3_lat_cases", "r3_as_lat_cases")):
            mv = {"lat3_records": "60", "lat3_reason_q": "54", "as3_n_dx": "30", "as3_n_rsna": "30", "as3_lat_cases": "15"}[mk]   # reasoning: 8 timed, the first excluded as warm-up (Supplementary Note S5)
            if rk in R3 and "missing" not in R3[rk]: assert R3[rk].replace(",", "") == mv, f"{rk}={R3[rk]} but the Methods state {mv}"
    put("lat3_reason_q", "55", "monitor queue commit 3bbfb94, latency_bench --n_reasoning", "n_reasoning", "questions timed with reasoning (first excluded as warm-up)")


def build_fig_training_v3():
    """Figure 3 for the present models: rank-0 training loss of the v3f runs and development accuracy by source (released Kev
    vs fine-tuned, dev_stock vs dev_cal). Overwrites the pilot figure written by build_fig_training()."""
    import numpy as np
    T = "artifacts/train_v3/artifacts/"
    # provenance: the checkpoints scored as RadKev-27B / RadKev-9B in eval_v3 must be the runs whose logs are plotted
    CK = rj("artifacts/eval_v3/artifacts/eval_v3_status.json")["checkpoints"]
    for k, run in (("v3_27", "v3f-kev-27b-dp"), ("v3_9", "v3f-kev-9b-dp")):
        assert CK[k].rstrip("/").endswith(f"runs/{run}/checkpoint"), f"{k} was evaluated from {CK[k]}, not {run}"
    L, A, TS = {}, {}, {}
    for k, f, cfg, met in (("rk27", "trainlog_v3f-kev-27b-dp.json", "kev-27b-dp_training_config.json", "training_metrics_27b.json"),
                           ("rk9", "trainlog_v3f-kev-9b-dp.json", "kev-9b-dp_training_config.json", "training_metrics_9b.json")):
        run = f[len("trainlog_"):-len(".json")]
        assert rj(T + f)["run"] == run and rj(T + cfg)["args"]["out"].rstrip("/").endswith(f"runs/{run}/checkpoint"), f"{f}: not run {run}"
        M = rj(T + met)
        assert M["optimizer_steps"] == 4620 == len(M["step_seconds"]) and abs(sum(M["step_seconds"]) - M["wall_seconds"]) < 1, f"{met}: step times incomplete"
        TS[k] = M["step_seconds"]
        R = rj(T + f); st = R["steps"]
        A[k] = [(e["step"], e["acc"]) for e in st]
        assert all("equal=True" in c for c in R["param_checks"]), f"{f}: data-parallel weight check failed"
        pts = [(e["step"], e["loss"]) for e in st]
        assert pts[0][0] == 10 and pts[-1][0] == 4620 and all(b[0] > a[0] for a, b in zip(pts, pts[1:])), f"{f}: incomplete log"
        L[k] = pts
        put(f"v3_loss_{k}_start", f"{np.mean([y for s_, y in pts if s_ <= 300]):.2f}", T + f, "steps[].loss, steps 10-300", "training loss, first 300 steps")
        put(f"v3_loss_{k}_end", f"{np.mean([y for s_, y in pts if s_ > 4320]):.2f}", T + f, "steps[].loss, last 300 steps", "training loss, last 300 steps")
        put(f"v3_dpchecks_{k}", n(len(R["param_checks"])), T + f, "param_checks", "data-parallel weight-equality checks, all equal")
    assert len(rj(T + "trainlog_v3f-kev-27b-dp.json")["param_checks"]) == len(rj(T + "trainlog_v3f-kev-9b-dp.json")["param_checks"])
    groups = [("IU", ("iu_",)), ("CT-RATE", ("ctrate_",)), ("ReXErr", ("rexerr",)), ("Eurorad diagnosis", ("eurorad_dx",)),
              ("Eurorad subspecialty", ("eurorad_route",)), ("RadCases", ("radcases_",)), ("MedMCQA", ("medmcqa_",)), ("MedQA", ("medqa",)),
              ("MMLU", ("mmlu_",)), ("PubMedQA", ("pubmedqa",)), ("MedXpertQA", ("medxpertqa",)), ("Teacher-labeled", ("teacher_",))]
    F = {"b27": "kev-27b-dp_dev_stock.json", "a27": "dev_cal_27b.json", "b9": "kev-9b-dp_dev_stock.json", "a9": "dev_cal_9b.json"}
    D = {k: rj(T + f)["tasks"] for k, f in F.items()}
    def acc(Tk, pre):
        ts = [v for t, v in Tk.items() if t.startswith(pre)]
        return sum(v["n"] * v["acc"] for v in ts) / sum(v["n"] for v in ts)
    assert all(any(t.startswith(p) for p in sum((g[1] for g in groups), ())) for t in D["a27"]), "a development task is not in any group"
    rows = [(lab, acc(D["b27"], pre), acc(D["a27"], pre), acc(D["b9"], pre), acc(D["a9"], pre)) for lab, pre in groups]
    for k, f in F.items():
        put(f"v3_dev_{k}", pct(rj(T + f)["overall"]["acc"]), T + f, "overall.acc", "development accuracy (%)")
    import figures
    figures.fig_training_v3(L, A, TS, rows, HERE / "figures")


FIGDATA = "artifacts/figure_data/artifacts/figure_data.json"   # jobs/figure_data.py: kev.train logs (loss every 10 steps)


def build_fig_training():
    """Figure: training loss (kev.train logs) and development accuracy before and after fine-tuning (dev_stock vs dev_cal)."""
    F = rj(FIGDATA)["logs"]
    names = {"rk27": "v2mg-kev-27b.log", "rk9": "v2x9-kev-9b.log", "base9": "v2x9-base-9b.log"}
    L = {}
    for k, f in names.items():
        pts = [(p[0], p[1]) for p in F[f]["points"] if p[0] is not None]
        assert pts[-1][0] >= 8500 and all(b[0] > a[0] for a, b in zip(pts, pts[1:])), f"{f}: incomplete or unordered loss log"
        L[k] = pts
    import numpy as np
    def win(k, lo, hi): return float(np.mean([y for s, y in L[k] if lo <= s <= hi]))
    for k in L:
        put(f"loss_{k}_start", f"{win(k, 1, 200):.2f}", FIGDATA, f"logs.{names[k]} mean of steps 1-200", "training loss, first 200 steps")
        put(f"loss_{k}_end", f"{win(k, 8000, 8521):.2f}", FIGDATA, f"logs.{names[k]} mean of steps 8000-8521", "training loss, last 521 steps")
    assert win("base9", 1, 200) > win("rk9", 1, 200), "claim: base initialization starts at a higher loss"
    groups = [("IU", ("iu_",)), ("CT-RATE", ("ctrate_",)), ("Eurorad diagnosis", ("eurorad_dx",)), ("Eurorad subspecialty", ("eurorad_route",)),
              ("MedMCQA", ("medmcqa_",)), ("MedQA", ("medqa",)), ("MMLU", ("mmlu_",)), ("PubMedQA", ("pubmedqa",)),
              ("MedXpertQA", ("medxpertqa",)), ("Teacher-labeled", ("teacher_",))]
    D = {k: rj(f"results/train/{p}")["tasks"] for k, p in (("b27", "v2mg/kev-27b_dev_stock.json"), ("a27", "v2mg/kev-27b_dev_cal.json"),
                                                          ("b9", "v2x9/kev-9b_dev_stock.json"), ("a9", "v2x9/kev-9b_dev_cal.json"))}
    def acc(T, pre):
        ts = [v for t, v in T.items() if t.startswith(pre)]
        return sum(v["n"] * v["acc"] for v in ts) / sum(v["n"] for v in ts)
    rows = [(lab, acc(D["b27"], pre), acc(D["a27"], pre), acc(D["b9"], pre), acc(D["a9"], pre)) for lab, pre in groups]
    for k, f in (("dev_b27", "v2mg/kev-27b_dev_stock.json"), ("dev_a27", "v2mg/kev-27b_dev_cal.json"), ("dev_b9", "v2x9/kev-9b_dev_stock.json"), ("dev_a9", "v2x9/kev-9b_dev_cal.json")):
        put(k, pct(rj(f"results/train/{f}")["overall"]["acc"]), f"results/train/{f}", "overall.acc", "development accuracy (%)")
    import figures
    figures.fig_training(L, rows, HERE / "figures")


FIGTEACH = "artifacts/figure_data_teacher/artifacts/figure_data.json"   # jobs/figure_data.py --teacher-only (Eurorad order questions)
TEACH_EX = [("agreed_pool", 32379, "eurorad/17094", "Agreement"), ("disagreed_pool", 31031, "eurorad/18086", "Disagreement")]


def build_fig_teacher():
    T = rj(FIGTEACH)["teacher"]
    ex = []
    for pool, tid, group, title in TEACH_EX:
        c = next(x for x in T[pool] if x["tid"] == tid)
        assert c["group"] == group and c["question"]["type"] == "choice"
        st = c["state"] if isinstance(c["state"], str) else " ".join(c["state"].values())
        st = re.sub(r"^(Indication|clinical question):\s*", "", st)
        opts = list(c["question"]["criteria"].items())
        pm, pq = c["medgemma"], c["qwen38"]
        kept = max(pm, key=pm.get) == max(pq, key=pq.get)
        assert kept == (pool == "agreed_pool")
        ex.append((title, st, c["question"]["instructions"], [(k, v.replace(" (radiograph or ultrasound)", "")) for k, v in opts], pm, pq, kept))
        put(f"tex_{title.lower()}_case", group.split("/")[1], FIGTEACH, f"{pool}[tid={tid}].group", "Eurorad case number shown in the teacher figure")
    import figures
    figures.fig_teacher(ex, HERE / "figures")


QWEN27_CFG = {"num_hidden_layers": 64, "linear": 48, "full": 16, "hidden_size": 5120}   # Qwen/Qwen3.8-27B config.json (text_config), rev 1d4bf0f


def build_fig_kev():
    V = {x["key"]: x["value"] for x in LEDGER}
    put("bb_layers", str(QWEN27_CFG["num_hidden_layers"]), "Qwen/Qwen3.8-27B config.json", "text_config.num_hidden_layers", "backbone layers")
    put("bb_linear", str(QWEN27_CFG["linear"]), "Qwen/Qwen3.8-27B config.json", "layer_types == linear_attention", "Gated DeltaNet layers")
    put("bb_full", str(QWEN27_CFG["full"]), "Qwen/Qwen3.8-27B config.json", "layer_types == full_attention", "full-attention layers")
    put("bb_hidden", n(QWEN27_CFG["hidden_size"]), "Qwen/Qwen3.8-27B config.json", "text_config.hidden_size", "hidden size")
    assert QWEN27_CFG["linear"] + QWEN27_CFG["full"] == QWEN27_CFG["num_hidden_layers"] and QWEN27_CFG["linear"] == 3 * QWEN27_CFG["full"]
    V = {x["key"]: x["value"] for x in LEDGER}
    import figures
    figures.fig_kev(V, HERE / "figures")


RB_PRIMARY = "results_book/tables/primary.csv"
RB_PERTASK = "results_book/tables/per_task.csv"
FAMILY_LABEL = {"report_cxr_human": "Chest radiograph report reading", "case_diagnosis": "Case diagnosis", "routing": "Subspecialty classification",
                "radiology_knowledge": "Radiology knowledge", "medical_knowledge": "Medical knowledge", "report_ct": "CT report reading",
                "report_cxr": "Chest radiograph finding status", "orders_protocols": "Imaging orders", "triage_followup": "Triage and follow-up"}
HUMAN_FAMS = ["report_cxr_human", "case_diagnosis", "routing", "radiology_knowledge", "medical_knowledge"]
MACHINE_FAMS = ["report_ct", "report_cxr", "orders_protocols", "triage_followup"]


def build_results_primary():
    """Results 3.1: the prespecified comparison (demonstration sample excluded) by family, overall and on human-key tasks."""
    P = {r["family"]: r for r in csv.DictReader(open(inp(RB_PRIMARY)))}
    f = lambda r, k: float(r[k])
    for key, fam in (("p_all", "overall_task_mean"), ("p_hk", "human_keys_task_mean"), ("p_rhk", "radiology_human_keys_task_mean")):
        r = P[fam]
        assert f(r, "lo_nodemo") > 0, f"{key}: CI no longer excludes zero"
        put(key, pct(f(r, "diff_nodemo")), RB_PRIMARY, f"{fam}.diff_nodemo", "prespecified difference (pp)")
        put(key + "_ci", ci(f(r, "lo_nodemo"), f(r, "hi_nodemo")), RB_PRIMARY, f"{fam}.lo/hi_nodemo", "95% CI")
        put(key + "_full", pct(f(r, "diff")), RB_PRIMARY, f"{fam}.diff", "full test set (pp)")
        put(key + "_full_ci", ci(f(r, "lo"), f(r, "hi")), RB_PRIMARY, f"{fam}.lo/hi", "95% CI")
        put(key + "_k27", pct(f(r, "kev27")), RB_PRIMARY, f"{fam}.kev27", "Kev-27B, full test (%)")
        put(key + "_rk27", pct(f(r, "radkev27")), RB_PRIMARY, f"{fam}.radkev27", "RadKev-27B, full test (%)")
    rows = [("summary", lab, 0, 100 * f(P[k], "diff_nodemo"), 100 * f(P[k], "lo_nodemo"), 100 * f(P[k], "hi_nodemo"))
            for lab, k in (("All 29 tasks (primary outcome)", "overall_task_mean"), ("16 human-key tasks", "human_keys_task_mean"))]
    for g, fams in (("human", HUMAN_FAMS), ("machine", MACHINE_FAMS)):
        for fam in fams:
            r = P[fam]; d, lo, hi = 100 * f(r, "diff_nodemo"), 100 * f(r, "lo_nodemo"), 100 * f(r, "hi_nodemo")
            rows.append((g, FAMILY_LABEL[fam], int(r["n_nodemo"]), d, lo, hi))
            put(f"pf_{fam}", (pct(f(r, "diff_nodemo")) if d <= 0 else "+" + pct(f(r, "diff_nodemo"))), RB_PRIMARY, f"{fam}.diff_nodemo", "family difference (pp)")
            put(f"pf_{fam}_ci", ci(f(r, "lo_nodemo"), f(r, "hi_nodemo")), RB_PRIMARY, f"{fam}.lo/hi_nodemo", "95% CI")
            put(f"pf_{fam}_n", n(int(r["n_nodemo"])), RB_PRIMARY, f"{fam}.n_nodemo", "questions")
            put(f"pf_{fam}_holm", f"{float(r['p_holm_nodemo']):.2f}", RB_PRIMARY, f"{fam}.p_holm_nodemo", "Holm-adjusted p")
    improved = [fam for fam in HUMAN_FAMS + MACHINE_FAMS if P[fam]["improves"] == "yes"]
    assert improved == ["report_cxr_human", "case_diagnosis", "medical_knowledge", "report_ct", "orders_protocols", "triage_followup"], improved
    assert all(float(P[fam]["p_holm_nodemo"]) < 0.05 for fam in improved)
    # share of the task-averaged gain by answer-key group (full test, per-task differences)
    T = list(csv.DictReader(open(inp(RB_PERTASK))))
    dd = {r["task"]: float(r["v2_27_acc"]) - float(r["stock27_acc"]) for r in T}
    assert len(dd) == 29
    grp = lambda t: "teacher" if t.startswith("teacher_") else "ctrate" if t.startswith("ctrate_") else "human"
    tot = sum(dd.values())
    for g in ("teacher", "human"):
        sg = sum(v for t, v in dd.items() if grp(t) == g)
        put(f"share_{g}", f"{100 * sg / tot:.0f}", RB_PERTASK, f"sum of {g} task differences / sum of all", "% of task-averaged gain")
        put(f"share_{g}_pp", pct(sg / len(dd)), RB_PERTASK, f"{g} contribution to the task mean", "pp")
    put("n_teacher_tasks", str(sum(1 for t in dd if grp(t) == "teacher")), RB_PERTASK, "teacher_ tasks", "teacher-labeled tasks")
    # forest table: full test set for every column (per-family accuracies exist only for the full test set)
    fr = [{"kind": "header", "label": "Task-averaged accuracy"}]
    for lab, k in (("All 29 tasks (primary outcome)", "overall_task_mean"), ("16 human-key tasks", "human_keys_task_mean")):
        r = P[k]; fr.append({"kind": "summary", "label": lab, "n": int(r["n"]), "a": 100 * f(r, "kev27"), "b": 100 * f(r, "radkev27"),
                             "d": 100 * f(r, "diff"), "lo": 100 * f(r, "lo"), "hi": 100 * f(r, "hi")})
    for g, head, fams in (("human", "Human answer keys", HUMAN_FAMS), ("machine", "Machine-derived answer keys", MACHINE_FAMS)):
        fr.append({"kind": "header", "label": head})
        for j, fam in enumerate(fams):
            r = P[fam]; fr.append({"kind": g, "label": FAMILY_LABEL[fam], "n": int(r["n"]), "a": 100 * f(r, "kev27"), "b": 100 * f(r, "radkev27"),
                                   "d": 100 * f(r, "diff"), "lo": 100 * f(r, "lo"), "hi": 100 * f(r, "hi"), "band": j % 2 == 0})
    for fam in HUMAN_FAMS + MACHINE_FAMS:
        r = P[fam]; d = f(r, "diff")
        put(f"pff_{fam}", ("+" if d > 0 else "") + pct(d), RB_PRIMARY, f"{fam}.diff", "family difference, full test (pp)")
        put(f"pff_{fam}_ci", ci(f(r, "lo"), f(r, "hi")), RB_PRIMARY, f"{fam}.lo/hi", "95% CI, full test")
        put(f"pff_{fam}_n", n(int(r["n"])), RB_PRIMARY, f"{fam}.n", "questions, full test")
    full_improves = [fam for fam in HUMAN_FAMS + MACHINE_FAMS if f(P[fam], "lo") > 0]
    assert full_improves == improved, "the full test set must lead to the same family conclusions"
    delta = max(abs(f(P[k], "diff") - f(P[k], "diff_nodemo")) for k in HUMAN_FAMS + MACHINE_FAMS + ["overall_task_mean", "human_keys_task_mean"])
    put("demo_maxdelta", pct(delta), RB_PRIMARY, "max |diff - diff_nodemo|", "largest change from excluding the demonstration sample (pp)")
    SHORT = {"report_cxr_human": "Radiograph\nreports", "case_diagnosis": "Case\ndiagnosis", "routing": "Sub-\nspecialty",
             "radiology_knowledge": "Radiology\nknowledge", "medical_knowledge": "Medical\nknowledge", "report_ct": "CT\nreports",
             "report_cxr": "Finding\nstatus", "orders_protocols": "Imaging\norders", "triage_followup": "Triage,\nfollow-up"}
    item = lambda lab, r, nn: (lab, nn, 100 * f(r, "kev27"), 100 * f(r, "radkev27"), 100 * f(r, "diff"))
    panels = [("Average over tasks", [item("All\ntasks", P["overall_task_mean"], 0),
                                      item("Human-\nlabeled", P["human_keys_task_mean"], 0)], False),
              ("Human-labeled families", [item(SHORT[k], P[k], int(P[k]["n"])) for k in HUMAN_FAMS], False),
              ("Model-labeled families", [item(SHORT[k], P[k], int(P[k]["n"])) for k in MACHINE_FAMS], False)]
    import figures
    figures.fig_bars(panels, HERE / "figures")


RB_ACC = "results_book/tables/accuracy.csv"; RB_RLEV = "results_book/tables/reasoning_levels.csv"; RB_RPAIR = "results_book/tables/reasoning_pairs.csv"
RSN = "artifacts/llm_reasoning/artifacts/llm_reasoning.json"


def build_results_llm():
    """Results 3.2: decision models and language models (accuracy, reasoning, latency) and Figure 5."""
    import figures
    from figures import C_RAD, C_KEV, C_DMO, C_LLM, C_LLMR
    A = {r["system"]: r for r in csv.DictReader(open(inp(RB_ACC)))}
    R = rj(RB)
    mg = "medgemma_fix" if "medgemma_fix" in A else "medgemma_brief"
    order = ["v2_27", "qwen38", "r9", "stock27", "stock9", mg, "kev4", "kev08", "gliner_decide", "laya_typed", "julia1", "laya"]
    names = {"v2_27": "RadKev-27B", "r9": "RadKev-9B", "stock27": "Kev-27B", "stock9": "Kev-9B", "kev4": "Kev-4B", "kev08": "Kev-0.8B",
             "qwen38": "Qwen3.8-27B", mg: "MedGemma-27B-text", "gliner_decide": "GLiNER2.5-Decide", "laya_typed": "Laya-typed",
             "laya": "Laya", "julia1": "Julia-1"}
    color = lambda k: C_RAD if k in ("v2_27", "r9") else C_KEV if k in ("stock27", "stock9", "kev4", "kev08") else C_LLM if k in ("qwen38", mg) else C_DMO
    acc = {k: 100 * float(A[k]["human_keys_micro"]) for k in order}
    order.sort(key=lambda k: -acc[k])
    assert order[0] == "v2_27"
    acc_rows = [(names[k], acc[k], color(k), k == "v2_27") for k in order]
    for k in order:
        put(f"ha_{k}", f"{acc[k]:.1f}", RB_ACC, f"{k}.human_keys_micro", "accuracy on human-labeled questions (%)")
    put("ha_others_max", f"{max(acc[k] for k in ('gliner_decide', 'laya_typed', 'laya', 'julia1')):.1f}", RB_ACC, "max of the four other decision models", "%")
    for key, k in (("hd_rk27_qwen", "d_v2_27_qwen38_human_keys"), ("hd_rk27_mg", f"d_v2_27_{mg}_human_keys"), ("hd_rk9_k27", "d_r9_stock27_human_keys"),
                   ("hd_qwen_k27", "d_qwen38_stock27_human_keys")):
        assert R[k + "_lo"] > 0, key
        put(key, pct(R[k]), RB, k, "difference on human-labeled questions (pp)"); put(key + "_ci", ci(R[k + "_lo"], R[k + "_hi"]), RB, k, "95% CI")
    # reasoning sample
    L = {r["system"]: r for r in csv.DictReader(open(inp(RB_RLEV)))}
    PR = {r["pair"]: r for r in csv.DictReader(open(inp(RB_RPAIR)))}
    for k in ("v2_27", "stock27", "qwen38", "qwen38_think", mg, "medgemma_think"):
        kk = "medgemma_fix" if k == mg else k
        put(f"rs_{k}", pct(float(L[kk]["micro"])), RB_RLEV, f"{kk}.micro", "accuracy on the reasoning sample (%)")
        put(f"rs_{k}_ece", f"{float(L[kk]['ece']):.3f}", RB_RLEV, f"{kk}.ece", "ECE on the reasoning sample")
        put(f"rs_{k}_cov", pct(float(L[kk]["cov5"])), RB_RLEV, f"{kk}.cov5", "coverage at 5% error (%)")
    for key, p in (("rp_qthink", "qwen38_think-qwen38"), ("rp_mthink", "medgemma_think-medgemma_fix"), ("rp_rk_qthink", "v2_27-qwen38_think"),
                   ("rp_rk_mthink", "v2_27-medgemma_think"), ("rp_rk_q", "v2_27-qwen38")):
        r = PR[p]
        put(key, pct(float(r["micro"])), RB_RPAIR, f"{p}.micro", "difference on the reasoning sample (pp)")
        put(key + "_ci", ci(float(r["lo"]), float(r["hi"])), RB_RPAIR, f"{p}.lo/hi", "95% CI")
        put(key + "_tm", pct(float(r["macro"])), RB_RPAIR, f"{p}.macro", "task-mean difference (pp)")
        put(key + "_tm_ci", ci(float(r["macro_lo"]), float(r["macro_hi"])), RB_RPAIR, f"{p}.macro_lo/hi", "95% CI")
    assert float(PR["v2_27-qwen38_think"]["lo"]) < 0 < float(PR["v2_27-qwen38_think"]["hi"]), "claim: RadKev-27B and Qwen with reasoning do not differ (questions)"
    assert float(PR["v2_27-qwen38_think"]["macro_hi"]) < 0, "claim: Qwen with reasoning is higher on the task mean"
    G = rj(RSN)["generation"]
    put("rs_q_cap", f"{100 * G['qwen38_think']['overall']['hit_cap_share']:.1f}", RSN, "generation.qwen38_think.overall.hit_cap_share", "% of outputs at the token limit")
    put("rs_q_medtok", n(G["qwen38_think"]["overall"]["median_new_tokens"]), RSN, "generation.qwen38_think.overall.median_new_tokens", "median reasoning tokens")
    put("rs_m_medtok", n(G["medgemma_think"]["overall"]["median_new_tokens"]), RSN, "generation.medgemma_think.overall.median_new_tokens", "median reasoning tokens")
    reason_rows = [("RadKev-27B", 100 * float(L["v2_27"]["micro"]), C_RAD, True), ("Kev-27B", 100 * float(L["stock27"]["micro"]), C_KEV, False),
                   ("Qwen3.8-27B", 100 * float(L["qwen38"]["micro"]), C_LLM, False), ("+ reasoning", 100 * float(L["qwen38_think"]["micro"]), C_LLMR, False),
                   ("MedGemma-27B-text", 100 * float(L["medgemma_fix"]["micro"]), C_LLM, False), ("+ reasoning", 100 * float(L["medgemma_think"]["micro"]), C_LLMR, False)]
    # latency (latency_bench, latency_9b): per question
    L27 = rj(LAT27); L9 = rj(LAT9)
    q = L27["models"]["qwen38"]
    lat = [("RadKev-9B", L9["models"]["radkev9"]["per_question_ms"]["median"], C_RAD, False),
           ("RadKev-27B", L27["models"]["radkev27"]["per_question_ms"]["median"], C_RAD, True),
           ("Qwen3.8-27B", q["letter"]["per_question_ms"]["median"], C_LLM, False),
           ("generated letter", q["direct"]["per_question_ms"]["median"], C_LLM, False),
           ("+ reasoning", q["reasoning"]["per_question_ms"]["median"], C_LLMR, False)]
    put("lt_rk9", f"{lat[0][1]:.0f}", LAT9, "radkev9.per_question_ms.median", "ms"); put("lt_rk27", f"{lat[1][1]:.0f}", LAT27, "radkev27", "ms")
    put("lt_q", f"{lat[2][1]:.0f}", LAT27, "qwen38.letter", "ms"); put("lt_qgen", f"{lat[3][1]:.0f}", LAT27, "qwen38.direct", "ms")
    put("lt_qthink", f"{lat[4][1] / 1000:.1f}", LAT27, "qwen38.reasoning", "s"); put("lt_qthink_n", str(q["reasoning"]["per_question_ms"]["n"]), LAT27, "qwen38.reasoning.n", "questions timed")
    put("lt_ratio", f"{lat[4][1] / lat[1][1]:.0f}", LAT27, "reasoning / RadKev-27B median", "x")
    put("thr_rk27_hr", n(int(round(3600 * 1000 / lat[1][1], -3))), LAT27, "3.6e6 / RadKev-27B median ms", "questions per hour, one request at a time")
    put("thr_rk9_hr", n(int(round(3600 * 1000 / lat[0][1], -3))), LAT9, "3.6e6 / RadKev-9B median ms", "questions per hour, one request at a time")
    figures.fig_llm(acc_rows, reason_rows, lat, HERE / "figures")


SPLIT = "artifacts/subset_split/artifacts/subset_split.json"   # jobs/subset_split.py: radiology vs examination human-labeled questions


def build_subset_split():
    """Results 3.2: the human-labeled comparison split into radiology and examination questions (same bootstrap as the book)."""
    S, R = rj(SPLIT), rj(RB)
    assert abs(S["pairs"]["v2_27-qwen38"]["radiology_human_keys"]["micro"] - R["d_v2_27_qwen38_radiology_human_keys"]) < 1e-9, "split must reproduce the book"
    put("ss_n_rad", n(S["n"]["radiology_human_keys"]), SPLIT, "n.radiology_human_keys", "human-labeled radiology questions")
    put("ss_n_ex", n(S["n"]["exam_human_keys"]), SPLIT, "n.exam_human_keys", "other human-labeled questions")
    for m, k in (("v2_27", "rk27"), ("qwen38", "q"), ("stock27", "k27"), ("r9", "rk9")):
        for s_, t in (("radiology_human_keys", "rad"), ("exam_human_keys", "ex")):
            put(f"ss_{t}_{k}", pct(S["acc"][m][s_]["micro"]), SPLIT, f"acc.{m}.{s_}.micro", "accuracy (%)")
    for pair, k in (("v2_27-qwen38", "rk27_q"), ("r9-qwen38", "rk9_q"), ("v2_27-stock27", "rk27_k27")):
        for s_, t in (("radiology_human_keys", "rad"), ("exam_human_keys", "ex")):
            x = S["pairs"][pair][s_]
            put(f"ss_{t}_{k}", pct(x["micro"]), SPLIT, f"pairs.{pair}.{s_}.micro", "difference (pp)")
            put(f"ss_{t}_{k}_ci", ci(*x["micro_ci"]), SPLIT, f"pairs.{pair}.{s_}.micro_ci", "95% CI")
    assert S["pairs"]["v2_27-qwen38"]["exam_human_keys"]["micro"] > 4 * S["pairs"]["v2_27-qwen38"]["radiology_human_keys"]["micro"], "claim: the lead over Qwen comes mainly from examination questions"
    T = {r["task"]: r for r in csv.DictReader(open(inp(RB_PERTASK)))}
    iu = sum(int(T[t]["n"]) for t in T if t.startswith("iu_"))
    assert sum(int(T[t]["n"]) for t in S["tasks"]["radiology_human_keys"]) == S["n"]["radiology_human_keys"]
    put("ss_n_iu", n(iu), RB_PERTASK, "sum of n over iu_ tasks", "IU/Open-i test questions")
    put("ss_iu_share", f"{100 * iu / S['n']['radiology_human_keys']:.0f}", RB_PERTASK, "IU / radiology human-labeled", "%")


EXT = "artifacts/external_tests/analysis.json"          # paper/external/analyse.py on the rows of jobs/external_tests.py
EXT_RC = "artifacts/external_tests/artifacts/external_tests.json"     # RadCases build manifest (job caf510cb)
EXT_RG = "artifacts/radgraph_xl_build/artifacts/radgraph_xl_build.json"
EXT_NAMES = {"v2_27": "RadKev-27B", "stock27": "Kev-27B", "r9": "RadKev-9B", "stock9": "Kev-9B", "qwen38": "Qwen3.8-27B", "medgemma_fix": "MedGemma-27B-text"}


def build_external():
    """Results: external tests (analysis-plan addendum 2026-10-05), Supplementary Note S6 and Table S-external."""
    A = rj(EXT)
    rc = rj(EXT_RC)["radcases_build"]
    put("ext_rc_matched", n(rc["stats"]["matched_cases"]), EXT_RC, "radcases_build.stats.matched_cases", "RadCases cases with rebuilt text")
    put("ext_rc_labels", n(rc["stats"]["label_rows"]), EXT_RC, "radcases_build.stats.label_rows", "label rows in the two subsets")
    put("ext_rc_overlap", n(rc["stats"].get("overlap_with_radkev_splits", 0)), EXT_RC, "radcases_build.stats.overlap_with_radkev_splits", "dropped: verbatim in RadKev splits")
    put("ext_rc_multi", n(rc["stats"].get("multi_panel_excluded", 0)), EXT_RC, "radcases_build.stats.multi_panel_excluded", "dropped from panel question: several panels")
    put("ext_rc_ntopics", n(rc["n_topics"]), EXT_RC, "radcases_build.n_topics", "ACR AC topics (options)")
    put("ext_rc_npanels", n(rc["n_panels"]), EXT_RC, "radcases_build.n_panels", "panel options incl. None")
    def block(name, tag, subsets):
        R = A[name]
        for s_ in subsets:
            put(f"ext_{tag}_n_{s_}", n(R["n"][s_]), EXT, f"{name}.n.{s_}", "questions")
        for m, v in R["models"].items():
            for s_ in subsets:
                put(f"ext_{tag}_{m}_{s_}", pct(v[s_]["acc"]), EXT, f"{name}.models.{m}.{s_}.acc", "accuracy (%)")
                put(f"ext_{tag}_{m}_{s_}_ci", ci(*v[s_]["acc_ci"]), EXT, f"{name}.models.{m}.{s_}.acc_ci", "95% CI")
            put(f"ext_{tag}_{m}_ece", f"{v['all']['ece']:.3f}", EXT, f"{name}.models.{m}.all.ece", "ECE")
        for pr, v in R["pairs"].items():
            k = pr.replace("-", "_")
            for s_ in subsets:
                put(f"ext_{tag}_d_{k}_{s_}", pct(v[s_]["d"]), EXT, f"{name}.pairs.{pr}.{s_}.d", "difference (pp)")
                put(f"ext_{tag}_d_{k}_{s_}_ci", ci(*v[s_]["ci"]), EXT, f"{name}.pairs.{pr}.{s_}.ci", "95% CI")
        return R
    P = block("radcases_panel", "rcp", ["all", "synthetic", "medbullets", "excl_none", "none_only"])
    T = block("radcases_topic", "rct", ["all", "synthetic", "medbullets"])
    have_rg = "radgraph_xl" in A
    if have_rg:
        G = rj(EXT_RG)
        put("ext_rg_reports", n(G["reports"]), EXT_RG, "reports", "Stanford RadGraph-XL reports"); put("ext_rg_excl", n(G["excluded"]), EXT_RG, "excluded", "overlap exclusions")
        put("ext_rg_excl_cxr", n(G["overlap_chexpert_plus"].get("cxr", 0)), EXT_RG, "overlap_chexpert_plus.cxr", "CXR reports overlapping CheXpert Plus")
        assert set(G["overlap_chexpert_plus"]) | set(G["overlap_radkev"]) <= {"cxr"}, "claim: all exclusions are chest radiograph reports"
        put("ext_rg_records", n(G["stats"]["records"]), EXT_RG, "stats.records", "reports with at least one question")
        mods = [m for m in ("chestct", "abdct", "brainmr", "cxr") if m in A["radgraph_xl"]["n"]]
        Rg = block("radgraph_xl", "rg", ["all"] + mods)
    PP = lambda f, pr, s_: A[f]["pairs"][pr][s_]["ci"]
    assert PP("radcases_panel", "v2_27-stock27", "all")[1] < 0 < PP("radcases_panel", "v2_27-stock27", "excl_none")[0], "claim: panel lower overall, higher excluding None"
    assert PP("radcases_panel", "v2_27-qwen38", "all")[0] > 0 and PP("radcases_panel", "v2_27-medgemma_fix", "all")[0] > 0, "claim: RadKev-27B above both LLMs (panel)"
    assert PP("radcases_topic", "v2_27-stock27", "all")[0] < 0 < PP("radcases_topic", "v2_27-stock27", "all")[1], "claim: topic does not differ"
    if have_rg:
        assert PP("radgraph_xl", "v2_27-stock27", "all")[0] < 0 < PP("radgraph_xl", "v2_27-stock27", "all")[1], "claim: no change at 27B (RadGraph-XL)"
        assert PP("radgraph_xl", "r9-stock9", "all")[0] > 0 and PP("radgraph_xl", "v2_27-qwen38", "all")[1] < 0, "claims: 9B gain; below Qwen"
    # Supplementary table: accuracy (%) of every system on every external test
    cols = [("radcases_panel", "all"), ("radcases_panel", "excl_none"), ("radcases_topic", "all")]
    head = r"System & \multicolumn{2}{c}{RadCases panel} & RadCases topic"
    sub = r" & All & Excluding None & (225 options)"
    if have_rg:
        cols += [("radgraph_xl", "all")] + [("radgraph_xl", m) for m in ("chestct", "abdct", "brainmr")]
        head += r" & \multicolumn{4}{c}{RadGraph-XL status}"; sub += r" & All & Chest CT & Abd./pelvis CT & Brain MRI"
    rows = []
    for m in ("v2_27", "stock27", "r9", "stock9", "qwen38", "medgemma_fix"):
        cells = []
        for f, s_ in cols:
            v = A[f]["models"].get(m)
            cells.append(pct(v[s_]["acc"]) if v else "--")
        rows.append(f"{EXT_NAMES[m]} & " + " & ".join(cells))
    rows.insert(0, "Questions" + "".join(f" & {n(A[f]['n'][s_])}" for f, s_ in cols))
    _tab("supp_external", "@{}l" + "r" * len(cols) + "@{}", head + r" \\" + sub, rows, midrules=(1, 5))


def build_results_spec():
    """Results 3.3: specialization vs scale and initialization (results book), Figure 6."""
    R = rj(RB)
    P = lambda k: 100 * R[k]
    def dif(key, k, sign=None, scale=100):
        lo, hi = R[k + "_lo"], R[k + "_hi"]
        if sign is not None: assert (lo > 0) if sign > 0 else (hi < 0), f"{key}: CI no longer excludes zero"
        put(key, pct(R[k]), RB, k, "difference (pp)"); put(key + "_ci", ci(lo, hi), RB, k, "95% CI")
    for key, k, sg in (("ss_spec27", "d_v2_27_stock27_human_keys", 1), ("ss_spec9", "d_r9_stock9_human_keys", 1), ("ss_scale", "d_stock27_stock9_human_keys", 1),
                       ("ss_did27", "did_specialisation27_minus_scale_hk", 1), ("ss_did9", "did_specialisation9_minus_scale_hk", 1),
                       ("ss_rspec27", "d_v2_27_stock27_radiology_human_keys", 1), ("ss_rspec9", "d_r9_stock9_radiology_human_keys", 1),
                       ("ss_rscale", "d_stock27_stock9_radiology_human_keys", -1), ("ss_r9k27", "d_r9_stock27_human_keys", 1),
                       ("ss_r9k27_rad", "d_r9_stock27_radiology_human_keys", 1),
                       ("ss_tm_spec27", "dm_v2_27_stock27_human_keys", 1), ("ss_tm_scale", "dm_stock27_stock9_human_keys", 1),
                       ("in_full", "d_r9_b9_human_keys", 1), ("in_f10", "d_r9f10_b9f10_human_keys", 1), ("in_did", "did_init_full_minus_init_10pct_hk", 1),
                       ("tr_k", "tr_r9_stock9", None), ("tr_b", "tr_b9_stock9", -1), ("tr_k10", "tr_r9f10_stock9", None), ("tr_b10", "tr_b9f10_stock9", -1),
                       ("tr_27", "tr_v2_27_stock27", None)):
        dif(key, k, sg)
    assert R["dm_stock27_stock9_human_keys"] > R["dm_v2_27_stock27_human_keys"], "claim: scale exceeds specialization as a task mean"
    for m in ("stock9", "r9", "stock27", "v2_27", "b9", "r9f10", "b9f10"):
        put(f"sa_{m}", f"{P(f'acc_{m}_human_keys'):.1f}", RB, f"acc_{m}_human_keys", "accuracy, human-labeled (%)")
    put("sa_b9_change", pct(R["acc_b9_human_keys"] - R["acc_b9f10_human_keys"]), RB, "acc_b9 - acc_b9f10", "base init, 10% -> all data (pp)")
    put("sa_r9_change", pct(R["acc_r9_human_keys"] - R["acc_r9f10_human_keys"]), RB, "acc_r9 - acc_r9f10", "Kev init, 10% -> all data (pp)")
    pa = [("9B models", P("acc_stock9_human_keys"), P("acc_r9_human_keys"), P("d_r9_stock9_human_keys")),
          ("27B models", P("acc_stock27_human_keys"), P("acc_v2_27_human_keys"), P("d_v2_27_stock27_human_keys"))]
    pb = [("9B models", P("acc_stock9_radiology_human_keys"), P("acc_r9_radiology_human_keys"), P("d_r9_stock9_radiology_human_keys")),
          ("27B models", P("acc_stock27_radiology_human_keys"), P("acc_v2_27_radiology_human_keys"), P("d_v2_27_stock27_radiology_human_keys"))]
    pc = [("10% of training data", P("acc_b9f10_human_keys"), P("acc_r9f10_human_keys"), P("d_r9f10_b9f10_human_keys")),
          ("All training data", P("acc_b9_human_keys"), P("acc_r9_human_keys"), P("d_r9_b9_human_keys"))]
    from figures import WP_ACC
    pd = [("10% of training data", [(P("tr_b9f10_stock9"), P("tr_b9f10_stock9_lo"), P("tr_b9f10_stock9_hi"), "#E3B98F"),
                            (P("tr_r9f10_stock9"), P("tr_r9f10_stock9_lo"), P("tr_r9f10_stock9_hi"), WP_ACC)]),
          ("All training data", [(P("tr_b9_stock9"), P("tr_b9_stock9_lo"), P("tr_b9_stock9_hi"), "#E3B98F"),
                         (P("tr_r9_stock9"), P("tr_r9_stock9_lo"), P("tr_r9_stock9_hi"), WP_ACC)])]
    import figures
    figures.fig_spec(pa, pb, pc, pd, HERE / "figures")


RB_CAL = "results_book/tables/calibration_human_keys.csv"; RB_CALP = "results_book/tables/calibration_pairs.csv"; RB_RECAL = "results_book/tables/recal.csv"


def build_results_calib():
    """Results 3.4: calibration and selective prediction on human-labeled questions, Figure 7."""
    C = {r["system"]: r for r in csv.DictReader(open(inp(RB_CAL)))}
    CP = {(r["a"], r["b"], r["scope"]): r for r in csv.DictReader(open(inp(RB_CALP)))}
    RC = {r["system"]: r for r in csv.DictReader(open(inp(RB_RECAL)))}
    mg = "medgemma_fix" if "medgemma_fix" in C else "medgemma_brief"
    g = lambda r, k: float(r[k])
    for m, nm in (("v2_27", "rk27"), ("stock27", "k27"), ("qwen38", "q"), (mg, "mg")):
        r = C[m]
        put(f"cal_{nm}_ece", f"{g(r, 'ece'):.3f}", RB_CAL, f"{m}.ece", "ECE, human-labeled")
        put(f"cal_{nm}_brier", f"{g(r, 'brier'):.3f}", RB_CAL, f"{m}.brier", "Brier, human-labeled")
        put(f"cal_{nm}_ce", pct(g(r, "conf_err")), RB_CAL, f"{m}.conf_err", "confident errors (% of questions)")
        put(f"cal_{nm}_cov", pct(g(r, "cov5")), RB_CAL, f"{m}.cov5", "coverage at 5% error (%)")
        q = RC[m]
        put(f"rc_{nm}_ece", f"{g(q, 'ece_after'):.3f}", RB_RECAL, f"{m}.ece_after", "ECE after cross-fitted recalibration")
        put(f"rc_{nm}_cov", pct(g(q, "cov5_after")), RB_RECAL, f"{m}.cov5_after", "coverage after recalibration (%)")
        put(f"rc_{nm}_ce", pct(g(q, "conf_err_after")), RB_RECAL, f"{m}.conf_err_after", "confident errors after recalibration (%)")
    for key, a, b_ in (("cp_k27", "v2_27", "stock27"), ("cp_q", "v2_27", "qwen38")):
        r = CP[(a, b_, "human_keys")]
        for met, scale, fmt in (("ece", 1, "{:.3f}"), ("conf_err", 100, "{:.1f}"), ("cov5", 100, "{:.1f}"), ("brier", 1, "{:.3f}")):
            d, lo, hi = g(r, f"{met}_d") * scale, g(r, f"{met}_lo") * scale, g(r, f"{met}_hi") * scale
            sgn = lambda v: ("+" if v > 0 else "−" if v < 0 else "") + fmt.format(abs(v))
            put(f"{key}_{met}", sgn(d), RB_CALP, f"{a}-{b_}.{met}_d", "paired difference")
            put(f"{key}_{met}_ci", f"{sgn(lo)} to {sgn(hi)}", RB_CALP, f"{a}-{b_}.{met}_lo/hi", "95% CI")
    r = CP[("v2_27", "stock27", "human_keys")]
    assert g(r, "ece_lo") > 0 and g(r, "conf_err_lo") > 0 and g(r, "cov5_lo") > 0 and g(r, "brier_hi") < 0, "calibration claims"
    assert g(RC["v2_27"], "cov5_after") > g(RC["stock27"], "cov5_after"), "claim: the coverage lead persists after recalibration"
    S = rj(ROB)["selective"]
    curves = [(lab, col, S[k]["human_keys"]["coverage"], [100 * (1 - a) for a in S[k]["human_keys"]["acc"]], lw)
              for k, lab, col, lw in (("v2_27", "RadKev-27B", "#2F6BD8", 1.6), ("stock27", "Kev-27B", "#9AA3B2", 1.3),
                                      ("qwen38", "Qwen3.8-27B", "#E8833A", 1.3), (mg, "MedGemma-27B-text", "#B4561A", 1.3))]
    names = (("stock27", "Kev"), ("v2_27", "RadKev"), ("qwen38", "Qwen"), (mg, "MedGemma"))
    cov_rows = [(lab, 100 * g(C[k], "cov5"), 100 * g(RC[k], "cov5_after")) for k, lab in names]
    ece_rows = [(lab, g(C[k], "ece"), g(RC[k], "ece_after")) for k, lab in names]
    import figures
    figures.fig_calib(curves, cov_rows, ece_rows, HERE / "figures")


RB_BLIND = "results_book/tables/blind.csv"


def build_results_robust():
    """Results 3.5: case withheld, answer words, answer space and wording; Figure 8."""
    R = rj(RB)
    B = {(r["task"], r["system"]): r for r in csv.DictReader(open(inp(RB_BLIND)))}
    g = lambda r, k: 100 * float(r[k])
    for task, tn in (("eurorad_dx", "dx"), ("medqa", "mq")):
        for m, nm in (("v2_27", "rk27"), ("stock27", "k27"), ("qwen38", "q"), ("medgemma_fix", "mg"), ("r9", "rk9"), ("stock9", "k9")):
            r = B[(task, m)]
            put(f"bl_{tn}_{nm}_full", f"{g(r, 'full'):.1f}", RB_BLIND, f"{task}.{m}.full", "accuracy with the case (%)")
            put(f"bl_{tn}_{nm}", f"{g(r, 'blind'):.1f}", RB_BLIND, f"{task}.{m}.blind", "accuracy, case withheld (%)")
        for kind in ("full", "blind"):
            v = 100 * R[f"blind_gain_v2_27_stock27_{tn if tn == 'medqa' else 'eurorad_dx'}_{kind}"] if tn == "dx" else 100 * R[f"blind_gain_v2_27_stock27_medqa_{kind}"]
            put(f"bl_{tn}_gain_{kind}", f"{v:+.1f}".replace("-", "−"), RB, f"blind_gain_v2_27_stock27_{task}_{kind}", "RadKev-27B minus Kev-27B (pp)")
    assert R["blind_gain_v2_27_stock27_eurorad_dx_blind"] > R["blind_gain_v2_27_stock27_eurorad_dx_full"]
    D = rj(AS_OUT)
    import numpy as np
    src = np.array(D["src"]); K = np.array([k or np.nan for k in D["K"]["orig"]], float)
    chance = [100 * np.nanmean(1 / K[src == s]) for s in ("eurorad_dx", "medqa")]
    put("bl_chance_dx", f"{chance[0]:.0f}", AS_OUT, "mean 1/K of original options", "Eurorad chance (%)")
    # answer words in the case (robustness2)
    E = rj(ROB2)["eurorad_leak"]
    fl, nf = E["answer_word_in_case"], E["no_answer_word"]
    put("aw_flag_n", str(fl["n"]), ROB2, "eurorad_leak.answer_word_in_case.n", "questions"); put("aw_none_n", str(nf["n"]), ROB2, "no_answer_word.n", "questions")
    for key, blk in (("aw_flag", fl), ("aw_none", nf)):
        d = blk["v2_27-stock27"]
        put(key, pct(d["d"]), ROB2, f"eurorad_leak.{key}.v2_27-stock27.d", "gain (pp)"); put(key + "_ci", ci(*d["ci"]), ROB2, "ci", "95% CI")
    put("aw_did", pct(R["leak_did"]), RB, "leak_did", "flagged minus unflagged gain (pp)"); put("aw_did_ci", ci(R["leak_did_lo"], R["leak_did_hi"]), RB, "leak_did", "95% CI")
    # wording (robustness)
    Wd = rj(ROB)["wording"]
    for sp in ("seen", "held_out"):
        x = Wd[sp]["v2_27-stock27"]["human_keys"]
        put(f"wd_{sp}", pct(x["micro"]), ROB, f"wording.{sp}.v2_27-stock27.human_keys.micro", "gain (pp)")
        put(f"wd_{sp}_ci", ci(*x["micro_ci"]), ROB, "micro_ci", "95% CI")
    put("wd_did", pct(R["did_wording_v2_27-stock27_heldout_minus_seen_hk"]), RB, "did_wording_v2_27-stock27_heldout_minus_seen_hk", "held-out minus seen (pp)")
    put("wd_did_ci", ci(R["did_wording_v2_27-stock27_heldout_minus_seen_hk_lo"], R["did_wording_v2_27-stock27_heldout_minus_seen_hk_hi"]), RB, "did_wording", "95% CI")
    # figure
    pa = [("Full case", g(B[("eurorad_dx", "stock27")], "full"), g(B[("eurorad_dx", "v2_27")], "full"), 100 * R["blind_gain_v2_27_stock27_eurorad_dx_full"]),
          ("Options only\n(case removed)", g(B[("eurorad_dx", "stock27")], "blind"), g(B[("eurorad_dx", "v2_27")], "blind"), 100 * R["blind_gain_v2_27_stock27_eurorad_dx_blind"])]
    pb = [("Full case", g(B[("medqa", "stock27")], "full"), g(B[("medqa", "v2_27")], "full"), 100 * R["blind_gain_v2_27_stock27_medqa_full"]),
          ("Options only\n(case removed)", g(B[("medqa", "stock27")], "blind"), g(B[("medqa", "v2_27")], "blind"), 100 * R["blind_gain_v2_27_stock27_medqa_blind"])]
    S = answer_space_summary(inp(AS_OUT))
    dd = lambda c: S["diff"][("eurorad_dx", c, "radkev27", "kev27")]
    pc = [(lab, 100 * dd(c)["d"], 100 * dd(c)["ci"][0], 100 * dd(c)["ci"][1]) for lab, c in
          (("Case's own\ndifferential", "orig"), ("Own + up to 8\nLLM-written", "orig_llm"), ("8 written\nby an LLM", "llm_8"), ("8 most similar\nin dataset", "sim_8"), ("64 random\nin dataset", "rand_64"))]
    pd = [("Answer word\nin case", 100 * fl["v2_27-stock27"]["d"], 100 * fl["v2_27-stock27"]["ci"][0], 100 * fl["v2_27-stock27"]["ci"][1]),
          ("No answer\nword", 100 * nf["v2_27-stock27"]["d"], 100 * nf["v2_27-stock27"]["ci"][0], 100 * nf["v2_27-stock27"]["ci"][1])]
    import figures
    figures.fig_robust(pa, pb, pc, pd, chance, HERE / "figures")


TASK_LABEL = {"iu_finding": "IU: finding present", "iu_normal": "IU: normal study", "iu_which": "IU: which finding",
              "eurorad_dx": "Eurorad: diagnosis", "eurorad_route": "Eurorad: subspecialty classification", "medmcqa_rad": "MedMCQA: radiology",
              "medmcqa_med": "MedMCQA: other subjects", "medqa": "MedQA", "medxpertqa": "MedXpertQA", "pubmedqa": "PubMedQA",
              "mmlu_anatomy": "MMLU: anatomy", "mmlu_clinical_knowledge": "MMLU: clinical knowledge", "mmlu_college_biology": "MMLU: college biology",
              "mmlu_college_medicine": "MMLU: college medicine", "mmlu_medical_genetics": "MMLU: medical genetics",
              "mmlu_professional_medicine": "MMLU: professional medicine", "ctrate_finding": "CT-RATE: finding present",
              "ctrate_normal": "CT-RATE: normal study", "ctrate_which": "CT-RATE: which finding",
              "teacher_cxr_findings_finding_status": "Teacher: finding status", "teacher_order_appropriate": "Teacher: appropriateness",
              "teacher_order_contrast": "Teacher: contrast", "teacher_order_exam": "Teacher: first examination", "teacher_order_priority": "Teacher: priority",
              "teacher_report_change": "Teacher: interval change", "teacher_report_critical": "Teacher: critical finding",
              "teacher_report_follow_up": "Teacher: follow-up", "teacher_report_incidental": "Teacher: incidental finding",
              "teacher_report_urgency": "Teacher: urgency"}


def _tab(name, colspec, header, rows, midrules=()):
    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule", header + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if i in midrules: lines.append(r"\midrule")
        lines.append(r + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "generated" / f"{name}.tex").write_text("\n".join(lines) + "\n")


def build_supp():
    """Supplementary tables generated from the results book and job outputs (every value with its source in this function)."""
    R = rj(RB); f = float
    p1 = lambda x: f"{100 * f(x):.1f}"
    pci = lambda a, lo, hi: f"{100 * f(a):.1f} ({100 * f(lo):.1f}, {100 * f(hi):.1f})"
    sg = lambda v: ("+" if v > 0.0005 else "−" if v < -0.0005 else "") + f"{abs(100 * v):.1f}"
    dci = lambda d, lo, hi: f"{sg(f(d))} ({sg(f(lo))}, {sg(f(hi))})"
    # S3: prespecified primary comparison by family
    P = {r["family"]: r for r in csv.DictReader(open(inp(RB_PRIMARY)))}
    rows = []
    for fam in HUMAN_FAMS + MACHINE_FAMS + ["overall_task_mean", "human_keys_task_mean", "radiology_human_keys_task_mean"]:
        r = P[fam]
        lab = {"overall_task_mean": "All tasks, task mean", "human_keys_task_mean": "Human-labeled tasks, task mean",
               "radiology_human_keys_task_mean": "Radiology human-labeled tasks, task mean"}.get(fam, FAMILY_LABEL.get(fam, fam))
        holm = r["p_holm_nodemo"] if r["p_holm_nodemo"] else "–"
        holm = "<0.001" if holm not in ("–",) and float(holm) < 0.001 else (f"{float(holm):.2f}" if holm != "–" else holm)
        rows.append(f"{lab} & {int(r['n_nodemo']):,} & {dci(r['diff_nodemo'], r['lo_nodemo'], r['hi_nodemo'])} & {holm} & {dci(r['diff'], r['lo'], r['hi'])}")
    rows.append(f"Mean over families (all) & & {sg(R['fmacro_v2_27_stock27_all'])} & & ")
    rows.append(f"Mean over families (human-labeled) & & {sg(R['fmacro_v2_27_stock27_hk'])} & & ")
    _tab("supp_primary", "@{}lrrrr@{}", r"Decision family & n & Prespecified & Holm $p$ & Full test set", rows, midrules=(5, 9, 12))
    # S4: per-task accuracy with prevalence baseline
    T = list(csv.DictReader(open(inp(RB_PERTASK))))
    rows = [f"{TASK_LABEL.get(r['task'], r['task'])} & {int(r['n']):,} & {p1(r['majority'])} & {p1(r['stock27_acc'])} & \\textbf{{{p1(r['v2_27_acc'])}}} & {p1(r['stock9_acc'])} & {p1(r['r9_acc'])} & {p1(r['qwen38_acc'])} & {p1(r['medgemma_fix_acc'])}" for r in T]
    first_ct = next(i for i, r in enumerate(T) if r["task"].startswith("ctrate_"))
    first_t = next(i for i, r in enumerate(T) if r["task"].startswith("teacher_"))
    hk = [i for i, r in enumerate(T) if not r["task"].startswith(("ctrate_", "teacher_"))]
    order = hk + [i for i, r in enumerate(T) if r["task"].startswith("ctrate_")] + [i for i, r in enumerate(T) if r["task"].startswith("teacher_")]
    rows = [rows[i] for i in order]
    _tab("supp_pertask", "@{}lrrrrrrrr@{}", r"Task & n & Baseline & Kev-27B & RadKev-27B & Kev-9B & RadKev-9B & Qwen & MedGemma", rows,
         midrules=(len(hk), len(hk) + sum(1 for r in T if r["task"].startswith("ctrate_"))))
    # S5: accuracy of every system
    A = list(csv.DictReader(open(inp(RB_ACC))))
    keep = ["v2_27", "r9", "stock27", "stock9", "kev4", "kev08", "laya", "laya_typed", "gliner_decide", "julia1", "qwen38", "medgemma_fix"]
    AA = {r["system"]: r for r in A}
    names = {"laya_typed": "Laya-typed-decisions"}
    rows = []
    for k in keep:
        r = AA[k]
        rows.append(f"{names.get(k, r['name'])} & {r['size']} & {pci(r['human_keys_micro'], r['human_keys_micro_lo'], r['human_keys_micro_hi'])} & "
                    f"{pci(r['radiology_human_keys_micro'], r['radiology_human_keys_micro_lo'], r['radiology_human_keys_micro_hi'])} & "
                    f"{pci(r['overall_micro'], r['overall_micro_lo'], r['overall_micro_hi'])} & {pci(r['overall_macro'], r['overall_macro_lo'], r['overall_macro_hi'])}")
    _tab("supp_accuracy", "@{}llrrrr@{}", r"System & Size & Human-labeled & Radiology, human-labeled & All questions & All tasks, task mean", rows, midrules=(2, 10))
    # S6: reasoning sample
    L = {r["system"]: r for r in csv.DictReader(open(inp(RB_RLEV)))}
    G = rj(RSN)["generation"]
    rlab = [("v2_27", "RadKev-27B"), ("r9", "RadKev-9B"), ("stock27", "Kev-27B"), ("stock9", "Kev-9B"), ("qwen38", "Qwen3.8-27B"),
            ("qwen38_think", "Qwen3.8-27B, reasoning"), ("medgemma_fix", "MedGemma-27B-text"), ("medgemma_think", "MedGemma-27B-text, reasoning")]
    rows = []
    for k, lab in rlab:
        r = L[k]; g = G.get(k, {}).get("overall")
        tok = f"{g['median_new_tokens']:,} / {100 * g['hit_cap_share']:.1f}" if g else "–"
        rows.append(f"{lab} & {p1(r['micro'])} & {p1(r['macro'])} & {p1(r['rad_hk'])} & {f(r['ece']):.3f} & {p1(r['cov5'])} & {tok}")
    _tab("supp_reasoning", "@{}lrrrrrr@{}", r"System & Accuracy & Task mean & Radiology & ECE & Coverage at 5\% & Tokens (median / \% at limit)", rows, midrules=(4,))
    # S7: latency
    L27 = rj(LAT27); L9 = rj(LAT9)
    lrow = lambda lab, mode, d: f"{lab} & {mode} & {d['n']} & {d['median']:.0f} & {d['p95']:.0f}"
    rows = [lrow("RadKev-27B", "one pass per record", L27["models"]["radkev27"]["per_question_ms"]),
            lrow("Kev-27B", "one pass per record", L27["models"]["stock27"]["per_question_ms"]),
            lrow("RadKev-9B", "one pass per record", L9["models"]["radkev9"]["per_question_ms"]),
            lrow("Kev-9B", "one pass per record", L9["models"]["stock9"]["per_question_ms"]),
            lrow("Qwen3.8-27B", "option-letter logits", L27["models"]["qwen38"]["letter"]["per_question_ms"]),
            lrow("Qwen3.8-27B", "generated answer letter", L27["models"]["qwen38"]["direct"]["per_question_ms"]),
            lrow("Qwen3.8-27B", "reasoning, then letter", L27["models"]["qwen38"]["reasoning"]["per_question_ms"]),
            lrow("MedGemma-27B-text", "option-letter logits", L27["models"]["medgemma"]["letter"]["per_question_ms"])]
    _tab("supp_latency", "@{}llrrr@{}", r"System & Mode & n & Median (ms) & 95th percentile (ms)", rows, midrules=(4,))
    # S8: 9B arms
    arms = [("stock9", "Kev-9B (released)", None), ("r9", "From Kev-9B, all data", "tr_r9_stock9"), ("b9", "From Qwen3.5-9B-Base, all data", "tr_b9_stock9"),
            ("r9f10", "From Kev-9B, 10\\% of data", "tr_r9f10_stock9"), ("b9f10", "From Qwen3.5-9B-Base, 10\\% of data", "tr_b9f10_stock9")]
    rows = []
    for k, lab, tr in arms:
        r = AA[k]
        trs = dci(R[tr], R[tr + "_lo"], R[tr + "_hi"]) if tr else "reference"
        rows.append(f"{lab} & {pci(r['human_keys_micro'], r['human_keys_micro_lo'], r['human_keys_micro_hi'])} & {pci(r['radiology_human_keys_micro'], r['radiology_human_keys_micro_lo'], r['radiology_human_keys_micro_hi'])} & {trs}")
    _tab("supp_init", "@{}lrrr@{}", r"9B model & Human-labeled & Radiology, human-labeled & Transfer suite, change vs Kev-9B", rows, midrules=(1,))
    # S9: calibration
    C = {r["system"]: r for r in csv.DictReader(open(inp(RB_CAL)))}; RC = {r["system"]: r for r in csv.DictReader(open(inp(RB_RECAL)))}
    rows = []
    for k, lab in (("v2_27", "RadKev-27B"), ("r9", "RadKev-9B"), ("stock27", "Kev-27B"), ("stock9", "Kev-9B"), ("b9", "9B, from base model"),
                   ("qwen38", "Qwen3.8-27B"), ("medgemma_fix", "MedGemma-27B-text")):
        r, q = C[k], RC[k]
        rows.append(f"{lab} & {f(r['ece']):.3f} & {f(q['ece_after']):.3f} & {f(r['brier']):.3f} & {p1(r['conf_err'])} & {p1(q['conf_err_after'])} & {p1(r['cov5'])} & {p1(q['cov5_after'])}")
    _tab("supp_calib", "@{}lrrrrrrr@{}", r"System & ECE & ECE, recal. & Brier & Conf. errors (\%) & Conf. errors, recal. & Coverage 5\% & Coverage 5\%, recal.", rows, midrules=(5,))
    # S10: options-only control
    B = list(csv.DictReader(open(inp(RB_BLIND))))
    nm = {"v2_27": "RadKev-27B", "stock27": "Kev-27B", "r9": "RadKev-9B", "stock9": "Kev-9B", "qwen38": "Qwen3.8-27B", "medgemma_fix": "MedGemma-27B-text"}
    rows = [f"{'Eurorad diagnosis' if r['task'] == 'eurorad_dx' else 'MedQA'} & {nm[r['system']]} & {pci(r['full'], r['full_lo'], r['full_hi'])} & {pci(r['blind'], r['blind_lo'], r['blind_hi'])}" for r in B]
    _tab("supp_blind", "@{}llrr@{}", r"Task & System & Full case & Options only (case removed)", rows, midrules=(6,))
    # S13: prespecified ablation and pilot comparators (full test set; shared bootstrap)
    RO = rj(ROB); PM, PP = RO["per_model"], RO["pairs"]
    acc = lambda m, sub, agg: pci(PM[m][sub][agg], *PM[m][sub][agg + "_ci"])
    dif = lambda k, sub, agg: dci(PP[k][sub][agg], *PP[k][sub][agg + "_ci"])
    rows = []
    for m, lab, pair in (("v2_27", "RadKev-27B (all sources)", None), ("v1med27", "27B, without CT-RATE and teacher-labeled data", "v2_27-v1med27"),
                         ("v0open27", "27B pilot", "v2_27-v0open27"), ("stock9", "Kev-9B (released)", None), ("ft9", "9B pilot", "ft9-stock9")):
        d = (f"{dif(pair, 'human_keys', 'micro')} & {dif(pair, 'overall', 'macro')}") if pair else "reference & reference"
        rows.append(f"{lab} & {acc(m, 'human_keys', 'micro')} & {acc(m, 'overall', 'macro')} & {d}")
    _tab("supp_ablation", "@{}lrrrr@{}", r"Model & Human-labeled & All tasks, task mean & Difference, human-labeled & Difference, task mean", rows, midrules=(3,))
    for k, key in (("v2_27-v1med27", "abl_v1med"), ("v2_27-v0open27", "abl_v0"), ("ft9-stock9", "abl_ft9")):
        for sub, agg, tag in (("human_keys", "micro", "hk"), ("overall", "macro", "tm")):
            x = PP[k][sub]
            put(f"{key}_{tag}", pct(x[agg]), ROB, f"pairs.{k}.{sub}.{agg}", "paired difference, full test set, points")
            put(f"{key}_{tag}_ci", ci(*x[agg + "_ci"]), ROB, f"pairs.{k}.{sub}.{agg}_ci", "95% CI")
    put("abl_v1med_hk_acc", p1(PM["v1med27"]["human_keys"]["micro"]), ROB, "per_model.v1med27.human_keys.micro", "accuracy, %")
    # reasoning read-out: share of questions decided by the written answer (it differed from the letter read-out)
    G = rj(RSN)["generation"]
    assert G["qwen38_think"]["overall"]["own_agrees_share"] == 1.0, "text: the written answer never overrode Qwen3.8-27B"
    for k, key in (("qwen38_think", "ovr_qwen"), ("medgemma_think", "ovr_mg")):
        o = G[k]["overall"]; put(f"{key}_pct", p1(o["own_parsed_share"] * (1 - o["own_agrees_share"])), RSN, f"generation.{k}.overall.own_parsed_share*(1-own_agrees_share)", "% of questions")
    put("fm_all", sg(R["fmacro_v2_27_stock27_all"]), RB, "fmacro_v2_27_stock27_all", "family-mean difference (pp)")
    put("fm_hk", sg(R["fmacro_v2_27_stock27_hk"]), RB, "fmacro_v2_27_stock27_hk", "family-mean difference, human-labeled (pp)")
    for r in T:   # claim in 3.1: every system exceeds the prevalence baseline on every human-labeled task
        if not r["task"].startswith(("ctrate_", "teacher_")):
            assert all(f(r[m + "_acc"]) > f(r["majority"]) for m in ("v2_27", "stock27", "r9", "stock9", "qwen38", "medgemma_fix")), r["task"]
    # S2: agreement with the regenerated teacher labels (as before)
    write_teacher_agreement(None)


def write_numbers():
    lines = ["% Generated by build.py. Do not edit; the ledger is numbers.csv.",
             r"\newcommand{\V}[1]{\ifcsname V@#1\endcsname\csname V@#1\endcsname\else\textcolor{red}{\textbf{??#1}}\fi}"]
    lines += [rf"\expandafter\def\csname V@{r['key']}\endcsname{{{r['value']}}}" for r in LEDGER]
    (HERE / "numbers.tex").write_text("\n".join(lines) + "\n")
    with open(HERE / "numbers.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["key", "value", "source", "field", "note"])
        w.writeheader()
        w.writerows(LEDGER)


def abstract_text():
    """The abstract as plain text, numbers filled in (what would be pasted into arXiv)."""
    tex = (HERE / "main.tex").read_text()
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    if not m:
        return None
    vals = {r["key"]: r["value"] for r in LEDGER}
    if (HERE / "v3" / "numbers_v3.csv").exists():
        vals |= {r["key"]: r["value"] for r in csv.DictReader(open(HERE / "v3" / "numbers_v3.csv"))}
    body0 = re.sub(r"\\input\{(v3/[^}]+\.tex)\}", lambda k: (HERE / k.group(1)).read_text(), m.group(1))
    paras = []
    for para in re.split(r"\n\s*\n", body0):
        body = re.sub(r"(?<!\\)%.*", "", para)  # comments, not \%
        body = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", body)
        body = re.sub(r"\\V\{([^}]*)\}", lambda k: vals.get(k.group(1), "??"), body)
        body = body.replace("\\%", "%").replace("--", "\u2013").replace("~", " ")
        body = re.sub(r"\\[a-zA-Z]+\*?", " ", body)
        body = " ".join(re.sub(r"[{}\\$]", "", body).split())
        if body:
            paras.append(body)
    return "\n".join(paras)


def compile_pdf():
    exe = shutil.which("tectonic")
    if not exe:
        print("tectonic not found; numbers written, PDF not compiled")
        return 1
    p = subprocess.run([exe, "--keep-logs", "main.tex"], cwd=HERE, capture_output=True, text=True)
    log = p.stdout + p.stderr
    bad = [l for l in log.splitlines() if re.search(r"error|undefined|\?\?", l, re.I)]
    print("\n".join(bad[:30]) or "compiled without errors")
    return p.returncode


def export_latex():
    r"""A self-contained, editable copy in export/: every \V{key} replaced by its value and every generated table inlined,
    with refs.bib, the .bbl and the figures, so it compiles on its own (Overleaf, arXiv). Compiled once to verify."""
    out = HERE / "export"; shutil.rmtree(out, ignore_errors=True); (out / "figures").mkdir(parents=True)
    vals = {r["key"]: r["value"] for r in LEDGER}
    V3L = {r["key"]: r["value"] for r in csv.DictReader(open(HERE / "v3" / "numbers_v3.csv"))}   # v3 results ledger (v3/build_v3.py)
    assert not set(V3L) & set(vals), "v3 ledger keys collide with build.py keys"
    vals |= V3L
    tex = (HERE / "main.tex").read_text()
    tex = re.sub(r"^\\input\{(?:numbers\.tex|v3/numbers_v3\.tex)\}.*\n", "", tex, flags=re.M)
    def inline(m):
        body = (HERE / m.group(1)).read_text()
        return body.rstrip("\n")
    for _ in range(3):
        tex = re.sub(r"\\input\{((?:generated|v3)/[^}]+\.tex)\}", inline, tex)
    tex = tex.replace("\\V{r3_*}", "r3_*")   # a wildcard in a comment of v3/results.tex, not a number
    missing = sorted(set(re.findall(r"\\V\{([^}]+)\}", tex)) - vals.keys())
    assert not missing, f"undefined numbers: {missing}"
    tex = re.sub(r"\\V\{([^}]+)\}", lambda m: vals[m.group(1)], tex)
    assert "\\V{" not in tex and "\\input{" not in tex
    header = ("% RadKev manuscript: editable, self-contained copy exported by build.py on "
              + __import__("datetime").date.today().isoformat() + ".\n"
              "% Numbers are written in as text. Edits here do not flow back to build.py; send this file back to merge them.\n")
    (out / "RadKev_manuscript.tex").write_text(header + tex)
    shutil.copy(HERE / "refs.bib", out / "refs.bib")
    for f in sorted(set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex))):
        shutil.copy(HERE / f, out / f)
    exe = shutil.which("tectonic")
    if exe:
        p = subprocess.run([exe, "--keep-intermediates", "RadKev_manuscript.tex"], cwd=out, capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-2000:]
        txt = subprocess.run(["pdftotext", "RadKev_manuscript.pdf", "-"], cwd=out, capture_output=True, text=True).stdout
        assert "??" not in txt, "unresolved reference in the exported copy"
        for f in out.iterdir():
            if f.suffix in (".aux", ".log", ".blg", ".out", ".toc") or f.name.endswith(".synctex.gz"): f.unlink()
    (HERE / "RadKev_manuscript_latex.zip").unlink(missing_ok=True)   # user 2026-10-06: deliverables are the local PDF and Overleaf only, no zip
    print(f"PDF: {out / 'RadKev_manuscript.pdf'} (export .tex is the Overleaf sync source)")


# Manuscript tables as CSV, named by their number in the manuscript (published in the code repository's results/tables/).
TABLE_CSV = [("Table1_data", "tab_data"), ("TableS1_teacher_labels", "tab_teacher"),   # manuscript order (order of first citation); S2-S12 by v3/build_v3.py
             ("TableS2_primary_by_task", "v3_primary_tasks"), ("TableS3_per_task", "v3_pertask"), ("TableS4_paired_differences", "v3_pairs"),
             ("TableS5_reasoning", "v3_reasoning"), ("TableS6_latency", "v3_latency"), ("TableS7_transfer", "v3_transfer"),
             ("TableS8_calibration", "v3_calib"), ("TableS9_preread", "v3_preread"), ("TableS10_options_only", "v3_blind"),
             ("TableS11_answer_space", "v3_answer_space"), ("TableS12_external", "v3_radgraph")]


def _plain(cell):
    """LaTeX table cell -> plain text."""
    t = re.sub(r"\\citep?\{[^}]*\}", "", cell).replace("\\ ", " ")
    t = re.sub(r"\\(?:multicolumn|multirow)\{[^}]*\}\{[^}]*\}\{(.*)\}", r"\1", t)
    t = re.sub(r"\\(?:textbf|textit|emph|mathrm|text)\{([^}]*)\}", r"\1", t)
    t = t.replace("\\%", "%").replace("\\&", "&").replace("$", "").replace("\\,", "").replace("~", " ")
    t = t.replace("\\Delta", "Δ").replace("\\leq", "≤").replace("\\geq", "≥").replace("\\times", "×").replace("--", "–")
    t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?", "", t).replace("{", "").replace("}", "")
    return re.sub(r"\s+", " ", t).strip()


def export_tables_csv():
    """Every manuscript table as CSV in tables_csv/ (values exactly as typeset; \\V{} resolved)."""
    import csv
    vals = {r["key"]: r["value"] for r in LEDGER}
    out = HERE / "tables_csv"; shutil.rmtree(out, ignore_errors=True); out.mkdir()
    for name, src in TABLE_CSV:
        if not (HERE / "generated" / f"{src}.tex").exists(): print(f"  {name}: generated/{src}.tex not built yet (run v3/build_v3.py)"); continue
        tex = re.sub(r"\\V\{([^}]+)\}", lambda m: vals[m.group(1)], (HERE / "generated" / f"{src}.tex").read_text())
        body = re.search(r"\\begin\{tabular\}\{(?:[^{}]|\{[^{}]*\})*\}\n?(.*?)\\end\{tabular\}", tex, flags=re.S).group(1)
        rows = []
        for line in re.split(r"\\\\(?:\[[^\]]*\])?", body):
            line = re.sub(r"\\(?:top|mid|bottom)rule|\\cmidrule(\([^)]*\))?\{[^}]*\}|\\addlinespace(\[[^\]]*\])?|\\hline", "", line).strip()
            if not line or line.startswith("%"): continue
            rows.append([_plain(c) for c in line.split("&")])
        with open(out / f"{name}.csv", "w", newline="") as f:
            csv.writer(f).writerows(rows)
    print(f"{len(TABLE_CSV)} tables written to {out.name}/")


# ======================================================================== public bundle (RadKev/paper/)
def _pick(*fields):
    """Keep only the dotted fields of a JSON object ('*' matches every key)."""
    def walk(s, d, ks):
        for k in (s if ks[0] == "*" else [ks[0]]):
            if len(ks) == 1: d[k] = s[k]
            else: walk(s[k], d.setdefault(k, {}), ks[1:])
    def f(x):
        out = {}
        for fld in fields: walk(x, out, fld.split("."))
        return out
    return f


PUBLISH = {  # inputs published as the subset the build reads; every other input in full
    ENV: _pick("steps.kev_env"),
    BAKEOFF: _pick("steps.parity"),
    TRAIN_RUN: _pick("steps.prepare.causal_conv1d"),
    PLAYGROUND: _pick("meta.n"),   # the 420 records are not redistributed (examples.tex is)
    FIGDATA: _pick("logs.*.points"),
    FIGTEACH: lambda x: {"teacher": {pool: [c for c in x["teacher"][pool] if c["tid"] == tid] for pool, tid, *_ in TEACH_EX}},
    AS_OUT: lambda x: {k: v for k, v in x.items() if k != "examples"},   # example questions (dataset text), not used here
}
ABS_PATH = re.compile(r"(?<![\w.:/~$-])/(?:[\w.~-]+/)+[\w.~-]*")


def _scrub(x):
    """Rewrite compute-environment paths: Hugging Face snapshots to org/name@revision, the job directory to $RADKEV_HOME."""
    if isinstance(x, dict): return {k: _scrub(v) for k, v in x.items()}
    if isinstance(x, list): return [_scrub(v) for v in x]
    if not isinstance(x, str): return x
    x = re.sub(r"/[^\s\"']*?/models--([^/\s]+?)--([^/\s]+)/snapshots/([0-9a-f]+)", r"\1/\2@\3", x)
    x = re.sub(r"GPU-[0-9a-f]{8}-[0-9a-f-]{27}", "GPU", x)   # device UUIDs of the compute node
    return re.sub(r"/[^\s\"']*?/jobs/radkev/", "$RADKEV_HOME/", x)


PUBLIC_README = """# Manuscript build

`build.py` regenerates every number, table and figure in the RadKev manuscript from the job outputs in `inputs/`.

```
python3 build.py --no-pdf   # numbers.csv / numbers.tex, generated/*.tex, tables_csv/*.csv, figures/
python3 build.py            # also compiles main.pdf (tectonic) and writes export/, a self-contained LaTeX copy
```

Requirements: Python 3.10 or later with numpy and matplotlib; the `radkev` package of this repository, which is
imported from the parent directory for the constants of data construction; tectonic and pdftotext for the PDF.

Every number in `main.tex` is a `\\V{key}` macro. The ledger `numbers.csv` lists each key with its printed value, the
input it was read from (`source`, a path under `inputs/`, a module of this repository, or an upstream file), the
field, and a description. Assertions stop the build if a statement in the text no longer holds for the inputs.

## Inputs

`inputs/` holds the job outputs as the build reads them, at their paths in the study's working repository. This
directory is written by `build.py --export-public` there; the same `build.py` runs in both places and produces
identical `numbers.csv`, `generated/*.tex` and `tables_csv/*.csv`. Differences from the raw job outputs:

- Absolute paths of the compute environment are rewritten: the job directory to `$RADKEV_HOME/`, Hugging Face
  snapshots to `org/name@revision`.
- Files of which the build reads only a few fields are reduced to those fields: `environment/setup.json`
  (`steps.kev_env`), `artifacts/bakeoff/artifacts/bakeoff.json` (`steps.parity`), `results/train/v2mg/train.json`
  (`steps.prepare.causal_conv1d`), `artifacts/figure_data/artifacts/figure_data.json` (training-loss points) and
  `artifacts/figure_data_teacher/artifacts/figure_data.json` (the two Eurorad cases shown in the teacher figure).
  `artifacts/answer_space2/artifacts/answer_space2.json` omits its example questions, which the build does not use.
- `results/playground/playground_00.json` holds only `meta.n`; its 420 test records are not redistributed.
  Supplementary Note S1, generated from them, is included as `inputs/examples.tex` and copied by the build.
- `inputs/constants.json` holds the constants the build reads from the job scripts; its keys are the ledger fields
  and `_from` names the script each was read from.

Dataset excerpts in `inputs/examples.tex` and in the teacher-figure input are reproduced under the licences stated in
the manuscript (Eurorad: CC BY-NC-SA 4.0, European Society of Radiology).
"""


def export_public(dest):
    """Write the public bundle into dest: this script, figures.py, main.tex, refs.bib, README.md and inputs/, scrubbed
    copies of every input the build read (USED), with the job-script constants and the generated examples."""
    assert not PUBLIC and USED, "export from the private repository, after the build"
    dest = Path(dest).resolve(); out = dest / "inputs"
    shutil.rmtree(out, ignore_errors=True); out.mkdir(parents=True)
    for rel in sorted(USED):
        p = out / rel; p.parent.mkdir(parents=True, exist_ok=True); raw = inp(rel).read_bytes()   # verbatim unless changed
        if rel.endswith(".json"):
            x = json.loads(raw); y = _scrub(PUBLISH.get(rel, lambda v: v)(x))
            raw = raw if y == x else (json.dumps(y, ensure_ascii=False) + "\n").encode()
        hit = ABS_PATH.search(raw.decode())
        assert not hit, f"{rel}: absolute path {hit.group(0)[:60]}"
        p.write_bytes(raw)
    (out / "constants.json").write_text(json.dumps({"_from": JOB_CONSTANTS, **job_constants()}, indent=1) + "\n")
    shutil.copy(HERE / "generated" / "examples.tex", out / "examples.tex")
    if (dest.parent / "radkev").is_dir():   # the public modules must give the constants the private build used
        bd, te = code_modules()
        sys.path.insert(0, str(dest.parent)); from radkev import data as pb, teacher as pt
        for a, b, names in ((bd, pb, "IU_VOCAB SECTIONS MEDMCQA_CAP PRESENT_T NORMAL_T WHICH_T"),
                            (te, pt, "URGENCY FOLLOW CHANGE STATUS EXAMS CONTRAST PRIORITY APPROPRIATE REPORT_Q ORDER_Q")):
            for k in names.split():
                assert getattr(a, k) == getattr(b, k), f"{b.__name__}.{k} differs from the private module"
    for f in ("build.py", "figures.py", "main.tex", "refs.bib"):
        shutil.copy(HERE / f, dest / f)
    (dest / "README.md").write_text(PUBLIC_README)
    files = sorted(p for p in dest.rglob("*") if p.is_file() and (p.parent == dest or out in p.parents))
    print(f"public bundle: {len(files)} files, {sum(p.stat().st_size for p in files) / 1e6:.1f} MB in {dest}")


if __name__ == "__main__":
    build_numbers()
    build_rb()
    build_fig_compare()
    build_fig_training()
    build_fig_training_v3()
    write_design_v3()
    build_fig_teacher()
    build_methods()
    build_answer_space()
    build_fig_kev()
    build_results_primary()
    build_results_llm()
    build_subset_split()
    build_external()
    build_results_spec()
    build_results_calib()
    build_results_robust()
    build_supp()
    write_numbers()
    export_tables_csv()
    print(f"{len(LEDGER)} numbers written to numbers.tex / numbers.csv")
    a = abstract_text()
    if a:
        (HERE / "abstract.txt").write_text(a + "\n")  # plain text for the arXiv form
        print(f"abstract: {len(a.split())} words, {len(a)} characters (arXiv limit 1,920); plain text in abstract.txt")
    if "--export-public" in sys.argv:
        export_public(sys.argv[sys.argv.index("--export-public") + 1])
        sys.exit(0)
    if "--no-pdf" not in sys.argv:
        rc = compile_pdf()
        if rc == 0: export_latex()
        sys.exit(rc)
