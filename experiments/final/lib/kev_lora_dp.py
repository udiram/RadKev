"""radkev: data-parallel LoRA training for kev.train (gated by RADKEV_LORA_DP=1; without it kev.train is unchanged).

Kev distributes only full-weight runs (FSDP2). For a LoRA delta fine-tune of Kev-27B, each rank here holds the whole backbone
split across its own GPU pair (KEV_DEVICE_MAP=auto), the epoch's records are shared between the ranks exactly as for FSDP2
(full_ft.rank_share), and before each optimizer step the trainable gradients (LoRA and pointer head) are summed over the ranks.
The trainer already divides each micro-batch's loss by the step's global record count (FSDP2 uses a divide factor of 1), so the
summed gradient equals the single-process one. Rank 0 alone saves. A final check compares the parameter checksums of the ranks.

    python kev_lora_dp.py apply <kev checkout>      # idempotent, exact-match edits of kev/train.py
"""
import sys
from pathlib import Path

HELPER = '''

def _radkev_allreduce_grads(model):
    """radkev (RADKEV_LORA_DP): sum the trainable gradients over the ranks; each rank's parameters may sit on two GPUs."""
    import torch.distributed as dist
    ps = list(model.trainable_parameters())
    for p in ps:
        if p.grad is None: p.grad = torch.zeros_like(p)
    main = torch.device("cuda", torch.cuda.current_device())
    flat = torch.cat([p.grad.detach().reshape(-1).to(main, torch.float32) for p in ps])
    dist.all_reduce(flat)
    o = 0
    for p in ps:
        n = p.numel(); p.grad.copy_(flat[o:o + n].view_as(p.grad).to(p.grad.device, p.grad.dtype)); o += n


def _radkev_param_check(model):
    """radkev (RADKEV_LORA_DP): the ranks' trainable parameters must be identical after training."""
    import torch.distributed as dist
    s = torch.tensor([float(sum(p.detach().double().sum().item() for p in model.trainable_parameters()))], dtype=torch.float64, device="cuda")
    lo, hi = s.clone(), s.clone()
    dist.all_reduce(lo, op=dist.ReduceOp.MIN); dist.all_reduce(hi, op=dist.ReduceOp.MAX)
    print(f"radkev_dp_param_check min={lo.item():.10e} max={hi.item():.10e} equal={bool(lo.item() == hi.item())}", flush=True)


def _radkev_acc(terms, logits, v):
    """radkev (RADKEV_LOG_ACC): training-batch accuracy, overall and by source (soft targets: the target's argmax)."""
    for z, q in zip(logits, v.rec["questions"]):
        t = q.get("target"); y = max(range(len(t)), key=lambda i: t[i]) if t is not None else int(q["label"])
        ok = float(int(z.argmax()) == int(y)) * v.share; src = str(getattr(v, "source", None) or "other")
        terms["acc"] += ok; terms["acc_n"] += v.share; terms["acc:" + src] += ok; terms["accn:" + src] += v.share


def _radkev_logx(run, sched, norm, model, step, world):
    """radkev (RADKEV_LOG_ACC): extra fields on the 10-step log line; every 500 steps the ranks' parameter checksums."""
    if os.environ.get("RADKEV_LOG_ACC") != "1": return ""
    if world > 1 and step % 500 == 0: _radkev_param_check(model)
    src = {k[4:]: [round(run[k], 2), round(run["accn:" + k[4:]], 2)] for k in list(run) if k.startswith("acc:")}
    return (f" acc {run['acc'] / max(run['acc_n'], 1e-9):.3f} lr {sched.get_last_lr()[0]:.3e} gnorm {norm if norm is not None else float('nan'):.3f}"
            f" src {json.dumps(src, separators=(',', ':'))}")


def main():'''

EDITS = [
    ("    rank, world = full_ft.init_distributed(dev) if a.full_ft else (0, 1)",
     "    rank, world = full_ft.init_distributed(dev) if (a.full_ft or os.environ.get(\"RADKEV_LORA_DP\") == \"1\") else (0, 1)   # radkev: LoRA data parallel"),
    ("    if world > 1: full_ft.shard(model)", "    if world > 1 and a.full_ft: full_ft.shard(model)   # radkev: LoRA ranks keep a whole (split) model"),
    ("                if not a.full_ft: norm = float(torch.nn.utils.clip_grad_norm_(",
     "                if not a.full_ft and world > 1: _radkev_allreduce_grads(model)   # radkev\n                if not a.full_ft: norm = float(torch.nn.utils.clip_grad_norm_("),
    ("    seen, tokens_seen = full_ft.global_sum([seen, tokens_seen])",
     "    seen, tokens_seen = full_ft.global_sum([seen, tokens_seen])\n    if world > 1 and not a.full_ft: _radkev_param_check(model)   # radkev"),
    ("    else: model.lm.save_pretrained(a.out)", "    elif not rank: model.lm.save_pretrained(a.out)   # radkev: LoRA ranks hold identical adapters; rank 0 saves"),
    ("        terms[\"ce\"] += ce.item(); loss = loss + ce",
     "        terms[\"ce\"] += ce.item(); loss = loss + ce\n        if os.environ.get(\"RADKEV_LOG_ACC\") == \"1\": _radkev_acc(terms, logits, v)   # radkev"),
    ("s/rec\", flush=True)", "s/rec\" + _radkev_logx(run, sched, locals().get(\"norm\"), model, step, world), flush=True)"),
    ("\n\ndef main():", HELPER),
]


VERSION = "radkev-dp-v2"


def apply(kev):
    """kev/train.py is not touched by kev_multigpu.patch, so an older version of these edits is undone with git first."""
    import subprocess
    f = Path(kev) / "kev/train.py"; s = f.read_text()
    if VERSION in s: return "already applied"
    if "_radkev_allreduce_grads" in s:
        subprocess.run(["git", "-C", str(kev), "checkout", "--", "kev/train.py"], check=True); s = f.read_text()
    for old, new in EDITS:
        if s.count(old) != 1: raise SystemExit(f"kev/train.py: expected exactly one match for {old[:60]!r}, found {s.count(old)}")
        s = s.replace(old, new)
    f.write_text(s + f"\n# {VERSION}\n"); return "applied"


if __name__ == "__main__":
    print(apply(sys.argv[2]))
