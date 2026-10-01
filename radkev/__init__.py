"""RadKev: a radiology-specialised decision model, fine-tuned from Kev.

Modules (each is also a command: python -m radkev.<module> --help):
    data       open and gated sources -> Kev training/eval records (patient/case splits, held-out wordings)
    teacher    two-teacher LLM labels for orders/triage, and the zero-shot LLM baseline scorer
    fetch      download the open sources
    evaluate   score a checkpoint (or precomputed probabilities) with Kev's metrics
    compare    paired cluster-bootstrap comparisons per decision family and task
    predict    ask a checkpoint for decisions in-process
"""
__version__ = "1.0.0"
