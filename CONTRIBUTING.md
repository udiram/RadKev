# Contributing

Thanks for your interest. Issues and pull requests are welcome, especially reproductions on other hardware, new open radiology
sources for `radkev.data`, and fixes to the experiment runners.

## Development

```bash
uv venv && source .venv/bin/activate
uv pip install "torch>=2.6,<2.9" "kev @ git+https://github.com/jaredpalmer/kev@f2bb629d670f5b746f712fc05550a098526c836b" -e ".[dev]"
pytest            # synthetic fixtures; the Kev-metric tests run when kev is importable
ruff check .
```

## Ground rules

- **No dataset text in the repository**, not even a few lines in a test, apart from the attributed excerpts that the manuscript
  itself prints (`paper/`: the example records of Supplementary Note S1 and the two Eurorad cases of Figure 1). Tests use
  invented fixtures (`tests/conftest.py`); examples are written manually. Released records are distributed through the Hugging
  Face dataset, not this repository.
- **No protected health information, ever**, in code, issues, logs or results. If you evaluate on institutional data, report
  aggregates only.
- **Keep the builders deterministic.** A change to `radkev/data.py` or `radkev/teacher.py` that alters the records for the same
  raw files changes every result; state this in the pull request; published results change only with a new run.
- **Report what you measured.** New numbers need the command that produced them and the comparison they are paired with.
- Match the existing style: compact, commented where the reason is not obvious, `ruff check .` clean.

## Security

Please report security issues privately to usram@wisc.edu rather than in a public issue.
