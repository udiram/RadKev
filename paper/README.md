# Manuscript build

`build.py` regenerates every number, table and figure in the RadKev manuscript from the job outputs in `inputs/`.

```
python3 build.py --no-pdf   # numbers.csv / numbers.tex, generated/*.tex, tables_csv/*.csv, figures/
python3 build.py            # also compiles main.pdf (tectonic) and writes export/, a self-contained LaTeX copy
```

Requirements: Python 3.10 or later with numpy and matplotlib; the `radkev` package of this repository, which is
imported from the parent directory for the constants of data construction; tectonic and pdftotext for the PDF.

Every number in `main.tex` is a `\V{key}` macro. The ledger `numbers.csv` lists each key with its printed value, the
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
