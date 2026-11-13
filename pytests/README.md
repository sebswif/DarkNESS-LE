# pytests

Simple pytest checks. Run from the repo root:

```bash
python -m pytest
```

Two kinds of test:

- **Sanity** — shapes, limits, and scaling laws that follow from the formulas
  themselves (e.g. QE is between 0 and 1, counts scale as g^4). These run now.
- **Reference** — marked `@pytest.mark.reference` and skipped. Each one names
  the published value it needs. When the reference is sourced, fill in the
  number, cite it, and remove the skip. See `memory/references.md`.

Tests that need `pymsis` or `xraylib` skip if those aren't installed
(`python -m pip install -r requirements.txt` installs them).

The older `tests/` folder (unittest) still works and is left as is.
