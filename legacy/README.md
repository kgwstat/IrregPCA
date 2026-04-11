# Legacy Archive

This folder contains pre-fix versions of files that were changed to correct the documented input orientation for `IrregPCA`.

These files are retained **for historical reference only** and are **not part of the supported API**. Do not import from this directory.

## Background

Before the orientation fix, the README documented the input tensor with the wrong orientation (`(d+2, N)` instead of the correct `(N, d+2)`). The implementation already used the correct `(N, d+2)` layout; only the documentation and validation were wrong.

## Contents

- `README.pre_orientation_fix.md` — the README as it existed before the fix
- `src/irregpca/core.py` — the `IrregPCA` entry point before input validation and docstring were added
