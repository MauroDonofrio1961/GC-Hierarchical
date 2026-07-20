# Step 1 instructions

## A. Install

Double-click `install.command`.

macOS may block the file because it came from the internet. In that case:

1. Control-click `install.command`.
2. Choose **Open**.
3. Confirm **Open**.

The script creates a private Python environment named `.venv` and installs only
the packages required for Step 1. FSPS is intentionally postponed.

## B. Prepare and validate NGC 5128

Double-click `run.command`.

Expected final message:

```text
Data preparation completed.
Science catalogue: ...
Confirmed catalogue: ...
Background catalogue: ...
Manifest: ...
```

## C. Run tests

Double-click `run_tests.command`.

Expected result:

```text
4 passed
```

## D. Files to return for review

Compress and upload either:

- `data/NGC5128/processed/`, or
- `results/NGC5128/`.

For Step 1, `manifest.json` is sufficient if the tests also pass.

## E. Do not run yet

Do not double-click `run_science.command` yet. The full v5 calculation requires
a working FSPS installation and is intentionally separate from the first
infrastructure test.
