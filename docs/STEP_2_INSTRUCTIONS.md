# Step 2 instructions

Step 2 converts the frozen v5 likelihood from an externally launched script
into a library model with standard framework interfaces.

## Install or upgrade

This package can be placed in a new folder. Double-click `install.command`.

## Architecture test

Double-click:

`run_step2_tests.command`

The expected result is:

`10 passed`

These tests do not require FSPS. They verify:

- project configuration;
- raw catalogue validation;
- generated catalogue selection;
- Galaxy metadata and DatasetBundle interfaces;
- the 17-parameter v5 model contract;
- three-component softmax normalization;
- exact object-by-object agreement between DataManager selection and frozen v5
  catalogue selection.

## Optional FSPS check

Do not do this until a working FSPS installation is available.

1. Double-click `install_fsps.command`.
2. In Terminal, run:

   `source .venv/bin/activate`

   `python run.py check --galaxy NGC5128`

## Optional quick likelihood fit

After the FSPS check succeeds:

`python run.py quick --galaxy NGC5128`

The likelihood now runs in the same Python process through:

- `Galaxy`
- `DatasetBundle`
- `ThreeComponentV5Model`
- `FitResult`

The old `run_v5_frozen.py` subprocess bridge is no longer used.
