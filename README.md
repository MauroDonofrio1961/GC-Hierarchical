# GC-Hierarchical 1.0-alpha3

Step 3 provides the final high-level object model around the validated NGC 5128
three-component likelihood.

## Run the architecture tests

1. Double-click `install.command`.
2. Double-click `run_step3_tests.command`.

Expected result:

```text
16 passed
```

## Check the existing FSPS installation

Double-click:

```text
check_fsps.command
```

A system-level FSPS installation and the Python package inside this project's
`.venv` are separate. If the check fails, run `install_fsps.command`.

## Run the first real fit

After the FSPS check succeeds, double-click:

```text
run_quick.command
```

Do not run the full science fit before the quick diagnostic succeeds.

See `docs/STEP_3_INSTRUCTIONS.md` for complete instructions.
