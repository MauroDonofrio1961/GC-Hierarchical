# Step 3 instructions

Step 3 introduces the final high-level scientific object model.

## What changed

The new `HierarchicalGCModel` owns:

- the `Galaxy`;
- the prepared `DatasetBundle`;
- the selection function;
- the FSPS stellar-population grid;
- the background density;
- the blue/red GC population density;
- the posterior;
- the optimizer;
- the sampler;
- output provenance and membership probabilities.

The frozen v5 scientific implementation is still unchanged. Step 3 wraps it in
explicit component and inference interfaces.

## Install and test

1. Unzip the alpha3 package into a new folder.
2. Double-click `install.command`.
3. Double-click `run_step3_tests.command`.

Expected result:

`16 passed`

The tests use dependency injection for the high-level model, so they do not
need to calculate an FSPS grid.

## Use the FSPS already installed on your Mac

A package being installed somewhere on the Mac does not guarantee it is
visible inside this project's `.venv`.

Double-click:

`check_fsps.command`

If it prints `FSPS Python import passed`, the project environment can use your
existing FSPS installation.

If it reports that `fsps` cannot be imported, double-click
`install_fsps.command`. This installs the Python binding inside `.venv`; it
does not replace the underlying FSPS installation.

## First real diagnostic fit

Only after `check_fsps.command` succeeds, double-click:

`run_quick.command`

The quick fit now uses:

```python
galaxy = Galaxy("NGC5128", root=".")
model = HierarchicalGCModel.from_galaxy(galaxy, mode="quick")
model.build()
result = model.fit(mode="quick", output_directory=run_directory)
```

Upload the newly created folder under `results/NGC5128/` for verification.

## Full science fit

Do not use `run_science.command` until the quick fit completes successfully.
