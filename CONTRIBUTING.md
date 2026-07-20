# Contributing to GC-Hierarchical

GC-Hierarchical is scientific research software for hierarchical inference
of globular-cluster systems.

## Development principles

1. The validated scientific baseline must remain reproducible.
2. Changes to inference infrastructure must not silently modify the frozen
   three-component v5 likelihood.
3. New work should be developed on a separate branch.
4. Tests must pass before changes are merged into `main`.
5. Scientific changes must be documented explicitly.

## Branch workflow

The `main` branch contains validated code.

Create a feature branch before making changes:

```bash
git switch main
git pull origin main
git switch -c feature-name

