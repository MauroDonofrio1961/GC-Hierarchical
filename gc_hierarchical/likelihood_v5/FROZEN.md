# Frozen implementation

The Python modules in this directory are copied from the validated
`NGC5128_three_component_v5` package.

They are intentionally kept scientifically unchanged in 1.0-alpha.
Infrastructure code may call them, but should not silently alter their
likelihood, priors, softmax mixture normalization, or posterior calculations.
