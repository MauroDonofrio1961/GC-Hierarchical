
# Version 5 mixture correction

For each rank \(k\), two unconstrained logits are fitted. The third logit is
fixed to zero for identifiability:

\[
\boldsymbol{\pi}_k
=
\operatorname{softmax}(a_{b,k},a_{r,k},0).
\]

This parameterization guarantees:

\[
\pi_{b,k}>0,\quad \pi_{r,k}>0,\quad \pi_{{\rm bg},k}>0,
\]

and

\[
\pi_{b,k}+\pi_{r,k}+\pi_{{\rm bg},k}=1.
\]

Moderately informative Dirichlet priors encode the expectation that Gold is
cleaner than Silver while still allowing the data to determine the fractions.

All posterior membership vectors are explicitly checked for normalization,
finiteness, and valid probability bounds before results are written.
