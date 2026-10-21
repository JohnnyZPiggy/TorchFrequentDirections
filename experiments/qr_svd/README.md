# Issue #9 CPU experiment

This isolated subclass is opt-in research code. The public package and default
direct SVD remain unchanged. Issue #9 supplies the scoped mathematical hypothesis;
GPT Architect retains scientific acceptance and integration decisions.

R1: Fast FD has a sequential shrink recurrence, with dense m-by-d factorization
at each full buffer and some snapshots. Snapshots with at most ell rows bypass
factorization. Earlier short-buffer work remains in both variants; buffer clearing
and all production finite checks remain intact (PRs #4, #6, #8).

R2: For m <= d, reduced QR gives C^T = QR, Q^T Q = I_m. If
R^T = U diag(s) W^T, then C = U diag(s) W^T Q^T. Thus Vh = W^T Q^T
has orthonormal rows and the same singular values. Applying the same normalized
ell-th-singular-value shrink produces equal Gram matrices in exact arithmetic.
For m > d use direct SVD; empty input uses the existing zero output path.
This reduces the SVD to m-by-m but adds QR and right-vector recovery, all with
O(d m^2) work. It does not remove stream dependencies or imply a device speedup.
Rank deficiency and tied singular values allow arbitrary singular bases; equality
of raw rows is not a contract. Finite precision can accumulate across shrinks.

R3: Independently test explicit float64 Gram eigensolves, full-input SVD tails,
and right-subspace projection bounds for each integer 0 <= k < ell, k <= d.
Tolerance is 32 eps(input dtype) max(n,d,2ell,1) ||A||_F^2; cross-variant Gram
Frobenius gap uses the same tolerance. This energy-scaled absolute bound does
not certify relative accuracy of tiny spectral components. Test streams, severe
spectra and faults before timing. No prior chat measurements are evidence here.
Status at source creation: scoped HYPOTHESIS, independent CPU run pending;
this engineer note does not grant RESEARCH_READY or scientific acceptance.

Reproduce in a clean Linux checkout with existing CPU PyTorch, no extra packages:

```sh
timeout 1000 python -m experiments.qr_svd.driver /outside/checkout/result.json
```

The driver records exact clean SHA, seed, CPU, build and threads. It sets intra-op
and inter-op threads to one. Correctness gates precede timing. Two warmups per
variant precede 20 pairs, exactly ten AB and ten BA in deterministic randomized
order. Both variants share tensors, process and thread settings. Raw ordered times,
median, nearest-rank p95, sample standard deviation and 2,000-resample descriptive
paired-median bootstrap intervals are retained. Intervals have no simultaneous
coverage correction and do not establish cross-session reproducibility.

Predeclared screening criterion: every primary wide full-stream configuration
must exceed 1.05 median paired speedup and have descriptive interval lower bound
above 1.0, after all mathematical gates pass. Small, no-shrink and m>d controls
are reported separately. This is only an integration-candidate screen. Performance
is single-host/single-session, CPU-only; no GPU extrapolation is permitted.
Instrumented QR, small SVD, Vh recovery and shrink/output diagnostics are separate
intrusive measurements, not additive median components of the primary timing.
The process has a 900-second internal bound and an external 1000-second timeout.
