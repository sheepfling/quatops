# Rotation precision investigation

**Historical snapshot:** These measurements predate the later narrow NumPy
coefficient-order parity fix. The near-diagonal NumPy example below is now fixed;
the general cancellation case is still a known limitation. The table preserves
the original measured evidence.

This is a local experiment, not an implemented accuracy mode or a guarantee.
The current `rotate` implementation is unchanged by this investigation. Its
float64 rotation coefficients can lose accuracy before the compensated vector
dot product is evaluated. The exact-rational reference uses the exact binary
values of the float inputs and rounds the final rational results to float.

| Input (`q`, `v`) | Exact first or second component | Current output |
| --- | --- | --- |
| `(1, 1, 1e-8, 0)`, `(0, 1e308, 0)` | second `5e291` | NumPy `0`; scalar modules `5e291` |
| `(5, 12, 13, 0)`, `(1e308, 1, 1)` | first `17/13` ≈ `1.3076923076923077` | all modules ≈ `1.1102230246251566e292` |

For the second example, dividing the quaternion by its largest component
rounds it before the coefficient calculation. Pairing opposite signs in the
final vector sum cannot recover information already lost in that coefficient.

A local run of `rotation_benchmark.py` used CPython 3.12.14, NumPy 2.3.5,
SciPy 1.17.0, and macOS ARM64. Inputs were preconstructed. Results are best
and median of five runs; each scalar run used 3,000 calls and each array run
used 10,000 quaternion/vector pairs. The Fraction and Decimal prototypes are
not complete implementations and omit the library's input validation and
backend-specific output behavior. Decimal used 1,200 digits; this experiment
establishes agreement for the examples, not a general exactness guarantee.

| Workload | Current | Fraction prototype | Decimal prototype |
| --- | ---: | ---: | ---: |
| Ordinary scalar, µs/call best / median | 4.116 / 4.155 | 16.745 / 16.793 | 3.417 / 3.463 |
| Zero-coefficient scalar, µs/call | 4.033 / 4.051 | 18.622 / 18.658 | 11.338 / 11.349 |
| Near-zero-coefficient scalar, µs/call | 4.042 / 4.070 | 24.720 / 24.867 | 22.156 / 22.221 |
| 10,000 ordinary pairs, ms/batch | NumPy 0.692 / 0.699 | Python loop 175.344 / 175.794 | not measured |

Run the current experiment from the repository root with the local development
environment:

```sh
.venv/bin/python experiments/run.py rotation-benchmark
```

## Research directions, not selected API changes

The accuracy options below are ideas for future work, not pending API
decisions. The current `rotate` behavior and its known limitations are defined
in the [API contract](API.md).

1. Improve the ordinary float64 kernel, e.g. exact power-of-two scaling and
   compensated coefficient sums. A local prototype fixes the two reported
   examples but is not generally exact: product rounding, underflow and
   cancellation still matter. For example, the exact Pythagorean relation in
   `(2000001, 2000002000000, 2000002000001, 0)` can still produce a spurious
   coefficient after rounded products. A cheap improvement needs parity and
   adversarial tests and a clearly bounded error policy.
2. Offer an explicitly selected accurate operation or backend based on exact
   arithmetic for the binary inputs. It can provide a substantially stronger
   contract but needs a defined rounding and overflow policy and will cost more,
   especially for NumPy batches. It would not be hidden representation dispatch.
3. Explore a fast path with a fallback only when a proved error bound says the
   requested accuracy cannot be met. A heuristic near-zero test alone is not a
   guarantee: it misses other coefficient, product, scaling, and dot-product
   errors. Fallback frequency and batch costs need measurement on real workloads.

Recommendation: first choose the accuracy guarantee expected from `rotate`.
Until then retain the current fast implementation and expose the known cases;
prototype bounded or explicitly accurate alternatives without claiming these
measurements establish universal speed or correctness.
