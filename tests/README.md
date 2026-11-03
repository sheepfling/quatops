# Test layout

Test code follows the product boundaries it covers:

- `quatops/` mirrors the package under `src/quatops/`. Its `hamilton/`,
  `jpl/`, `numpy/`, `types/`, `scalar/`, and `engines/` subtrees group tests
  by API family.
- `experiments/` mirrors the evidence products under `experiments/`: native
  engines, converters, mathematical oracles, and transports. Shared oracle
  checks live under `experiments/oracles/shared/`.
- `benchmarks/` groups benchmark cases by the QuatOps or NumPy APIs they time.
- `support/` contains helpers shared by tests.
- `golden/` and `oracles/` hold retained fixtures and evidence schemas. Replay
  suites live under the matching product area in `quatops/` or `experiments/`.
- `typing_tests/` contains static consumer examples selected by Pyright.

Keep new tests in the matching product subtree instead of adding modules at the
top of `tests/`. Keep captured values and test helpers separate from test code.
