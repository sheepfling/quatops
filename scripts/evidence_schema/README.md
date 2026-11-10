# Evidence schema modules

These modules separate target capture formats from the common evidence
contract:

- `converter_captures.py` defines strict Pydantic models for converter and
  transport inputs, contracts, and native observations.
- `engine_captures.py` defines the corresponding engine capture and contract
  models.
- `semantic_serializers/` maps each target's validated models into the shared
  semantic witness types in `crystallized.py`.
- `normalizer.py` checks those witnesses against canonical inputs and returns
  the resolved common capture type.
- [`scripts/converter_evidence.py`](../converter_evidence.py) and
  `engine_evidence.py` load versioned packages, verify their provenance, and
  call the right serializer.
- `tolerances.py` contains the numeric acceptance thresholds used for each
  converter target.

Pydantic models own parsing, field constraints, strictness, and immutability.
An after-validator is used only for a cross-field or ordered-record invariant
that cannot be expressed by an individual field declaration.
