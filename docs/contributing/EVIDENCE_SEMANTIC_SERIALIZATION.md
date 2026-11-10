# Semantic evidence serialization

The interoperability product has three separate stages. Native output stays
target-specific; a target adapter gives it explicit mathematical and frame
meaning; the common normalizer accepts only a compatible contract and witness
set. The normalized record shape is version 2; native capture package schemas
remain versioned independently.

```text
native capture + experiment-owned semantic contract
                 ->
one target semantic serializer
                 ->
one concrete serialization model
                 ->
shared normalizer
                 ->
CrystallizedCapture / EvidenceProduct
```

## Responsibilities

The typed input models and normalization boundaries are indexed in the
[evidence schema guide](../../scripts/evidence_schema/README.md).

- `scripts/evidence_schema/converter_captures.py` validates each target's input and
  native-output shape. It preserves native field names and does not assign
  shared physical meaning to them.
- Each pinned experiment owns `semantic_contract.json` beside its capture
  code and inputs. Its retained manifest carries the repository-relative path
  and SHA-256 digest; the normalized provenance points back to that file.
- `scripts/evidence_schema/semantic_serializers/` contains one adapter for
  each converter experiment: Eigen, MuJoCo, OpenCV, SpiceyPy, ROS 2, Cesium,
  glTF/Three.js, and OpenUSD. Each adapter owns its evidence layer, contract,
  native field selection, and interpretation of the declared contract and
  inputs. The registry in
  `semantic_serializers/registry.py` is keyed by the validated contract model,
  never by a product-name string.
- Each serializer returns a concrete Pydantic model. Its layer, contract type,
  and witness family are linked in the model itself: engine quaternion, native
  quaternion API, rotation vector, frame chain, geographic transport, or scene
  transport. `SemanticSerialization` is the closed union accepted by the
  normalizer; it is not a generic model with independent contract and witness
  unions.
- `scripts/evidence_schema/contracts.py` provides the common constructor used
  by the three direct-quaternion library adapters. OpenCV and the transport
  adapters construct their distinct contract shapes in their own modules.
- `scripts/evidence_schema/semantic_serializers/engines.py` owns the engine
  semantic serializer. The pinned engine captures share one typed witness
  format, and each version's explicit frame contract supplies its own axes,
  coefficient order, precision, and authority. `engine_evidence.py` loads those
  captures and attaches provenance before calling the shared normalizer.
- `scripts/evidence_schema/normalizer.py` checks coefficient orders, scene
  roles, and named frames against the serialized contract before it emits a
  `CrystallizedCapture`. The concrete serialization model has already rejected
  incompatible layers, contracts, and witness families. The normalizer also
  checks that signed basis maps imply the declared destination handedness and
  that witness precision appears in target metadata.
- `scripts/evidence_schema/product.py` assembles the normalized records and
  provenance. It does not reinterpret target axes or repair target values.

Native capture programs do not import the common normalizer. They run in their
pinned target environment, preserve native output, and identify the experiment
contract they used. The repository-side evidence pipeline validates that
contract and capture, imports the target semantic serializer, then calls the
common normalizer. This keeps target API access independent of normalization
while giving every product one explicit path from target evidence to the
common contract.

Supporting oracle packages such as Sophus and OpenXR also keep an
experiment-owned semantic config and validate it against their manifest hash.
Their dedicated validators consume the typed config, but those packages remain
outside the accepted-converter normalizer while their registry status says
supporting or declared evidence.

PROJ remains supporting frame evidence. Its raw observations are validated by
`scripts/evidence_schema/proj_capture.py` before they enter the generic capture
provenance envelope; vector and basis-endpoint dimensions are fixed by that
target schema.

`FrameSemantics` keeps a frame's role and the authority for that role separate
from its signed-axis values:

| Authority | Meaning |
| --- | --- |
| `target_documented` | A versioned target API or specification defines the named frame role; `evidence_reference` is required. |
| `application_declared` | The capture or consuming application explicitly assigns this frame and mapping. |
| `fixture_only` | The axes are selected to exercise numeric behavior and make no physical-frame claim. |
| `unspecified` | The evidence does not establish the frame role. |

For example, the Cesium serializer identifies the ENU frame as documented by
the pinned API while recording the QuatOps-to-ENU alignment as
application-declared. The current glTF hierarchy is serialized as a
fixture-only probe basis. That basis is not relabeled as an asset or camera
frame. OpenUSD's stage basis is recorded as an application declaration made by
the fixture.

## Recapture rule

Changing a semantic label does not require a new native run when the existing
raw observations, inputs, runtime identity, and provenance still support the
corrected claim. Keep the raw capture and change the serializer/contract layer
when only its interpretation or scope was wrong.

Create a new capture and provenance revision if the native API call, input
asset, target runtime, observed fields, or chosen frame assignment changes, or
if the raw capture cannot establish what the serializer needs. Never edit a
retained native value to make it fit the common schema; make the target
serializer explain the conversion, or mark the role unresolved.

JSON remains the interchange format for native runtimes. It is not accepted as
an untyped semantic object: the repository-side loader validates the JSON
against a target-specific strict model before the target serializer receives
it. The common output is a normalized view, not a replacement for native
evidence. Native capture validation, mathematical-oracle checks, and pinned
runtime validation remain separately reportable evidence.
