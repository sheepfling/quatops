# Long goal: complete target-contract refinement

**Status:** Implementation complete for the currently registered target set
**Updated:** 2026-11-12

## Goal

Make the evidence boundary explicit and reviewable from native capture through
the common semantic contract. A target serializer must not infer storage,
frame, action, or hierarchy meaning from loosely structured text or from a
product name.

## Completion criteria

- Interpretation selectors have target-specific literals or finite typed
  vocabularies: axis meanings and directions, handedness, quaternion order,
  actions, scalar profiles and tolerances, signed-axis maps, native entry
  points, node roles, and transform operations.
- Target identity, version, unit scale, and serialized orientation field are
  retained and checked against capture provenance where applicable.
- Generated input frames and maps agree with both the semantic contract and
  the shared canonical corpus. Mapped probes, translations, and quaternion
  coefficients follow the declared map.
- For engine, direct-quaternion, and OpenCV profiles, captured actions on basis
  vectors and selected probes agree with the captured representation and
  declared target frame.
- Each serializer returns a concrete validated semantic product; explanation
  text remains descriptive and does not select a conversion path.
- Semantic-contract hashes in retained manifests match the edited contracts,
  and the audit distinguishes closed corrections from optional evidence
  expansions.

## Completed in this pass

- Typed frame-axis meanings, forward directions, up-axis labels, target
  identities, pinned API lists, node names, transform-op names, and positive
  stage unit scales. Engine and converter profile names, API identifiers,
  scalar profiles, and tolerance domains are finite and validated.
- Split Cesium quaternion storage, geodetic datum, and units into explicit
  fields; split glTF format/loader identity and coefficient orders; made the
  OpenUSD source frame, orientation op, target identity, and stage units
  explicit.
- Added cross-checks for canonical frames, target frames, signed-axis maps,
  probe vectors, target translations and quaternion inputs, engine and direct
  quaternion coefficients and captured vector actions, OpenCV rotation-vector
  round trips and matrix actions, ROS lookup/message identity and numeric
  width, OpenUSD stage metadata and stack/path/reset observations, and modeled
  target identity fields shared by the semantic contract, native capture, and
  manifest.
- Preserved retained native observations and refreshed the Cesium, glTF,
  Blender, OpenCV, OpenUSD, MuJoCo, and SpiceyPy semantic-contract digests in
  their manifests.
- Corrected the audit's stale next-work list and documented its current
  dispositions.

## Explicitly outside this goal

The current contracts do not claim native OpenXR VIEW or LOCAL_FLOOR runtime
behavior, Godot Camera3D node action, or mesh-specific glTF asset-front
behavior. Those require new matching fixtures and native captures only if those
stronger claims become part of the requested scope.

## Follow-up native qualification

Rick subsequently requested the excluded native gaps be addressed. Supplemental
Camera3D, authored glTF mesh, and OpenXR VIEW/LOCAL_FLOOR static-offset captures
now exist. See section 7 of [the audit](SEMANTIC_TARGET_CONTRACT_AUDIT.md) and the
registry for exact boundaries. No physical calibration, arbitrary mesh
alignment, real headset/floor behavior, or QuatOps adapter acceptance is implied.
