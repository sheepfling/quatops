# Historical capture source bundle

The exact source bytes used to produce the retained historical captures are
kept in [`../reproduction-v1.zip`](../reproduction-v1.zip). Evidence metadata
references a member as `experiments/provenance/reproduction-v1.zip#<member>`;
the member's SHA-256 remains the source identity recorded by the capture.

The archive groups historical standalone `.py.txt` and `.cpp.txt` snapshots
without changing their contents, including the original Cesium npm manifest,
lockfile, and later input-generator snapshot. New reproduction sources belong beside their experiment and should
use a typed capture model at the export boundary.
