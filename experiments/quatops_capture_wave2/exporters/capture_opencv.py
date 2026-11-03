"""Execute OpenCV Rodrigues, preserving raw float32/float64 outputs and Jacobians."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from shared.capture_io import Capture, Observation, contract, numeric, parser


def main() -> None:
    args = parser(__doc__ or "OpenCV capture").parse_args()
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    binary = next(iter(Path(cv2.__file__).parent.glob("*.so")), None)
    capture = Capture("opencv_rodrigues", "OpenCV", cv2.__version__, args.corpus, args.output,
        Path(__file__), {"input": "binary32 and binary64, per record", "arithmetic": "vendor implementation; build information retained",
        "output": "binary32 and binary64, per record", "serialized": "JSON decimal plus big-endian IEEE-754 bits"},
        dependencies=[{"name": "numpy", "version": np.__version__}], native_binary=binary,
        limitations=["Only cv2.Rodrigues; not C++ cv::Quat algebra.", "No QuatOps checkout or representation replay.",
            "Matrix-to-rotvec at zero/pi may use native thresholds; raw results are not repaired."])
    build = args.output / "opencv-build-information.txt"
    build.write_text(cv2.getBuildInformation(), encoding="utf-8")
    capture.artifacts.append((build, "native_build_information"))
    vector_inputs = [(item["case_id"], np.asarray(item["axis_xyz"], dtype=np.float64) * item["angle_radians"])
                     for item in corpus["axis_angle_inputs"]]
    rng = np.random.default_rng(0xC4A2026)
    for i in range(64):
        vector_inputs.append((f"rodrigues-random-{i:03d}", rng.uniform(-3.0, 3.0, size=3)))
    for dtype, label in ((np.float32, "binary32"), (np.float64, "binary64")):
        for case_id, value in vector_inputs:
            rv = np.asarray(value, dtype=dtype).reshape(3, 1)
            matrix, jacobian = cv2.Rodrigues(rv)
            back, back_jacobian = cv2.Rodrigues(matrix)
            reconstructed, _ = cv2.Rodrigues(back)
            capture.add(Observation(case_id, f"rodrigues-forward-{label}", "cv2.Rodrigues", contract(),
                {"rotation_vector": numeric(rv.tolist(), label)},
                {"matrix_rows": numeric(matrix.tolist(), label), "jacobian_native": numeric(jacobian.tolist(), label),
                 "readback_rotvec": numeric(back.tolist(), label), "inverse_jacobian_native": numeric(back_jacobian.tolist(), label),
                 "reconstructed_matrix_rows": numeric(reconstructed.tolist(), label)}))
        for item in corpus["matrix_inputs"]:
            if item["domain"] != "proper_rotation":
                capture.add(Observation(item["case_id"], f"rodrigues-matrix-{label}", "cv2.Rodrigues", contract(),
                    {"matrix_rows": numeric(item["rows"], label)}, {}, "not_run", "Invalid rotation probes excluded from valid-input baseline"))
                continue
            matrix = np.asarray(item["rows"], dtype=dtype)
            rv, jacobian = cv2.Rodrigues(matrix)
            back, _ = cv2.Rodrigues(rv)
            capture.add(Observation(item["case_id"], f"rodrigues-matrix-{label}", "cv2.Rodrigues", contract(),
                {"matrix_rows": numeric(matrix.tolist(), label)},
                {"rotation_vector": numeric(rv.tolist(), label), "jacobian_native": numeric(jacobian.tolist(), label),
                 "reconstructed_matrix_rows": numeric(back.tolist(), label)}))
    print(capture.finish(["MATRIX", "AXIS_ROTVEC", "DIFFERENTIAL"]))


if __name__ == "__main__":
    main()
