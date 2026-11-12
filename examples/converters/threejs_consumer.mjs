// Observe authored, asymmetric forward/up markers through native Object3D APIs.
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const moduleUrl = pathToFileURL(process.argv[2]);
const { Object3D, Vector3 } = await import(moduleUrl.href);
const { version } = JSON.parse(readFileSync(new URL("../package.json", moduleUrl), "utf8"));
if (version !== "0.187.0") throw new Error(`Expected Three.js 0.187.0, got ${version}`);
const input = JSON.parse(readFileSync(0, "utf8"));

function observe(coefficients) {
  const arrow = new Object3D();
  arrow.position.set(1.5, -2, 0.25);
  arrow.quaternion.fromArray(coefficients);
  const forward = new Object3D();
  forward.position.fromArray(input.forward_tip_xyz);
  const up = new Object3D();
  up.position.fromArray(input.up_tip_xyz);
  arrow.add(forward, up);
  arrow.updateMatrixWorld(true);
  const tail = arrow.getWorldPosition(new Vector3());
  return {
    coefficients: arrow.quaternion.toArray(),
    forward: forward.getWorldPosition(new Vector3()).sub(tail).normalize().toArray(),
    up: up.getWorldPosition(new Vector3()).sub(tail).normalize().toArray(),
  };
}

const native = observe(input.quaternion_xyzw);
// This explicit storage cast is separate from Three.js's ordinary Number API.
const rounded = observe(new Float32Array(input.quaternion_xyzw));
console.log(JSON.stringify({
  version,
  quaternion_xyzw: native.coefficients,
  forward_xyz: native.forward,
  up_xyz: native.up,
  float32_quaternion_xyzw: rounded.coefficients,
  float32_forward_xyz: rounded.forward,
  float32_up_xyz: rounded.up,
}));
