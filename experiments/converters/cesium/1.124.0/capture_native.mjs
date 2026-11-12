/** Execute QuatOps-generated inputs against CesiumJS 1.124 native APIs. */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import os from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { numeric, stringifyEvidence } from '../../shared/numeric_evidence.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const repositoryRoot = path.resolve(here, '../../../..');
const semanticContractPath = path.join(here, 'semantic_contract.json');
const inputPath = path.join(here, 'converter_inputs.json');
const [outputPath, packageRoot] = process.argv.slice(2);
if (!outputPath || !packageRoot) throw new Error('Usage: node capture_native.mjs OUTPUT_DIRECTORY CESIUM_DISTRIBUTION');
const bundlePath = path.join(packageRoot, 'index.cjs');
const C = await import(pathToFileURL(bundlePath));
if (fs.existsSync(outputPath) && fs.readdirSync(outputPath).length) throw new Error('Refusing to overwrite output directory');
fs.mkdirSync(outputPath, { recursive: true });
const input = JSON.parse(fs.readFileSync(inputPath, 'utf8'));
const sha = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const qv = q => [q.x, q.y, q.z, q.w];
const xyz = v => [v.x, v.y, v.z];
const quat = xyzw => new C.Quaternion(xyzw[0], xyzw[1], xyzw[2], xyzw[3]);
const points = input.probe_vectors_xyz.map(p => C.Cartesian3.fromArray(p));
const observations = [];
if (new Set(input.cases.map(item => item.case_id)).size !== input.cases.length) throw new Error('Duplicate Cesium case IDs');

for (const item of input.cases) {
  const origin = C.Cartesian3.fromDegrees(item.longitude_degrees, item.latitude_degrees, item.height_m, C.Ellipsoid.WGS84);
  const enuToEcef = C.Transforms.eastNorthUpToFixedFrame(origin, C.Ellipsoid.WGS84, new C.Matrix4());
  const localQ = quat(item.quatops_cesium_local_xyzw);
  const localR = C.Matrix3.fromQuaternion(localQ, new C.Matrix3());
  const localM = C.Matrix4.fromRotationTranslation(localR, C.Cartesian3.fromArray(item.local_translation_xyz_m), new C.Matrix4());
  const worldM = C.Matrix4.multiply(enuToEcef, localM, new C.Matrix4());
  const worldInverse = C.Matrix4.inverseTransformation(worldM, new C.Matrix4());
  const worldProbe = xyz(C.Matrix4.multiplyByPoint(worldM, points.at(-1), new C.Cartesian3()));
  const worldDirections = points.slice(0, item.case_id === 'mixed_a@geo-03' ? 4 : 3)
    .map(p => xyz(C.Matrix4.multiplyByPointAsVector(worldM, p, new C.Cartesian3())));
  const localProbe = xyz(C.Matrix4.multiplyByPoint(worldInverse, C.Cartesian3.fromArray(worldProbe), new C.Cartesian3()));

  const ecefRotation = C.Matrix4.getMatrix3(worldM, new C.Matrix3());
  const enuRotation = C.Matrix4.getMatrix3(enuToEcef, new C.Matrix3());
  const ecefQ = C.Quaternion.fromRotationMatrix(ecefRotation, new C.Quaternion());
  const enuQ = C.Quaternion.fromRotationMatrix(enuRotation, new C.Quaternion());
  const recoveredLocalQ = C.Quaternion.multiply(C.Quaternion.conjugate(enuQ, new C.Quaternion()), ecefQ, new C.Quaternion());

  const outputs = {
      model_to_ecef_quaternion_xyzw: numeric(qv(ecefQ)),
      local_quaternion_recovered_from_ecef_xyzw: numeric(qv(recoveredLocalQ)),
      world_points_xyz_m: numeric([worldProbe]),
      world_directions_xyz: numeric(worldDirections),
      point_roundtrip_local_xyz_m: numeric([localProbe]),
      ecef_origin_xyz_m: numeric(xyz(origin)),
  };
  if (item.case_id === 'mixed_a@geo-03') {
    outputs.local_basis_and_probe_images = numeric(points.map(p => xyz(C.Matrix3.multiplyByVector(localR, p, new C.Cartesian3()))));
  }
  observations.push({case_id: item.case_id, outputs});
}

const rowsPath = path.join(outputPath, 'witnesses.json');
fs.writeFileSync(rowsPath, stringifyEvidence({schema_version: '1.0', evidence_kind: 'native_cesium_quatops_converter_capture', observations}));
const pinPath = path.join(here, 'distribution.json');
const pin = JSON.parse(fs.readFileSync(pinPath, 'utf8'));
const packageInfo = JSON.parse(fs.readFileSync(path.join(packageRoot, 'package.json'), 'utf8'));
if (packageInfo.name !== pin.name || packageInfo.version !== pin.version || C.VERSION !== '1.124') {
  throw new Error('Cesium distribution differs from the pinned release');
}
const exporterPath = fileURLToPath(import.meta.url);
const manifest = {
  schema_version: '1.0', evidence_kind: 'native_cesium_quatops_converter_capture',
  semantic_contract: { file: path.relative(repositoryRoot, semanticContractPath).split(path.sep).join('/'), sha256: sha(semanticContractPath) },
  target: { name: 'CesiumJS', package_version: packageInfo.version, api_version: C.VERSION, package_entry_sha256: sha(bundlePath), distribution_integrity: pin.integrity },
  execution: { was_run: true, utc: new Date().toISOString(), node: process.version, v8: process.versions.v8, platform: os.platform(), architecture: os.arch() },
  inputs: { path: path.relative(here, inputPath), sha256: sha(inputPath), kind: 'QuatOps-generated explicit-frame converter inputs' },
  exporter: { path: path.relative(here, exporterPath), sha256: sha(exporterPath) },
  distribution_sha256: sha(pinPath),
  distribution_installer_sha256: sha(path.join(here, '../distribution.py')),
  numeric_helper_sha256: sha(path.resolve(here, '../../shared/numeric_evidence.mjs')),
  observations: { path: 'witnesses.json', sha256: sha(rowsPath), count: observations.length },
  output: path.resolve(outputPath),
};
fs.writeFileSync(path.join(outputPath, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
console.log(JSON.stringify({ output: outputPath, observations: observations.length, cesium: C.VERSION, node: process.version }));
