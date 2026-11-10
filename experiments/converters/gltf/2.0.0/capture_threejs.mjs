/** Load the QuatOps-generated glTF asset with pinned Three.js GLTFLoader. */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import os from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { numeric, stringifyEvidence } from '../../shared/numeric_evidence.mjs';

const here = path.dirname(fileURLToPath(import.meta.url));
const repositoryRoot = path.resolve(here, '../../../..');
const semanticContractPath = path.join(here, 'semantic_contract.json');
const packageRoot = path.resolve(here, '../../../transports/gltf/2.0.0/threejs/0.187.0');
const THREE = await import(pathToFileURL(path.join(packageRoot, 'node_modules/three/build/three.module.js')));
const { GLTFLoader } = await import(pathToFileURL(path.join(packageRoot, 'node_modules/three/examples/jsm/loaders/GLTFLoader.js')));
const assetPath = path.join(here, 'quatops_hierarchy.gltf');
const converterInputs = JSON.parse(fs.readFileSync(path.join(here, 'converter_inputs.json'), 'utf8'));
const outputPath = process.argv[2];
if (!outputPath) throw new Error('Usage: node capture_threejs.mjs OUTPUT_DIRECTORY');
if (fs.existsSync(outputPath) && fs.readdirSync(outputPath).length) throw new Error('Refusing to overwrite output directory');
fs.mkdirSync(outputPath, { recursive: true });
const sha = file => crypto.createHash('sha256').update(fs.readFileSync(file)).digest('hex');
const asset = JSON.parse(fs.readFileSync(assetPath, 'utf8'));
const loader = new GLTFLoader();
const gltf = await new Promise((resolve, reject) => loader.parse(JSON.stringify(asset), '', resolve, reject));
const scene = gltf.scene;
scene.updateMatrixWorld(true);
const vectors = converterInputs.gltf_probe_vectors_xyz;
const rows = matrix => Array.from({length:4},(_,r)=>Array.from({length:4},(_,c)=>matrix.elements[c*4+r]));
const transformProbe = (object, source) => new THREE.Vector3(...source).applyMatrix4(object.matrixWorld).toArray();
const directionProbe = (object, source) => {
  const value = new THREE.Vector3(...source).transformDirection(object.matrixWorld);
  // Direction probe uses the rotation alone and is normalized by transformDirection.
  return value.toArray();
};
const observations = [];
for (const name of ['QuatOpsParent', 'QuatOpsChild']) {
  const object = scene.getObjectByName(name);
  const localImages = vectors.map(source => new THREE.Vector3(...source).applyQuaternion(object.quaternion).toArray());
  const worldPoints = vectors.map(source => transformProbe(object, source));
  const worldDirections = vectors.map(source => directionProbe(object, source));
  const inverse = object.matrixWorld.clone().invert();
  const roundtrip = worldPoints.map(source => new THREE.Vector3(...source).applyMatrix4(inverse).toArray());
  observations.push({
    node: name,
    native_entry_points: ['GLTFLoader.parse', 'Object3D.updateMatrixWorld', 'Object3D.localToWorld', 'Matrix4.invert'],
    outputs: {
      local_quaternion_xyzw: numeric(object.quaternion.toArray()),
      local_matrix_rows: numeric(rows(object.matrix)),
      world_matrix_rows: numeric(rows(object.matrixWorld)),
      local_basis_and_probe_images: numeric(localImages),
      world_points_xyz: numeric(worldPoints),
      world_unit_directions_xyz: numeric(worldDirections),
      world_to_local_point_roundtrip_xyz: numeric(roundtrip),
    },
  });
}
const outputFile = path.join(outputPath, 'observations.json');
fs.writeFileSync(outputFile, stringifyEvidence(observations));
const packageJson = JSON.parse(fs.readFileSync(path.join(packageRoot, 'node_modules/three/package.json'), 'utf8'));
const manifest = {
  schema_version: '1.0', evidence_kind: 'native_threejs_gltf_converter_capture',
  semantic_contract: { file: path.relative(repositoryRoot, semanticContractPath).split(path.sep).join('/'), sha256: sha(semanticContractPath) },
  target: { name: 'Three.js GLTFLoader', package_version: packageJson.version, revision: THREE.REVISION },
  execution: { was_run: true, utc: new Date().toISOString(), node: process.version, v8: process.versions.v8, platform: os.platform(), architecture: os.arch() },
  asset: { path: path.basename(assetPath), sha256: sha(assetPath), input_sha256: sha(path.join(here, 'converter_inputs.json')) },
  exporter: { path: path.basename(fileURLToPath(import.meta.url)), sha256: sha(fileURLToPath(import.meta.url)) },
  package_lock_sha256: sha(path.join(packageRoot, 'package-lock.json')),
  numeric_helper_sha256: sha(path.resolve(here, '../../shared/numeric_evidence.mjs')),
  observations: { path: 'observations.json', sha256: sha(outputFile), count: observations.length },
};
fs.writeFileSync(path.join(outputPath, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
console.log(JSON.stringify({output:outputPath, observations:observations.length, three:packageJson.version, node:process.version}));
