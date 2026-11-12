/** Observe authored geometry through the pinned GLTFLoader; do not infer frame roles. */
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import os from 'node:os';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const packageRoot = path.resolve(here, '../threejs/0.187.0');
const packageFile = path.join(packageRoot, 'node_modules/three/package.json');
const pkg = JSON.parse(fs.readFileSync(packageFile, 'utf8'));
if (pkg.version !== '0.187.0') throw new Error('Expected Three.js 0.187.0');
const THREE = await import(pathToFileURL(path.join(packageRoot, 'node_modules/three/build/three.module.js')));
const { GLTFLoader } = await import(pathToFileURL(path.join(packageRoot, 'node_modules/three/examples/jsm/loaders/GLTFLoader.js')));
// Node's fetch supports data URIs but does not supply browser ProgressEvent.
if (!globalThis.ProgressEvent) globalThis.ProgressEvent = class ProgressEvent {
  constructor(type, init) { this.type = type; Object.assign(this, init); }
};
const output = process.argv[2];
if (!output) throw new Error('Usage: node capture.mjs EMPTY_OUTPUT_DIRECTORY');
if (fs.existsSync(output) && fs.readdirSync(output).length) throw new Error('Refusing overwrite');
const assetPath = path.join(here, 'oriented_mesh.gltf');
const loader = new GLTFLoader();
const gltf = await loader.parseAsync(fs.readFileSync(assetPath, 'utf8'), '');
gltf.scene.updateMatrixWorld(true);
const mesh = gltf.scene.getObjectByName('OrientedMesh');
if (!mesh?.isMesh) throw new Error('Expected imported Mesh, not marker-only node');
const positions = mesh.geometry.getAttribute('position');
const vertices = [];
for (let i = 0; i < positions.count; i++) {
  const point = new THREE.Vector3().fromBufferAttribute(positions, i);
  vertices.push({ index: i, local_xyz: point.toArray(),
    parent_xyz: point.clone().applyMatrix4(mesh.matrix).toArray(),
    world_xyz: mesh.localToWorld(point.clone()).toArray() });
}
const observations = { schema_version: '1.0',
  evidence_kind: 'native_threejs_authored_mesh_capture',
  node: mesh.name, parent: mesh.parent.name, geometry_type: mesh.geometry.type,
  indices: Array.from(mesh.geometry.index.array),
  local_quaternion_xyzw: mesh.quaternion.toArray(),
  parent_quaternion_xyzw: mesh.parent.quaternion.toArray(),
  vertices };
const sha = p => crypto.createHash('sha256').update(fs.readFileSync(p)).digest('hex');
fs.mkdirSync(output, {recursive:true});
const witness = path.join(output, 'witnesses.json');
fs.writeFileSync(witness, JSON.stringify(observations, null, 2)+'\n');
const manifest = { schema_version:'1.0', evidence_kind:observations.evidence_kind,
 target:{name:'Three.js GLTFLoader',version:pkg.version,revision:THREE.REVISION},
 execution:{was_run:true,utc:new Date().toISOString(),node:process.version,
   platform:os.platform(),architecture:os.arch()},
 scope:'Authored asymmetric mesh vertices transformed through static parent-child hierarchy',
 authority:'application_declared',
 excludes:['Arbitrary mesh-front inference','Physical asset calibration','Animation','Other loader versions'],
 provenance:{asset_sha256:sha(assetPath),capture_sha256:sha(fileURLToPath(import.meta.url)),
   package_lock_sha256:sha(path.join(packageRoot,'package-lock.json')),
   witnesses_sha256:sha(witness)} };
fs.writeFileSync(path.join(output,'provenance.json'),JSON.stringify(manifest,null,2)+'\n');
console.log(JSON.stringify({output,version:pkg.version,node:process.version,vertices:vertices.length}));