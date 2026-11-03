import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { basename, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';
import process from 'node:process';
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

const require = createRequire(import.meta.url);
const validator = require('gltf-validator');
const SCRIPT_PATH = fileURLToPath(import.meta.url);
const SCRIPT_DIR = resolve(SCRIPT_PATH, '..');
const ASSET_DIR = resolve(SCRIPT_DIR, '../../assets');
const DEFAULT_ASSET = resolve(ASSET_DIR, 'rotation_transport.gltf');
const DEFAULT_OUTPUT = resolve(SCRIPT_DIR, '../../../../../../tests/golden/transports/gltf/2.0.0/threejs/0.187.0/loader_rotation.json');

// Node.js does not currently provide the browser ProgressEvent constructor.
// Three.js FileLoader uses it while resolving data URIs during GLTFLoader parse.
if (typeof globalThis.ProgressEvent === 'undefined') {
  globalThis.ProgressEvent = class ProgressEvent extends Event {
    constructor(type, init = {}) {
      super(type);
      this.lengthComputable = init.lengthComputable ?? false;
      this.loaded = init.loaded ?? 0;
      this.total = init.total ?? 0;
    }
  };
}

function args(argv) {
  const result = { asset: DEFAULT_ASSET, output: DEFAULT_OUTPUT };
  for (let i = 2; i < argv.length; i += 2) {
    const key = argv[i];
    const value = argv[i + 1];
    if (!value || !['--asset', '--output'].includes(key)) {
      throw new Error(`Expected --asset PATH or --output PATH; received ${key}`);
    }
    result[key.slice(2)] = resolve(value);
  }
  return result;
}

function sha256(data) {
  return createHash('sha256').update(data).digest('hex');
}

function matrix4(m) {
  return {
    elements_column_major: [...m.elements],
    rows: Array.from({ length: 4 }, (_, r) =>
      Array.from({ length: 4 }, (_, c) => m.elements[c * 4 + r]),
    ),
  };
}

function objectTransform(object) {
  object.updateMatrix();
  object.updateWorldMatrix(true, false);
  return {
    position_xyz: object.position.toArray(),
    rotation_xyzw: object.quaternion.toArray(),
    scale_xyz: object.scale.toArray(),
    local_matrix: matrix4(object.matrix),
    world_matrix: matrix4(object.matrixWorld),
  };
}

function animationSamples(gltf, clip) {
  const track = clip.tracks[0];
  const keyTimes = Array.from(track.times);
  const [t0, t1, t2] = keyTimes;
  const epsilon = Math.min(1e-6, (t1 - t0) / 1000, (t2 - t1) / 1000);
  const samples = [
    { sample: 'key-0', time_seconds: t0 },
    { sample: 'midpoint-0-1', time_seconds: (t0 + t1) / 2 },
    { sample: 'before-key-1', time_seconds: t1 - epsilon },
    { sample: 'key-1', time_seconds: t1 },
    { sample: 'after-key-1', time_seconds: t1 + epsilon },
    { sample: 'midpoint-1-2', time_seconds: (t1 + t2) / 2 },
    { sample: 'key-2', time_seconds: t2 },
  ];
  const mixer = new THREE.AnimationMixer(gltf.scene);
  const action = mixer.clipAction(clip);
  action.setLoop(THREE.LoopOnce, 0);
  action.clampWhenFinished = true;
  action.play();
  const targetName = clip.name.replace('rotation_', 'animated_');
  const target = gltf.scene.getObjectByName(targetName);
  if (!target) throw new Error(`Missing target node ${targetName} for ${clip.name}`);
  const rows = samples.map((sample) => {
    mixer.setTime(sample.time_seconds);
    return {
      ...sample,
      quaternion_xyzw: target.quaternion.toArray(),
    };
  });
  mixer.stopAllAction();
  return {
    animation_name: clip.name,
    track_name: track.name,
    key_times_seconds: keyTimes,
    track_values_xyzw: Array.from(track.values),
    samples: rows,
  };
}

async function main() {
  const options = args(process.argv);
  const gltfBytes = await readFile(options.asset);
  const gltfText = gltfBytes.toString('utf8');
  const source = JSON.parse(gltfText);
  const binaryUri = source.buffers[0].uri;
  const binaryPath = resolve(ASSET_DIR, binaryUri);
  const binaryBytes = await readFile(binaryPath);
  const validationReport = await validator.validateString(gltfText, {
    uri: basename(options.asset),
    writeTimestamp: false,
    externalResourceFunction: async (uri) => {
      if (uri !== binaryUri) throw new Error(`Unexpected external URI ${uri}`);
      return new Uint8Array(binaryBytes);
    },
  });
  const embedded = {
    ...source,
    buffers: [{
      ...source.buffers[0],
      uri: `data:application/octet-stream;base64,${binaryBytes.toString('base64')}`,
    }],
  };
  const adaptedText = JSON.stringify(embedded);
  const gltf = await new GLTFLoader().parseAsync(adaptedText, '');
  gltf.scene.updateMatrixWorld(true);

  const staticTransforms = {};
  for (const name of ['parent_frame', 'static_trs', 'static_matrix']) {
    const object = gltf.scene.getObjectByName(name);
    if (!object) throw new Error(`Missing static node ${name}`);
    staticTransforms[name] = objectTransform(object);
  }
  for (const name of ['source_axis_origin', 'source_axis_x', 'source_axis_y', 'source_axis_z']) {
    const object = gltf.scene.getObjectByName(name);
    if (!object) throw new Error(`Missing axis probe ${name}`);
    staticTransforms[name] = objectTransform(object);
  }
  const animationRows = gltf.animations.map((clip) => animationSamples(gltf, clip));
  const issues = validationReport.issues ?? {};

  const lockBytes = await readFile(resolve(SCRIPT_DIR, 'package-lock.json'));
  const scriptBytes = await readFile(SCRIPT_PATH);
  const result = {
    schema_version: '1.0',
    evidence_kind: 'native_gltf_loader_capture',
    transport: {
      name: 'glTF',
      version: '2.0',
      container: 'JSON glTF with external binary buffer',
      rotation_components: 'source accessor order is XYZW',
    },
    loader: {
      name: 'Three.js GLTFLoader',
      package_version: '0.187.0',
      three_revision: THREE.REVISION,
      node_version: process.version,
      v8_version: process.versions.v8,
      platform: process.platform,
      architecture: process.arch,
      numeric_representation: 'JavaScript Number (binary64) after float32 asset decoding',
    },
    provenance: {
      gltf_sha256: sha256(gltfBytes),
      binary_sha256: sha256(binaryBytes),
      adapted_gltf_sha256: sha256(adaptedText),
      package_lock_sha256: sha256(lockBytes),
      exporter_sha256: sha256(scriptBytes),
      validator_version: validator.version(),
      exact_invocation: 'node capture_loader.mjs --asset <rotation_transport.gltf> --output <loader_rotation.json>',
      captured_at_utc: new Date().toISOString(),
      loader_buffer_adaptation: 'original external buffer bytes were embedded as a data URI in memory; byte payload unchanged',
    },
    validation: {
      evidence_kind: 'Khronos glTF Validator report; format/schema validation only',
      validator_version: validator.version(),
      asset_version: validationReport.asset?.version,
      summary: validationReport.issues?.numMessages,
      num_errors: issues.numErrors,
      num_warnings: issues.numWarnings,
      num_infos: issues.numInfos,
      num_hints: issues.numHints,
      messages: issues.messages ?? [],
    },
    source_asset: {
      file: basename(options.asset),
      external_buffer_file: basename(binaryPath),
      buffer_byte_length: binaryBytes.length,
      binary_buffer_data_uri_used_for_loader: true,
      animation_interpolation_modes: source.animations.map(
        (animation) => animation.samplers[0].interpolation,
      ),
      animation_target_nodes: source.animations.map(
        (animation) => source.nodes[animation.channels[0].target.node].name,
      ),
    },
    observations: {
      static_transforms: staticTransforms,
      animations: animationRows,
    },
  };
  await writeFile(options.output, `${JSON.stringify(result, null, 2)}\n`, 'utf8');
  process.stdout.write(`${JSON.stringify({
    output: options.output,
    asset_sha256: result.provenance.gltf_sha256,
    validator_messages: result.validation.summary,
    animation_count: animationRows.length,
  })}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error.stack ?? error}\n`);
  process.exitCode = 1;
});
