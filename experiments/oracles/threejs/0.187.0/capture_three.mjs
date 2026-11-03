import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { basename, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import process from 'node:process';
import * as THREE from 'three';

const SCRIPT_PATH = fileURLToPath(import.meta.url);
const SCRIPT_DIR = resolve(SCRIPT_PATH, '..');
const ORDERS = ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX'];
const DEFAULT_CORPUS = resolve(SCRIPT_DIR, '../../shared/orientation_inputs_v1.json');
const DEFAULT_OUTPUT = resolve(SCRIPT_DIR, '../../../../tests/golden/oracles/threejs/0.187.0/native_rotation.json');

function args(argv) {
  const result = { corpus: DEFAULT_CORPUS, output: DEFAULT_OUTPUT };
  for (let i = 2; i < argv.length; i += 2) {
    const key = argv[i];
    const value = argv[i + 1];
    if (!value || !['--corpus', '--output'].includes(key)) {
      throw new Error(`Expected --corpus PATH or --output PATH; received ${key}`);
    }
    result[key.slice(2)] = resolve(value);
  }
  return result;
}

function sha256(buffer) {
  return createHash('sha256').update(buffer).digest('hex');
}

function xyzw(q) {
  return [q.x, q.y, q.z, q.w];
}

function matrixRows(m) {
  const e = m.elements;
  return Array.from({ length: 3 }, (_, r) =>
    Array.from({ length: 3 }, (_, c) => e[c * 4 + r]),
  );
}

function rotateBasis(q) {
  return [
    new THREE.Vector3(1, 0, 0).applyQuaternion(q).toArray(),
    new THREE.Vector3(0, 1, 0).applyQuaternion(q).toArray(),
    new THREE.Vector3(0, 0, 1).applyQuaternion(q).toArray(),
  ];
}

function matrixFromRows(rows) {
  return new THREE.Matrix4().set(
    rows[0][0], rows[0][1], rows[0][2], 0,
    rows[1][0], rows[1][1], rows[1][2], 0,
    rows[2][0], rows[2][1], rows[2][2], 0,
    0, 0, 0, 1,
  );
}

function eulerObservation(caseId, family, inputXyz, order, source = null) {
  const e = new THREE.Euler(inputXyz[0], inputXyz[1], inputXyz[2], order);
  const q = new THREE.Quaternion().setFromEuler(e);
  const m = new THREE.Matrix4().makeRotationFromEuler(e);
  const readback = new THREE.Euler().setFromQuaternion(q, order);
  const roundtrip = new THREE.Quaternion().setFromEuler(readback);
  const matrixQuaternion = new THREE.Quaternion().setFromRotationMatrix(m);
  return {
    case_id: caseId,
    family,
    order,
    input_radians_xyz: [...inputXyz],
    input_source: source,
    euler_quaternion_xyzw: xyzw(q),
    euler_matrix_rows: matrixRows(m),
    quaternion_rotated_positive_basis_vectors_xyz: rotateBasis(q),
    matrix_rotated_positive_basis_vectors_xyz: [
      new THREE.Vector3(1, 0, 0).applyMatrix4(m).toArray(),
      new THREE.Vector3(0, 1, 0).applyMatrix4(m).toArray(),
      new THREE.Vector3(0, 0, 1).applyMatrix4(m).toArray(),
    ],
    quaternion_readback_roundtrip_xyzw: xyzw(roundtrip),
    matrix_quaternion_xyzw: xyzw(matrixQuaternion),
  };
}

function makeEngine(corpus) {
  const generalEuler = [];
  for (const input of corpus.euler_inputs) {
    for (const order of ORDERS) {
      generalEuler.push(eulerObservation(input.case_id, input.family, input.xyz, order, {
        numeric_format: input.numeric_format,
        units: input.units,
        actual_bits_hex: input.actual_bits_hex,
      }));
    }
  }
  const boundaryEuler = corpus.euler_boundaries.map((input) => {
    const radians = input.units === 'degrees'
      ? input.xyz.map(THREE.MathUtils.degToRad)
      : [...input.xyz];
    return eulerObservation(input.case_id, input.family, radians, input.order_label, {
      numeric_format: input.numeric_format,
      units: input.units,
      input_xyz: input.xyz,
      actual_bits_hex: input.actual_bits_hex,
    });
  });

  const axisAngle = corpus.axis_angle_inputs.map((input) => {
    const q = new THREE.Quaternion().setFromAxisAngle(
      new THREE.Vector3(...input.axis_xyz), input.angle_radians,
    );
    const extractedAxis = new THREE.Vector3();
    const extractedAngle = 2 * Math.acos(THREE.MathUtils.clamp(q.w, -1, 1));
    const sine = Math.sqrt(Math.max(0, 1 - q.w * q.w));
    if (sine > 1e-12) extractedAxis.set(q.x / sine, q.y / sine, q.z / sine);
    else extractedAxis.set(1, 0, 0);
    return {
      case_id: input.case_id,
      input_axis_xyz: [...input.axis_xyz],
      input_angle_radians: input.angle_radians,
      quaternion_xyzw: xyzw(q),
      reconstructed_matrix_rows: matrixRows(new THREE.Matrix4().makeRotationFromQuaternion(q)),
      extracted_axis_xyz: extractedAxis.toArray(),
      extracted_angle_radians: extractedAngle,
    };
  });

  const quaternionDomains = corpus.quaternion_inputs.map((input) => {
    const [w, x, y, z] = input.wxyz;
    const q = new THREE.Quaternion(x, y, z, w);
    const normalized = q.clone().normalize();
    return {
      case_id: input.case_id,
      domain: input.domain,
      input_wxyz: [...input.wxyz],
      input_xyzw: xyzw(q),
      length: q.length(),
      length_squared: q.lengthSq(),
      normalized_xyzw: xyzw(normalized),
    };
  });

  const products = corpus.multiplication_pairs.map((input) => {
    const [aw, ax, ay, az] = input.a_wxyz;
    const [bw, bx, by, bz] = input.b_wxyz;
    const a = new THREE.Quaternion(ax, ay, az, aw);
    const b = new THREE.Quaternion(bx, by, bz, bw);
    return {
      case_id: input.case_id,
      a_input_wxyz: [...input.a_wxyz],
      b_input_wxyz: [...input.b_wxyz],
      a_multiply_b_xyzw: xyzw(a.clone().multiply(b)),
      b_multiply_a_xyzw: xyzw(b.clone().multiply(a)),
    };
  });

  const interpolationWeights = corpus.interpolation_weights;
  const interpolation = corpus.interpolation_pairs.map((input) => {
    const [aw, ax, ay, az] = input.a_wxyz;
    const [bw, bx, by, bz] = input.b_wxyz;
    const a = new THREE.Quaternion(ax, ay, az, aw);
    const b = new THREE.Quaternion(bx, by, bz, bw);
    return {
      case_id: input.case_id,
      family: input.family,
      a_input_wxyz: [...input.a_wxyz],
      b_input_wxyz: [...input.b_wxyz],
      samples: interpolationWeights.map((t) => ({
        t,
        quaternion_xyzw: xyzw(a.clone().slerp(b, t)),
      })),
    };
  });

  const matrices = corpus.matrix_inputs.map((input) => {
    const m = matrixFromRows(input.rows);
    const q = new THREE.Quaternion().setFromRotationMatrix(m);
    return {
      case_id: input.case_id,
      domain: input.domain,
      input_matrix_rows: input.rows,
      matrix_to_quaternion_xyzw: xyzw(q),
      quaternion_reconstructed_matrix_rows: matrixRows(
        new THREE.Matrix4().makeRotationFromQuaternion(q),
      ),
      quaternion_rotated_positive_basis_vectors_xyz: rotateBasis(q),
    };
  });

  const vectorInputs = corpus.vectors_xyz.map((v) => new THREE.Vector3(...v));
  const twoVector = [];
  for (let i = 0; i < vectorInputs.length; i += 1) {
    for (let j = 0; j < vectorInputs.length; j += 1) {
      if (i === j || vectorInputs[i].lengthSq() === 0 || vectorInputs[j].lengthSq() === 0) continue;
      const from = vectorInputs[i].clone().normalize();
      const to = vectorInputs[j].clone().normalize();
      const q = new THREE.Quaternion().setFromUnitVectors(from, to);
      twoVector.push({
        from_case_index: i,
        to_case_index: j,
        from_unit_xyz: from.toArray(),
        to_unit_xyz: to.toArray(),
        quaternion_xyzw: xyzw(q),
        rotated_from_xyz: from.clone().applyQuaternion(q).toArray(),
      });
    }
  }

  const longEuler = new THREE.Euler(2 * Math.PI + 0.3, -0.4, 0.7, 'XYZ');
  const beforeReorder = new THREE.Quaternion().setFromEuler(longEuler);
  longEuler.reorder('ZYX');
  const afterReorder = new THREE.Quaternion().setFromEuler(longEuler);

  const parent = new THREE.Object3D();
  parent.name = 'parent';
  parent.position.set(1.25, -2.5, 0.75);
  parent.quaternion.setFromEuler(new THREE.Euler(0.31, -0.52, 0.23, 'XYZ'));
  const child = new THREE.Object3D();
  child.name = 'child';
  child.position.set(-0.4, 1.1, 2.2);
  child.quaternion.setFromEuler(new THREE.Euler(-0.18, 0.39, 0.61, 'ZYX'));
  parent.add(child);
  parent.updateMatrixWorld(true);

  return {
    euler_orders: generalEuler,
    euler_boundaries: boundaryEuler,
    axis_angle: axisAngle,
    quaternion_normalization: quaternionDomains,
    multiplication: products,
    shortest_slerp: interpolation,
    independent_matrices: matrices,
    two_vector_alignment: twoVector,
    euler_reorder: {
      source_order: 'XYZ',
      target_order: 'ZYX',
      input_radians_xyz: [2 * Math.PI + 0.3, -0.4, 0.7],
      source_quaternion_xyzw: xyzw(beforeReorder),
      reordered_euler_radians_xyz: [longEuler.x, longEuler.y, longEuler.z],
      reordered_quaternion_xyzw: xyzw(afterReorder),
    },
    object3d_hierarchy: {
      scale_policy: 'unit scale on parent and child',
      parent_local_matrix4_elements_column_major: [...parent.matrix.elements],
      child_local_matrix4_elements_column_major: [...child.matrix.elements],
      child_world_matrix4_elements_column_major: [...child.matrixWorld.elements],
      parent_world_matrix4_elements_column_major: [...parent.matrixWorld.elements],
    },
  };
}

async function main() {
  const options = args(process.argv);
  const corpusBytes = await readFile(options.corpus);
  const corpus = JSON.parse(corpusBytes.toString('utf8'));
  const lockBytes = await readFile(resolve(SCRIPT_DIR, 'package-lock.json'));
  const scriptBytes = await readFile(SCRIPT_PATH);
  const result = {
    schema_version: '1.0',
    evidence_kind: 'native_threejs_node_runtime_capture',
    target: {
      name: 'Three.js',
      revision: THREE.REVISION,
      package_version: '0.187.0',
      node_version: process.version,
      v8_version: process.versions.v8,
      platform: process.platform,
      architecture: process.arch,
      numeric_representation: 'JavaScript Number (binary64)',
    },
    provenance: {
      corpus_sha256: sha256(corpusBytes),
      corpus_file_name: basename(options.corpus),
      package_lock_sha256: sha256(lockBytes),
      exporter_sha256: sha256(scriptBytes),
      exporter_file: 'capture_three.mjs',
      exact_invocation: 'node capture_three.mjs --corpus <orientation_inputs_v1.json> --output <native_rotation.json>',
      captured_at_utc: new Date().toISOString(),
    },
    counts: {
      euler_orders: corpus.euler_inputs.length * ORDERS.length,
      euler_boundaries: corpus.euler_boundaries.length,
      axis_angle: corpus.axis_angle_inputs.length,
      quaternion_normalization: corpus.quaternion_inputs.length,
      multiplication: corpus.multiplication_pairs.length,
      shortest_slerp: corpus.interpolation_pairs.length,
      independent_matrices: corpus.matrix_inputs.length,
      two_vector_alignment: 12,
    },
    observations: makeEngine(corpus),
  };
  await writeFile(options.output, `${JSON.stringify(result, null, 2)}\n`, 'utf8');
  process.stdout.write(`${JSON.stringify({ output: options.output, counts: result.counts, corpus_sha256: result.provenance.corpus_sha256 })}\n`);
}

main().catch((error) => {
  process.stderr.write(`${error.stack ?? error}\n`);
  process.exitCode = 1;
});
