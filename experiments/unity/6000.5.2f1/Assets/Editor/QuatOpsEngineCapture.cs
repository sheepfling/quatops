using System;
using System.IO;
using UnityEditor;
using UnityEngine;
using Unity.Mathematics;

public static class QuatOpsEngineCapture
{
    [Serializable]
    private sealed class Vec3Record
    {
        public float x;
        public float y;
        public float z;

        public Vec3Record(float xValue, float yValue, float zValue) { x = xValue; y = yValue; z = zValue; }
        public Vec3Record(Vector3 value) { x = value.x; y = value.y; z = value.z; }
        public Vec3Record(float3 value) { x = value.x; y = value.y; z = value.z; }
    }

    [Serializable]
    private sealed class QuatRecord
    {
        public float x;
        public float y;
        public float z;
        public float w;

        public QuatRecord(Quaternion value) { x = value.x; y = value.y; z = value.z; w = value.w; }
        public QuatRecord(quaternion value) { x = value.value.x; y = value.value.y; z = value.value.z; w = value.value.w; }
    }

    [Serializable]
    private sealed class QuatTextRecord
    {
        public string x;
        public string y;
        public string z;
        public string w;

        public QuatTextRecord(quaternion value)
        {
            x = value.value.x.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
            y = value.value.y.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
            z = value.value.z.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
            w = value.value.w.ToString("R", System.Globalization.CultureInfo.InvariantCulture);
        }
    }

    [Serializable]
    private sealed class EngineInfo
    {
        public string name;
        public string version;
        public string platform;
        public string capture_method;
        public string precision;
        public string mathematics_package;
    }

    [Serializable]
    private sealed class FrameInfo
    {
        public string handedness;
        public string positive_x;
        public string positive_y;
        public string positive_z;
    }

    [Serializable]
    private sealed class EulerInput
    {
        public string case_id;
        public Vec3Record input_degrees_xyz;
    }

    [Serializable]
    private sealed class MathApiCase
    {
        public QuatRecord quaternion_xyzw;
        public Vec3Record[] rotated_positive_basis_vectors_xyz;
    }

    [Serializable]
    private sealed class MathematicsEulerOrderCase
    {
        public string case_id;
        public string order;
        public Vec3Record input_degrees_xyz;
        public QuatRecord quaternion_xyzw;
        public Vec3Record[] rotated_positive_basis_vectors_xyz;
        public Vec3Record readback_degrees_xyz;
        public QuatRecord readback_roundtrip_quaternion_xyzw;
        public float readback_roundtrip_angle_error_degrees;
    }

    [Serializable]
    private sealed class EulerCase
    {
        public string case_id;
        public Vec3Record input_degrees_xyz;
        public QuatRecord unity_engine_quaternion_xyzw;
        public Vec3Record unity_engine_euler_readback_degrees;
        public QuatRecord unity_engine_euler_readback_roundtrip_quaternion_xyzw;
        public float unity_engine_euler_readback_roundtrip_angle_error_degrees;
        public Vec3Record[] unity_engine_rotated_positive_basis_vectors_xyz;
        public Vec3Record[] unity_engine_matrix_rows;
        public MathApiCase unity_mathematics_euler_xyz;
    }

    [Serializable]
    private sealed class CrossCase
    {
        public string case_id;
        public Vec3Record lhs_xyz;
        public Vec3Record rhs_xyz;
        public Vec3Record unity_engine_result_xyz;
        public Vec3Record unity_engine_reversed_result_xyz;
        public Vec3Record unity_mathematics_result_xyz;
        public Vec3Record unity_mathematics_reversed_result_xyz;
    }

    [Serializable]
    private sealed class CompositionApi
    {
        public QuatRecord lhs_xyzw;
        public QuatRecord rhs_xyzw;
        public QuatRecord lhs_times_rhs_xyzw;
        public Vec3Record probe_vector_xyz;
        public Vec3Record product_rotated_vector_xyz;
        public Vec3Record rhs_then_lhs_rotated_vector_xyz;
        public Vec3Record lhs_then_rhs_rotated_vector_xyz;
        public bool product_matches_rhs_then_lhs;
    }

    [Serializable]
    private sealed class QuaternionComposition
    {
        public CompositionApi unity_engine;
        public CompositionApi unity_mathematics;
    }

    [Serializable]
    private sealed class TransformComposition
    {
        public QuatRecord parent_rotation_xyzw;
        public QuatRecord child_local_rotation_xyzw;
        public QuatRecord child_world_rotation_xyzw;
        public Vec3Record parent_translation_xyz;
        public Vec3Record child_local_translation_xyz;
        public Vec3Record probe_position_xyz;
        public Vec3Record parent_child_transform_position_xyz;
        public Vec3Record child_local_then_parent_world_position_xyz;
        public Vec3Record parent_world_then_child_local_position_xyz;
        public bool parent_child_matches_child_then_parent;
    }

    [Serializable]
    private sealed class AxisProbe
    {
        public string source_rotation_axis;
        public QuatRecord source_quaternion_xyzw;
        public QuatRecord unity_quaternion_xyzw;
        public Vec3Record[] unity_rotated_positive_basis_vectors_xyz;
        public bool preserves_mapped_rotation_action;
    }

    [Serializable]
    private sealed class AxisMapInfo
    {
        public string[] destination_axes_from_source;
        public int determinant;
        public AxisProbe[] principal_rotation_probes;
    }

    [Serializable]
    private sealed class CanonicalInteropOrientation
    {
        public string case_id;
        public float[] canonical_wxyz;
        public float[] native_quaternion;
        public Vec3Record[] native_rotated_positive_basis_vectors_xyz;
        public Vec3Record native_rotated_probe_vector_xyz;
    }

    [Serializable]
    private sealed class CanonicalInteropComposition
    {
        public string left_case_id;
        public string right_case_id;
        public string product_order;
        public float[] native_product_quaternion;
        public Vec3Record[] native_rotated_positive_basis_vectors_xyz;
        public Vec3Record native_rotated_probe_vector_xyz;
    }

    [Serializable]
    private sealed class CanonicalInteropInfo
    {
        public string contract_id;
        public string native_quaternion_order;
        public string[] frame_map_destination_axes_from_source;
        public Vec3Record probe_vector_canonical_xyz;
        public Vec3Record probe_vector_native_xyz;
        public CanonicalInteropOrientation[] orientation_cases;
        public CanonicalInteropComposition composition;
        public CanonicalInteropComposition reverse_composition;
    }

    [Serializable]
    private sealed class ProofProfile
    {
        public string euler_to_quaternion;
        public string euler_readback_and_threshold_neighbors;
        public string matrix_to_quaternion;
        public string axis_angle;
        public string rotation_vector;
        public string slerp_shortest;
        public string slerp_full_path;
        public string normalization_edges;
        public string basis_vector_action;
        public string composition_and_frame_map;
    }

    [Serializable]
    private sealed class ExtendedCapabilities
    {
        public string euler_orders;
        public string matrix_to_quaternion;
        public string axis_angle;
        public string rotation_vector;
        public string slerp_shortest;
        public string slerp_full_path;
        public string lerp_nlerp;
        public string normalization;
    }

    [Serializable]
    private sealed class MatrixConversionCase
    {
        public string case_id;
        public Vec3Record[] matrix_rows_native;
        public Vec3Record[] matrix_columns_native;
        public QuatRecord source_quaternion_xyzw;
        public QuatRecord unity_engine_matrix_rotation_xyzw;
        public QuatRecord unity_mathematics_matrix_constructor_xyzw;
        public bool engine_same_rotation;
        public bool mathematics_same_rotation;
    }

    [Serializable]
    private sealed class AxisAngleCase
    {
        public string case_id;
        public Vec3Record input_axis_xyz;
        public float input_angle_radians;
        public QuatRecord unity_engine_constructed_quaternion_xyzw;
        public QuatRecord unity_mathematics_constructed_quaternion_xyzw;
        public QuatRecord negative_quaternion_xyzw;
        public Vec3Record axis_angle_readback_axis_xyz;
        public float axis_angle_readback_angle_radians;
        public Vec3Record negative_axis_angle_readback_axis_xyz;
        public float negative_axis_angle_readback_angle_radians;
        public Vec3Record rotation_vector_readback_xyz;
        public Vec3Record negative_rotation_vector_readback_xyz;
        public float constructed_norm;
    }

    [Serializable]
    private sealed class RotationVectorCase
    {
        public string case_id;
        public Vec3Record input_rotation_vector_xyz;
        public QuatRecord unity_engine_constructed_quaternion_xyzw;
        public QuatRecord unity_mathematics_constructed_quaternion_xyzw;
        public Vec3Record rotation_vector_readback_xyz;
        public float constructed_norm;
    }

    [Serializable]
    private sealed class SlerpApiCase
    {
        public QuatRecord result_xyzw;
        public float norm;
        public bool is_finite;
    }

    [Serializable]
    private sealed class SlerpCase
    {
        public string case_id;
        public float alpha;
        public QuatRecord lhs_xyzw;
        public QuatRecord rhs_xyzw;
        public SlerpApiCase unity_engine_slerp;
        public SlerpApiCase unity_engine_slerp_unclamped;
        public SlerpApiCase unity_mathematics_slerp;
    }

    [Serializable]
    private sealed class LerpCase
    {
        public string case_id;
        public float alpha;
        public QuatRecord lhs_xyzw;
        public QuatRecord rhs_xyzw;
        public SlerpApiCase unity_engine_lerp;
        public SlerpApiCase unity_engine_lerp_unclamped;
        public SlerpApiCase unity_mathematics_nlerp;
    }

    [Serializable]
    private sealed class NormalizationCase
    {
        public string case_id;
        public QuatRecord input_xyzw;
        public float input_norm_squared;
        public QuatRecord unity_engine_normalized_xyzw;
        public float unity_engine_normalized_norm;
        public bool unity_engine_normalized_is_finite;
        public QuatTextRecord unity_mathematics_normalized_xyzw_components;
        public string unity_mathematics_normalized_norm;
        public bool unity_mathematics_normalized_is_finite;
    }

    [Serializable]
    private sealed class ExtendedConversions
    {
        public ExtendedCapabilities capabilities;
        public MathematicsEulerOrderCase[] unity_mathematics_euler_orders;
        public MatrixConversionCase[] matrix_to_quaternion;
        public AxisAngleCase[] axis_angle;
        public RotationVectorCase[] rotation_vector;
        public SlerpCase[] slerp;
        public LerpCase[] lerp;
        public NormalizationCase[] normalization;
    }

    [Serializable]
    private sealed class Fixture
    {
        public int schema_version;
        public string evidence_kind;
        public string quaternion_algebra;
        public string quaternion_order;
        public string rotation_action;
        public string matrix_layout;
        public ProofProfile proof_profile;
        public EngineInfo engine;
        public FrameInfo source_frame;
        public FrameInfo unity_world_frame;
        public AxisMapInfo source_to_unity_axis_map;
        public EulerCase[] cases;
        public CrossCase[] cross_products;
        public QuaternionComposition quaternion_composition;
        public TransformComposition transform_composition;
        public ExtendedConversions extended_conversions;
        public CanonicalInteropInfo canonical_interop;
    }

    private static readonly (string Id, Vector3 Angles)[] Cases =
    {
        ("identity", new Vector3(0, 0, 0)),
        ("x_positive_90", new Vector3(90, 0, 0)),
        ("y_positive_90", new Vector3(0, 90, 0)),
        ("z_positive_90", new Vector3(0, 0, 90)),
        ("mixed_positive", new Vector3(31.41f, -49.16f, 12.34f)),
        ("mixed_negative", new Vector3(-50, 28, 39)),
        ("near_gimbal", new Vector3(89.9f, 27, -41)),
        ("x_gimbal_threshold_below_lower", new Vector3(89.91f, 27, -41)),
        ("x_gimbal_threshold_above_lower", new Vector3(89.93f, 27, -41)),
        ("x_gimbal_threshold_below_upper", new Vector3(90.07f, 27, -41)),
        ("x_gimbal_threshold_above_upper", new Vector3(90.09f, 27, -41)),
        ("gimbal", new Vector3(90, 27, -41)),
        ("x_over_90", new Vector3(105, 23, -17)),
        ("y_positive_180", new Vector3(0, 180, 0)),
        ("y_negative_180", new Vector3(0, -180, 0)),
    };

    private static readonly string[] MathematicsEulerOrders =
    {
        "XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX",
    };
    private const int MathematicsEulerRandomSeed = 0x514F5053;

    public static void Capture()
    {
        var fixture = new Fixture
        {
            schema_version = 1,
            evidence_kind = "native_unity_editor_runtime_capture",
            quaternion_algebra = "Hamilton",
            quaternion_order = "xyzw",
            rotation_action = "active rotation applied to vectors",
            matrix_layout = "rows copied from UnityEngine.Matrix4x4.Rotate; columns are images of positive basis vectors",
            proof_profile = new ProofProfile
            {
                euler_to_quaternion = "native UnityEngine.Quaternion.Euler and all six Unity.Mathematics quaternion.Euler order constructors",
                euler_readback_and_threshold_neighbors = "native UnityEngine.Quaternion.eulerAngles plus all six Unity.Mathematics math.Euler readbacks; threshold probes around each middle axis",
                matrix_to_quaternion = "native UnityEngine.Matrix4x4.rotation and Unity.Mathematics.quaternion(float3x3)",
                axis_angle = "native UnityEngine.Quaternion.AngleAxis/ToAngleAxis and Unity.Mathematics.quaternion.AxisAngle",
                rotation_vector = "derived natively through AngleAxis/ToAngleAxis; no direct rotation-vector log/exp API",
                slerp_shortest = "native UnityEngine.Quaternion.Slerp/SlerpUnclamped and Unity.Mathematics.math.slerp",
                slerp_full_path = "not exposed by the captured public quaternion APIs",
                normalization_edges = "native UnityEngine.Quaternion.Normalize and Unity.Mathematics.math.normalize",
                basis_vector_action = "native quaternion-vector action and Matrix4x4.Rotate",
                composition_and_frame_map = "native quaternion/Transform composition and explicit source-frame probes",
            },
            engine = new EngineInfo
            {
                name = "Unity Editor",
                version = Application.unityVersion,
                platform = Application.platform.ToString(),
                capture_method = "Unity Editor batchmode C# executeMethod",
                precision = "float32",
                mathematics_package = "com.unity.mathematics 1.4.0 (built-in)",
            },
            source_frame = new FrameInfo
            {
                handedness = "right-handed",
                positive_x = "right",
                positive_y = "forward",
                positive_z = "up",
            },
            unity_world_frame = new FrameInfo
            {
                handedness = "left-handed",
                positive_x = "right",
                positive_y = "up",
                positive_z = "forward",
            },
            source_to_unity_axis_map = CaptureAxisMap(),
            cases = CaptureEulerCases(),
            cross_products = CaptureCrossProducts(),
            quaternion_composition = CaptureQuaternionComposition(),
            transform_composition = CaptureTransformComposition(),
            extended_conversions = CaptureExtendedConversions(),
            canonical_interop = CaptureCanonicalInterop(),
        };

        var path = GetOutputPath();
        Directory.CreateDirectory(Path.GetDirectoryName(path));
        File.WriteAllText(path, JsonUtility.ToJson(fixture, true) + Environment.NewLine);
        Debug.Log($"Captured native Unity fixtures to {path} (Unity {Application.unityVersion}).");
    }

    private static string GetOutputPath()
    {
        foreach (var argument in Environment.GetCommandLineArgs())
        {
            const string prefix = "-QuatOpsCaptureOutput=";
            if (argument.StartsWith(prefix, StringComparison.Ordinal))
                return Path.GetFullPath(argument.Substring(prefix.Length));
        }
        return Path.GetFullPath(Path.Combine(Path.GetTempPath(), "quatops-unity-full.json"));
    }

    private static EulerCase[] CaptureEulerCases()
    {
        var result = new EulerCase[Cases.Length];
        for (var i = 0; i < Cases.Length; ++i)
        {
            var (id, angles) = Cases[i];
            var unityQuat = Quaternion.Euler(angles);
            var mathQuat = quaternion.EulerXYZ(math.radians(new float3(angles.x, angles.y, angles.z)));
            var eulerReadback = unityQuat.eulerAngles;
            var eulerReadbackRoundtrip = Quaternion.Euler(eulerReadback);
            var matrix = Matrix4x4.Rotate(unityQuat);
            result[i] = new EulerCase
            {
                case_id = id,
                input_degrees_xyz = new Vec3Record(angles),
                unity_engine_quaternion_xyzw = new QuatRecord(unityQuat),
                unity_engine_euler_readback_degrees = new Vec3Record(eulerReadback),
                unity_engine_euler_readback_roundtrip_quaternion_xyzw = new QuatRecord(eulerReadbackRoundtrip),
                unity_engine_euler_readback_roundtrip_angle_error_degrees = OrientationErrorDegrees(unityQuat, eulerReadbackRoundtrip),
                unity_engine_rotated_positive_basis_vectors_xyz = new[]
                {
                    new Vec3Record(unityQuat * Vector3.right),
                    new Vec3Record(unityQuat * Vector3.up),
                    new Vec3Record(unityQuat * Vector3.forward),
                },
                unity_engine_matrix_rows = new[]
                {
                    new Vec3Record(matrix.m00, matrix.m01, matrix.m02),
                    new Vec3Record(matrix.m10, matrix.m11, matrix.m12),
                    new Vec3Record(matrix.m20, matrix.m21, matrix.m22),
                },
                unity_mathematics_euler_xyz = new MathApiCase
                {
                    quaternion_xyzw = new QuatRecord(mathQuat),
                    rotated_positive_basis_vectors_xyz = new[]
                    {
                        new Vec3Record(math.rotate(mathQuat, new float3(1, 0, 0))),
                        new Vec3Record(math.rotate(mathQuat, new float3(0, 1, 0))),
                        new Vec3Record(math.rotate(mathQuat, new float3(0, 0, 1))),
                    },
                },
            };
        }
        return result;
    }

    private static ExtendedConversions CaptureExtendedConversions()
    {
        return new ExtendedConversions
        {
            capabilities = new ExtendedCapabilities
            {
                euler_orders = "Unity.Mathematics six Euler constructors/readbacks; 29 fixed boundary/mixed triples plus 8 seeded triples from System.Random seed 0x514F5053",
                matrix_to_quaternion = "UnityEngine Matrix4x4.rotation; Unity.Mathematics quaternion(float3x3)",
                axis_angle = "UnityEngine Quaternion.AngleAxis/ToAngleAxis; Unity.Mathematics quaternion.AxisAngle",
                rotation_vector = "derived with native AngleAxis/ToAngleAxis because neither captured API exposes log/exp",
                slerp_shortest = "UnityEngine Quaternion.Slerp/SlerpUnclamped and Unity.Mathematics math.slerp",
                slerp_full_path = "no full-path mode is exposed by these public APIs",
                lerp_nlerp = "UnityEngine Quaternion.Lerp/LerpUnclamped and Unity.Mathematics math.nlerp",
                normalization = "UnityEngine Quaternion.Normalize and Unity.Mathematics math.normalize",
            },
            unity_mathematics_euler_orders = CaptureMathematicsEulerOrders(),
            matrix_to_quaternion = CaptureMatrixConversions(),
            axis_angle = CaptureAxisAngleCases(),
            rotation_vector = CaptureRotationVectorCases(),
            slerp = CaptureSlerpCases(),
            lerp = CaptureLerpCases(),
            normalization = CaptureNormalizationCases(),
        };
    }

    private static MathematicsEulerOrderCase[] CaptureMathematicsEulerOrders()
    {
        var cases = new System.Collections.Generic.List<(string Id, Vector3 Angles)>(Cases);
        cases.Add(("diagnostic_10_20_30", new Vector3(10, 20, 30)));
        cases.Add(("outside_canonical_ranges", new Vector3(420, -390, 765)));

        foreach (var axis in new[] { "y", "z" })
        {
            foreach (var (suffix, middle) in new[]
            {
                ("near_gimbal", 89.9f),
                ("threshold_below_lower", 89.91f),
                ("threshold_above_lower", 89.93f),
                ("gimbal", 90.0f),
                ("threshold_below_upper", 90.07f),
                ("threshold_above_upper", 90.09f),
            })
            {
                var angles = axis == "y"
                    ? new Vector3(27, middle, -41)
                    : new Vector3(27, -41, middle);
                cases.Add(($"{axis}_{suffix}", angles));
            }
        }

        var random = new System.Random(MathematicsEulerRandomSeed);
        for (var i = 0; i < 8; ++i)
        {
            var angles = new Vector3(
                (float)(random.NextDouble() * 1440.0 - 720.0),
                (float)(random.NextDouble() * 1440.0 - 720.0),
                (float)(random.NextDouble() * 1440.0 - 720.0));
            cases.Add(($"seeded_random_{i:D2}", angles));
        }

        var result = new System.Collections.Generic.List<MathematicsEulerOrderCase>(
            cases.Count * MathematicsEulerOrders.Length);
        foreach (var (caseId, angles) in cases)
        {
            var anglesXYZ = new float3(angles.x, angles.y, angles.z);
            var radiansXYZ = math.radians(anglesXYZ);
            foreach (var order in MathematicsEulerOrders)
            {
                var value = MathematicsEuler(order, radiansXYZ);
                var readbackRadians = MathematicsEulerReadback(order, value);
                var roundtrip = MathematicsEuler(order, readbackRadians);
                result.Add(new MathematicsEulerOrderCase
                {
                    case_id = caseId,
                    order = order,
                    input_degrees_xyz = new Vec3Record(angles),
                    quaternion_xyzw = new QuatRecord(value),
                    rotated_positive_basis_vectors_xyz = new[]
                    {
                        new Vec3Record(math.rotate(value, new float3(1, 0, 0))),
                        new Vec3Record(math.rotate(value, new float3(0, 1, 0))),
                        new Vec3Record(math.rotate(value, new float3(0, 0, 1))),
                    },
                    readback_degrees_xyz = new Vec3Record(math.degrees(readbackRadians)),
                    readback_roundtrip_quaternion_xyzw = new QuatRecord(roundtrip),
                    readback_roundtrip_angle_error_degrees = OrientationErrorDegrees(value, roundtrip),
                });
            }
        }
        return result.ToArray();
    }

    private static quaternion MathematicsEuler(string order, float3 anglesRadians)
    {
        switch (order)
        {
            case "XYZ": return quaternion.EulerXYZ(anglesRadians);
            case "XZY": return quaternion.EulerXZY(anglesRadians);
            case "YXZ": return quaternion.EulerYXZ(anglesRadians);
            case "YZX": return quaternion.EulerYZX(anglesRadians);
            case "ZXY": return quaternion.EulerZXY(anglesRadians);
            case "ZYX": return quaternion.EulerZYX(anglesRadians);
            default: throw new ArgumentOutOfRangeException(nameof(order), order, "Unknown Unity Mathematics Euler order.");
        }
    }

    private static float3 MathematicsEulerReadback(string order, quaternion value)
    {
        switch (order)
        {
            case "XYZ": return math.EulerXYZ(value);
            case "XZY": return math.EulerXZY(value);
            case "YXZ": return math.EulerYXZ(value);
            case "YZX": return math.EulerYZX(value);
            case "ZXY": return math.EulerZXY(value);
            case "ZYX": return math.EulerZYX(value);
            default: throw new ArgumentOutOfRangeException(nameof(order), order, "Unknown Unity Mathematics Euler order.");
        }
    }

    private static MatrixConversionCase[] CaptureMatrixConversions()
    {
        var invSqrt14 = 1.0f / Mathf.Sqrt(14.0f);
        var cases = new[]
        {
            ("identity", Quaternion.identity),
            ("x_90", Quaternion.AngleAxis(90, Vector3.right)),
            ("y_90", Quaternion.AngleAxis(90, Vector3.up)),
            ("z_90", Quaternion.AngleAxis(90, Vector3.forward)),
            ("x_180", Quaternion.AngleAxis(180, Vector3.right)),
            ("y_180", Quaternion.AngleAxis(180, Vector3.up)),
            ("z_180", Quaternion.AngleAxis(180, Vector3.forward)),
            ("mixed_axis", Quaternion.AngleAxis(70.65f, new Vector3(1, 2, 3).normalized)),
            ("negative_trace", Quaternion.AngleAxis(170, new Vector3(invSqrt14, 2 * invSqrt14, 3 * invSqrt14))),
        };
        var result = new MatrixConversionCase[cases.Length];
        for (var i = 0; i < cases.Length; ++i)
        {
            var (id, source) = cases[i];
            var matrix = Matrix4x4.Rotate(source);
            var c0 = new Vector3(matrix.m00, matrix.m10, matrix.m20);
            var c1 = new Vector3(matrix.m01, matrix.m11, matrix.m21);
            var c2 = new Vector3(matrix.m02, matrix.m12, matrix.m22);
            var matrix3 = new float3x3(
                new float3(c0.x, c0.y, c0.z),
                new float3(c1.x, c1.y, c1.z),
                new float3(c2.x, c2.y, c2.z));
            var engineResult = matrix.rotation;
            var mathResult = new quaternion(matrix3);
            var engineSame = OrientationErrorDegrees(source, engineResult) <= 1e-3f;
            var mathSame = MathSameRotation(
                new quaternion(new float4(source.x, source.y, source.z, source.w)), mathResult, 2e-5f);
            if (!engineSame || !mathSame)
                throw new InvalidOperationException($"Unity matrix-to-quaternion conversion changed {id}.");
            result[i] = new MatrixConversionCase
            {
                case_id = id,
                matrix_rows_native = new[]
                {
                    new Vec3Record(matrix.m00, matrix.m01, matrix.m02),
                    new Vec3Record(matrix.m10, matrix.m11, matrix.m12),
                    new Vec3Record(matrix.m20, matrix.m21, matrix.m22),
                },
                matrix_columns_native = new[] { new Vec3Record(c0), new Vec3Record(c1), new Vec3Record(c2) },
                source_quaternion_xyzw = new QuatRecord(source),
                unity_engine_matrix_rotation_xyzw = new QuatRecord(engineResult),
                unity_mathematics_matrix_constructor_xyzw = new QuatRecord(mathResult),
                engine_same_rotation = engineSame,
                mathematics_same_rotation = mathSame,
            };
        }
        return result;
    }

    private static AxisAngleCase[] CaptureAxisAngleCases()
    {
        var invSqrt14 = 1.0f / Mathf.Sqrt(14.0f);
        var cases = new[]
        {
            ("zero_angle", Vector3.right, 0.0f),
            ("tiny_angle", new Vector3(1, 2, 3).normalized, 1e-7f),
            ("nonprincipal_angle", new Vector3(1, 2, 3).normalized, 2.4f),
            ("exact_pi", Vector3.forward, Mathf.PI),
            ("just_above_pi", new Vector3(-2 * invSqrt14, invSqrt14, 3 * invSqrt14), Mathf.PI + 0.2f),
        };
        var result = new AxisAngleCase[cases.Length];
        for (var i = 0; i < cases.Length; ++i)
        {
            var (id, axis, angle) = cases[i];
            var normalizedAxis = axis.normalized;
            var engine = Quaternion.AngleAxis(angle * Mathf.Rad2Deg, normalizedAxis);
            var mathAxis = new float3(normalizedAxis.x, normalizedAxis.y, normalizedAxis.z);
            var mathValue = quaternion.AxisAngle(mathAxis, angle);
            var negative = new Quaternion(-engine.x, -engine.y, -engine.z, -engine.w);
            var mathSame = MathSameRotation(
                new quaternion(new float4(engine.x, engine.y, engine.z, engine.w)), mathValue, 2e-5f);
            if (!mathSame)
                throw new InvalidOperationException($"Unity.Mathematics axis-angle construction changed {id}.");
            engine.ToAngleAxis(out var readAngle, out var readAxis);
            negative.ToAngleAxis(out var negativeReadAngle, out var negativeReadAxis);
            result[i] = new AxisAngleCase
            {
                case_id = id,
                input_axis_xyz = new Vec3Record(axis),
                input_angle_radians = angle,
                unity_engine_constructed_quaternion_xyzw = new QuatRecord(engine),
                unity_mathematics_constructed_quaternion_xyzw = new QuatRecord(mathValue),
                negative_quaternion_xyzw = new QuatRecord(negative),
                axis_angle_readback_axis_xyz = new Vec3Record(readAxis),
                axis_angle_readback_angle_radians = readAngle * Mathf.Deg2Rad,
                negative_axis_angle_readback_axis_xyz = new Vec3Record(negativeReadAxis),
                negative_axis_angle_readback_angle_radians = negativeReadAngle * Mathf.Deg2Rad,
                rotation_vector_readback_xyz = new Vec3Record(PrincipalRotationVector(engine)),
                negative_rotation_vector_readback_xyz = new Vec3Record(PrincipalRotationVector(negative)),
                constructed_norm = QuaternionNorm(engine),
            };
        }
        return result;
    }

    private static RotationVectorCase[] CaptureRotationVectorCases()
    {
        var invSqrt14 = 1.0f / Mathf.Sqrt(14.0f);
        var cases = new[]
        {
            ("zero", Vector3.zero),
            ("tiny", new Vector3(1e-7f, -2e-7f, 3e-7f)),
            ("mixed", new Vector3(0.3f, -0.7f, 1.2f)),
            ("exact_pi", new Vector3(0, 0, Mathf.PI)),
            ("just_above_pi", new Vector3(-2 * invSqrt14, invSqrt14, 3 * invSqrt14) * (Mathf.PI + 0.2f)),
        };
        var result = new RotationVectorCase[cases.Length];
        for (var i = 0; i < cases.Length; ++i)
        {
            var (id, rotationVector) = cases[i];
            var angle = rotationVector.magnitude;
            var axis = angle == 0 ? Vector3.right : rotationVector / angle;
            var engine = Quaternion.AngleAxis(angle * Mathf.Rad2Deg, axis);
            var mathValue = quaternion.AxisAngle(
                new float3(axis.x, axis.y, axis.z), angle);
            var mathSame = MathSameRotation(
                new quaternion(new float4(engine.x, engine.y, engine.z, engine.w)), mathValue, 2e-5f);
            if (!mathSame)
                throw new InvalidOperationException($"Unity.Mathematics rotation-vector construction changed {id}.");
            result[i] = new RotationVectorCase
            {
                case_id = id,
                input_rotation_vector_xyz = new Vec3Record(rotationVector),
                unity_engine_constructed_quaternion_xyzw = new QuatRecord(engine),
                unity_mathematics_constructed_quaternion_xyzw = new QuatRecord(mathValue),
                rotation_vector_readback_xyz = new Vec3Record(PrincipalRotationVector(engine)),
                constructed_norm = QuaternionNorm(engine),
            };
        }
        return result;
    }

    private static SlerpCase[] CaptureSlerpCases()
    {
        var invSqrt14 = 1.0f / Mathf.Sqrt(14.0f);
        var identity = Quaternion.identity;
        var x90 = Quaternion.AngleAxis(90, Vector3.right);
        var z270 = Quaternion.AngleAxis(270, Vector3.forward);
        var arbitraryA = Quaternion.AngleAxis(63, new Vector3(invSqrt14, 2 * invSqrt14, 3 * invSqrt14));
        var arbitraryB = Quaternion.AngleAxis(126, new Vector3(-2 * invSqrt14, invSqrt14, 3 * invSqrt14));
        var nearA = Quaternion.AngleAxis(45.84f, Vector3.forward);
        var nearB = Quaternion.AngleAxis(45.8973f, Vector3.forward);
        var nearPi = Quaternion.AngleAxis(180 - 0.0057f, Vector3.forward);
        var inputs = new System.Collections.Generic.List<(string Id, Quaternion Lhs, Quaternion Rhs, float Alpha)>();
        foreach (var alpha in new[] { 0f, 0.25f, 0.5f, 0.75f, 1f })
        {
            inputs.Add(("identity_to_x90", identity, x90, alpha));
            inputs.Add(("long_path", identity, z270, alpha));
            inputs.Add(("antipodal", arbitraryA, Negate(arbitraryA), alpha));
        }
        foreach (var alpha in new[] { 0f, 0.5f, 1f })
            inputs.Add(("arbitrary_pair", arbitraryA, arbitraryB, alpha));
        inputs.Add(("nearly_identical", nearA, nearB, 0.5f));
        inputs.Add(("near_180_separation", identity, nearPi, 0.5f));
        inputs.Add(("extrapolation", identity, x90, -0.5f));
        inputs.Add(("extrapolation", identity, x90, 1.5f));

        var result = new SlerpCase[inputs.Count];
        for (var i = 0; i < inputs.Count; ++i)
        {
            var (id, lhs, rhs, alpha) = inputs[i];
            var engineClamped = Quaternion.Slerp(lhs, rhs, alpha);
            var engineUnclamped = Quaternion.SlerpUnclamped(lhs, rhs, alpha);
            var mathLhs = new quaternion(new float4(lhs.x, lhs.y, lhs.z, lhs.w));
            var mathRhs = new quaternion(new float4(rhs.x, rhs.y, rhs.z, rhs.w));
            var mathResult = math.slerp(mathLhs, mathRhs, alpha);
            result[i] = new SlerpCase
            {
                case_id = id,
                alpha = alpha,
                lhs_xyzw = new QuatRecord(lhs),
                rhs_xyzw = new QuatRecord(rhs),
                unity_engine_slerp = CaptureSlerpApi(engineClamped),
                unity_engine_slerp_unclamped = CaptureSlerpApi(engineUnclamped),
                unity_mathematics_slerp = CaptureSlerpApi(mathResult),
            };
        }
        return result;
    }

    private static LerpCase[] CaptureLerpCases()
    {
        var invSqrt14 = 1.0f / Mathf.Sqrt(14.0f);
        var identity = Quaternion.identity;
        var x90 = Quaternion.AngleAxis(90, Vector3.right);
        var z270 = Quaternion.AngleAxis(270, Vector3.forward);
        var arbitrary = Quaternion.AngleAxis(63, new Vector3(invSqrt14, 2 * invSqrt14, 3 * invSqrt14));
        var nearA = Quaternion.AngleAxis(45.84f, Vector3.forward);
        var nearB = Quaternion.AngleAxis(45.8973f, Vector3.forward);
        var inputs = new System.Collections.Generic.List<(string Id, Quaternion Lhs, Quaternion Rhs, float Alpha)>();
        foreach (var alpha in new[] { 0f, 0.25f, 0.5f, 0.75f, 1f })
        {
            inputs.Add(("identity_to_x90", identity, x90, alpha));
            inputs.Add(("long_path_endpoint_sign", identity, z270, alpha));
            inputs.Add(("antipodal", arbitrary, Negate(arbitrary), alpha));
        }
        inputs.Add(("nearly_identical", nearA, nearB, 0.5f));
        inputs.Add(("extrapolation", identity, x90, -0.5f));
        inputs.Add(("extrapolation", identity, x90, 1.5f));

        var result = new LerpCase[inputs.Count];
        for (var i = 0; i < inputs.Count; ++i)
        {
            var (id, lhs, rhs, alpha) = inputs[i];
            var mathLhs = new quaternion(new float4(lhs.x, lhs.y, lhs.z, lhs.w));
            var mathRhs = new quaternion(new float4(rhs.x, rhs.y, rhs.z, rhs.w));
            result[i] = new LerpCase
            {
                case_id = id,
                alpha = alpha,
                lhs_xyzw = new QuatRecord(lhs),
                rhs_xyzw = new QuatRecord(rhs),
                unity_engine_lerp = CaptureSlerpApi(Quaternion.Lerp(lhs, rhs, alpha)),
                unity_engine_lerp_unclamped = CaptureSlerpApi(Quaternion.LerpUnclamped(lhs, rhs, alpha)),
                unity_mathematics_nlerp = CaptureSlerpApi(math.nlerp(mathLhs, mathRhs, alpha)),
            };
        }
        return result;
    }

    private static NormalizationCase[] CaptureNormalizationCases()
    {
        var inputs = new[]
        {
            ("identity", Quaternion.identity),
            ("zero", new Quaternion(0, 0, 0, 0)),
            ("below_threshold", new Quaternion(5e-5f, 0, 0, 0)),
            ("above_threshold", new Quaternion(2e-4f, 0, 0, 0)),
            ("nonunit", new Quaternion(0.2f, -0.3f, 0.4f, 0.5f)),
        };
        var result = new NormalizationCase[inputs.Length];
        for (var i = 0; i < inputs.Length; ++i)
        {
            var (id, engineInput) = inputs[i];
            var mathInput = new quaternion(new float4(engineInput.x, engineInput.y, engineInput.z, engineInput.w));
            var engineNormalized = Quaternion.Normalize(engineInput);
            var mathNormalized = math.normalize(mathInput);
            result[i] = new NormalizationCase
            {
                case_id = id,
                input_xyzw = new QuatRecord(engineInput),
                input_norm_squared = engineInput.x * engineInput.x + engineInput.y * engineInput.y + engineInput.z * engineInput.z + engineInput.w * engineInput.w,
                unity_engine_normalized_xyzw = new QuatRecord(engineNormalized),
                unity_engine_normalized_norm = QuaternionNorm(engineNormalized),
                unity_engine_normalized_is_finite = IsFinite(engineNormalized),
                unity_mathematics_normalized_xyzw_components = new QuatTextRecord(mathNormalized),
                unity_mathematics_normalized_norm = FormatFloat(math.length(mathNormalized.value)),
                unity_mathematics_normalized_is_finite = IsFinite(mathNormalized),
            };
        }
        return result;
    }

    private static SlerpApiCase CaptureSlerpApi(Quaternion value)
    {
        return new SlerpApiCase
        {
            result_xyzw = new QuatRecord(value),
            norm = QuaternionNorm(value),
            is_finite = IsFinite(value),
        };
    }

    private static SlerpApiCase CaptureSlerpApi(quaternion value)
    {
        return new SlerpApiCase
        {
            result_xyzw = new QuatRecord(value),
            norm = math.length(value.value),
            is_finite = IsFinite(value),
        };
    }

    private static Vector3 PrincipalRotationVector(Quaternion value)
    {
        value.ToAngleAxis(out var angleDegrees, out var axis);
        if (angleDegrees > 180.0f)
        {
            angleDegrees = 360.0f - angleDegrees;
            axis = -axis;
        }
        return axis * (angleDegrees * Mathf.Deg2Rad);
    }

    private static Quaternion Negate(Quaternion value) => new Quaternion(-value.x, -value.y, -value.z, -value.w);

    private static float QuaternionNorm(Quaternion value) => Mathf.Sqrt(
        value.x * value.x + value.y * value.y + value.z * value.z + value.w * value.w);

    private static float OrientationErrorDegrees(quaternion lhs, quaternion rhs)
    {
        var relative = math.mul(math.conjugate(lhs), rhs);
        var vectorLength = math.length(new float3(
            relative.value.x, relative.value.y, relative.value.z));
        return math.degrees(2.0f * math.atan2(vectorLength, math.abs(relative.value.w)));
    }

    private static string FormatFloat(float value) =>
        value.ToString("R", System.Globalization.CultureInfo.InvariantCulture);

    private static bool IsFinite(Quaternion value) =>
        !float.IsNaN(value.x) && !float.IsInfinity(value.x) &&
        !float.IsNaN(value.y) && !float.IsInfinity(value.y) &&
        !float.IsNaN(value.z) && !float.IsInfinity(value.z) &&
        !float.IsNaN(value.w) && !float.IsInfinity(value.w);

    private static bool IsFinite(quaternion value) =>
        !float.IsNaN(value.value.x) && !float.IsInfinity(value.value.x) &&
        !float.IsNaN(value.value.y) && !float.IsInfinity(value.value.y) &&
        !float.IsNaN(value.value.z) && !float.IsInfinity(value.value.z) &&
        !float.IsNaN(value.value.w) && !float.IsInfinity(value.value.w);

    private static bool MathSameRotation(quaternion lhs, quaternion rhs, float tolerance)
    {
        var dot = math.abs(math.dot(lhs.value, rhs.value));
        return math.isfinite(dot) && math.abs(dot - 1.0f) <= tolerance;
    }

    private static CrossCase[] CaptureCrossProducts()
    {
        var axes = new[]
        {
            ("x_cross_y", Vector3.right, Vector3.up),
            ("y_cross_z", Vector3.up, Vector3.forward),
            ("z_cross_x", Vector3.forward, Vector3.right),
        };
        var result = new CrossCase[axes.Length];
        for (var i = 0; i < axes.Length; ++i)
        {
            var (id, lhs, rhs) = axes[i];
            var mathLhs = new float3(lhs.x, lhs.y, lhs.z);
            var mathRhs = new float3(rhs.x, rhs.y, rhs.z);
            result[i] = new CrossCase
            {
                case_id = id,
                lhs_xyz = new Vec3Record(lhs),
                rhs_xyz = new Vec3Record(rhs),
                unity_engine_result_xyz = new Vec3Record(Vector3.Cross(lhs, rhs)),
                unity_engine_reversed_result_xyz = new Vec3Record(Vector3.Cross(rhs, lhs)),
                unity_mathematics_result_xyz = new Vec3Record(math.cross(mathLhs, mathRhs)),
                unity_mathematics_reversed_result_xyz = new Vec3Record(math.cross(mathRhs, mathLhs)),
            };
            var expected = Vector3.Cross(lhs, rhs);
            AssertNear(Vector3.Cross(rhs, lhs), -expected, 1e-6f, $"Unity Vector3 cross anti-commutativity: {id}");
            AssertNear(math.cross(mathRhs, mathLhs), -new float3(expected.x, expected.y, expected.z), 1e-6f,
                $"Unity.Mathematics cross anti-commutativity: {id}");
        }
        return result;
    }

    private static QuaternionComposition CaptureQuaternionComposition()
    {
        var lhs = Quaternion.AngleAxis(90, Vector3.right);
        var rhs = Quaternion.AngleAxis(90, Vector3.up);
        var mathLhs = quaternion.AxisAngle(new float3(1, 0, 0), math.radians(90));
        var mathRhs = quaternion.AxisAngle(new float3(0, 1, 0), math.radians(90));
        var probe = new Vector3(1.25f, -2.5f, 3.75f);
        var mathProbe = new float3(probe.x, probe.y, probe.z);
        var unityRecord = CaptureComposition(lhs, rhs, probe);
        var mathProduct = math.mul(mathLhs, mathRhs);
        var mathImage = math.rotate(mathProduct, mathProbe);
        var mathRhsThenLhs = math.rotate(mathLhs, math.rotate(mathRhs, mathProbe));
        var mathLhsThenRhs = math.rotate(mathRhs, math.rotate(mathLhs, mathProbe));
        var mathRecord = new CompositionApi
        {
            lhs_xyzw = new QuatRecord(mathLhs),
            rhs_xyzw = new QuatRecord(mathRhs),
            lhs_times_rhs_xyzw = new QuatRecord(mathProduct),
            probe_vector_xyz = new Vec3Record(mathProbe),
            product_rotated_vector_xyz = new Vec3Record(mathImage),
            rhs_then_lhs_rotated_vector_xyz = new Vec3Record(mathRhsThenLhs),
            lhs_then_rhs_rotated_vector_xyz = new Vec3Record(mathLhsThenRhs),
            product_matches_rhs_then_lhs = Near(mathImage, mathRhsThenLhs, 1e-5f),
        };
        if (!unityRecord.product_matches_rhs_then_lhs || !mathRecord.product_matches_rhs_then_lhs)
            throw new InvalidOperationException("Unity quaternion product order did not match right operand first.");
        return new QuaternionComposition { unity_engine = unityRecord, unity_mathematics = mathRecord };
    }

    private static CompositionApi CaptureComposition(Quaternion lhs, Quaternion rhs, Vector3 probe)
    {
        var product = lhs * rhs;
        var productImage = product * probe;
        var rhsThenLhs = lhs * (rhs * probe);
        var lhsThenRhs = rhs * (lhs * probe);
        return new CompositionApi
        {
            lhs_xyzw = new QuatRecord(lhs),
            rhs_xyzw = new QuatRecord(rhs),
            lhs_times_rhs_xyzw = new QuatRecord(product),
            probe_vector_xyz = new Vec3Record(probe),
            product_rotated_vector_xyz = new Vec3Record(productImage),
            rhs_then_lhs_rotated_vector_xyz = new Vec3Record(rhsThenLhs),
            lhs_then_rhs_rotated_vector_xyz = new Vec3Record(lhsThenRhs),
            product_matches_rhs_then_lhs = Near(productImage, rhsThenLhs, 1e-5f),
        };
    }

    private static TransformComposition CaptureTransformComposition()
    {
        var lhsRotation = Quaternion.AngleAxis(90, Vector3.right);
        var rhsRotation = Quaternion.AngleAxis(90, Vector3.up);
        var parentPosition = new Vector3(3, -2, 5);
        var childPosition = new Vector3(-7, 11, 13);
        var probe = new Vector3(2, 3, 5);
        var parent = new GameObject("QuatOpsCaptureParent") { hideFlags = HideFlags.HideAndDontSave };
        var child = new GameObject("QuatOpsCaptureChild") { hideFlags = HideFlags.HideAndDontSave };
        try
        {
            parent.transform.SetPositionAndRotation(parentPosition, lhsRotation);
            child.transform.SetParent(parent.transform, false);
            child.transform.localPosition = childPosition;
            child.transform.localRotation = rhsRotation;
            child.transform.localScale = Vector3.one;

            var childLocal = Matrix4x4.TRS(childPosition, rhsRotation, Vector3.one);
            var childThenParent = parent.transform.TransformPoint(childLocal.MultiplyPoint3x4(probe));
            var parentThenChild = childLocal.MultiplyPoint3x4(parent.transform.TransformPoint(probe));
            var actual = child.transform.TransformPoint(probe);
            var expectedWorldRotation = parent.transform.rotation * child.transform.localRotation;
            var result = new TransformComposition
            {
                parent_rotation_xyzw = new QuatRecord(parent.transform.rotation),
                child_local_rotation_xyzw = new QuatRecord(child.transform.localRotation),
                child_world_rotation_xyzw = new QuatRecord(child.transform.rotation),
                parent_translation_xyz = new Vec3Record(parent.transform.position),
                child_local_translation_xyz = new Vec3Record(child.transform.localPosition),
                probe_position_xyz = new Vec3Record(probe),
                parent_child_transform_position_xyz = new Vec3Record(actual),
                child_local_then_parent_world_position_xyz = new Vec3Record(childThenParent),
                parent_world_then_child_local_position_xyz = new Vec3Record(parentThenChild),
                parent_child_matches_child_then_parent = Near(actual, childThenParent, 1e-4f),
            };
            if (!result.parent_child_matches_child_then_parent ||
                OrientationErrorDegrees(expectedWorldRotation, child.transform.rotation) > 1e-3f)
                throw new InvalidOperationException("Unity Transform parent/local composition did not match its expected world result.");
            return result;
        }
        finally
        {
            UnityEngine.Object.DestroyImmediate(child);
            UnityEngine.Object.DestroyImmediate(parent);
        }
    }

    private static AxisMapInfo CaptureAxisMap()
    {
        var halfSqrt = Mathf.Sqrt(0.5f);
        var sourceQuats = new[]
        {
            ("+X right", new Quaternion(halfSqrt, 0, 0, halfSqrt)),
            ("+Y forward", new Quaternion(0, halfSqrt, 0, halfSqrt)),
            ("+Z up", new Quaternion(0, 0, halfSqrt, halfSqrt)),
        };
        var probes = new AxisProbe[sourceQuats.Length];
        for (var i = 0; i < sourceQuats.Length; ++i)
        {
            var (axis, source) = sourceQuats[i];
            var mapped = new Quaternion(-source.x, -source.z, -source.y, source.w);
            var basis = new[] { Vector3.right, Vector3.up, Vector3.forward };
            var preserves = true;
            foreach (var sourceBasis in basis)
            {
                var sourceImage = source * sourceBasis;
                var mappedSourceImage = MapSourceVector(sourceImage);
                var mappedBasisImage = mapped * MapSourceVector(sourceBasis);
                preserves &= Near(mappedSourceImage, mappedBasisImage, 1e-6f);
            }
            probes[i] = new AxisProbe
            {
                source_rotation_axis = axis,
                source_quaternion_xyzw = new QuatRecord(source),
                unity_quaternion_xyzw = new QuatRecord(mapped),
                unity_rotated_positive_basis_vectors_xyz = new[]
                {
                    new Vec3Record(mapped * Vector3.right),
                    new Vec3Record(mapped * Vector3.up),
                    new Vec3Record(mapped * Vector3.forward),
                },
                preserves_mapped_rotation_action = preserves,
            };
            if (!preserves)
                throw new InvalidOperationException($"Unity source-axis map failed for {axis}.");
        }
        return new AxisMapInfo
        {
            destination_axes_from_source = new[] { "x", "z", "y" },
            determinant = -1,
            principal_rotation_probes = probes,
        };
    }

    private static CanonicalInteropInfo CaptureCanonicalInterop()
    {
        const string contractId = "quatops-canonical-hamilton-wxyz-active-rh-v1";
        var orientations = new (string id, float[] wxyz)[]
        {
            ("identity", new[] { 1f, 0f, 0f, 0f }),
            ("quarter_x", new[] { 0.7071067811865476f, 0.7071067811865476f, 0f, 0f }),
            ("quarter_y", new[] { 0.7071067811865476f, 0f, 0.7071067811865476f, 0f }),
            ("quarter_z", new[] { 0.7071067811865476f, 0f, 0f, 0.7071067811865476f }),
            ("mixed_a", new[] { 0.8f, 0.2f, -0.4f, 0.4f }),
            ("mixed_b", new[] { 0.7f, -0.3f, 0.2f, 0.6164414002968976f }),
            ("antipode_mixed_a", new[] { -0.8f, -0.2f, 0.4f, -0.4f }),
            ("nonprincipal_halfturn", new[] { 0f, 0.5773502691896258f, 0.5773502691896258f, 0.5773502691896258f }),
        };
        var canonicalProbe = new Vector3(0.25f, -1.5f, 2.75f);
        var nativeProbe = MapSourceVector(canonicalProbe);
        var nativeById = new System.Collections.Generic.Dictionary<string, Quaternion>();
        var cases = new CanonicalInteropOrientation[orientations.Length];
        for (var index = 0; index < orientations.Length; index++)
        {
            var item = orientations[index];
            var native = MapCanonicalQuaternion(item.wxyz);
            nativeById[item.id] = native;
            cases[index] = CaptureCanonicalInteropOrientation(item.id, item.wxyz, native, nativeProbe);
        }
        return new CanonicalInteropInfo
        {
            contract_id = contractId,
            native_quaternion_order = "xyzw",
            frame_map_destination_axes_from_source = new[] { "x", "z", "y" },
            probe_vector_canonical_xyz = new Vec3Record(canonicalProbe),
            probe_vector_native_xyz = new Vec3Record(nativeProbe),
            orientation_cases = cases,
            composition = CaptureCanonicalInteropComposition(
                "mixed_a", "mixed_b", nativeById["mixed_a"] * nativeById["mixed_b"], nativeProbe),
            reverse_composition = CaptureCanonicalInteropComposition(
                "mixed_b", "mixed_a", nativeById["mixed_b"] * nativeById["mixed_a"], nativeProbe),
        };
    }

    private static CanonicalInteropComposition CaptureCanonicalInteropComposition(
        string leftId, string rightId, Quaternion product, Vector3 nativeProbe) => new()
    {
        left_case_id = leftId,
        right_case_id = rightId,
        product_order = "lhs * rhs; rhs acts first",
        native_product_quaternion = QuaternionToArray(product),
        native_rotated_positive_basis_vectors_xyz = RotateBasis(product),
        native_rotated_probe_vector_xyz = new Vec3Record(product * nativeProbe),
    };

    private static CanonicalInteropOrientation CaptureCanonicalInteropOrientation(
        string caseId, float[] canonicalWxyz, Quaternion native, Vector3 nativeProbe)
    {
        return new CanonicalInteropOrientation
        {
            case_id = caseId,
            canonical_wxyz = canonicalWxyz,
            native_quaternion = QuaternionToArray(native),
            native_rotated_positive_basis_vectors_xyz = RotateBasis(native),
            native_rotated_probe_vector_xyz = new Vec3Record(native * nativeProbe),
        };
    }

    private static Quaternion MapCanonicalQuaternion(float[] wxyz)
    {
        // The fixture's explicit Unity frame map is (x, z, y), with det(B)=-1.
        return new Quaternion(-wxyz[1], -wxyz[3], -wxyz[2], wxyz[0]);
    }

    private static float[] QuaternionToArray(Quaternion value) =>
        new[] { value.x, value.y, value.z, value.w };

    private static Vec3Record[] RotateBasis(Quaternion value) => new[]
    {
        new Vec3Record(value * Vector3.right),
        new Vec3Record(value * Vector3.up),
        new Vec3Record(value * Vector3.forward),
    };

    private static Vector3 MapSourceVector(Vector3 value) => new Vector3(value.x, value.z, value.y);

    private static float OrientationErrorDegrees(Quaternion lhs, Quaternion rhs)
    {
        var relative = Quaternion.Inverse(lhs) * rhs;
        var relativeVectorLength = Mathf.Sqrt(
            relative.x * relative.x + relative.y * relative.y + relative.z * relative.z);
        return 2.0f * Mathf.Atan2(relativeVectorLength, Mathf.Abs(relative.w)) * Mathf.Rad2Deg;
    }

    private static bool Near(Vector3 lhs, Vector3 rhs, float tolerance) => (lhs - rhs).sqrMagnitude <= tolerance * tolerance;

    private static bool Near(float3 lhs, float3 rhs, float tolerance) => math.lengthsq(lhs - rhs) <= tolerance * tolerance;

    private static void AssertNear(Vector3 actual, Vector3 expected, float tolerance, string label)
    {
        if (!Near(actual, expected, tolerance))
            throw new InvalidOperationException($"{label}: got {actual}, expected {expected}.");
    }

    private static void AssertNear(float3 actual, float3 expected, float tolerance, string label)
    {
        if (!Near(actual, expected, tolerance))
            throw new InvalidOperationException($"{label}: got {actual}, expected {expected}.");
    }
}
