using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

public static class QuatOpsEngineCapture
{
    [Serializable]
    private sealed class VectorRecord
    {
        public float x;
        public float y;
        public float z;

        public VectorRecord(Vector3 value)
        {
            x = value.x;
            y = value.y;
            z = value.z;
        }
    }

    [Serializable]
    private sealed class EngineRecord
    {
        public string name;
        public string version;
        public string platform;
        public string capture_method;
        public string precision;
    }

    [Serializable]
    private sealed class FrameRecord
    {
        public string handedness;
        public string positive_x;
        public string positive_y;
        public string positive_z;
    }

    [Serializable]
    private sealed class AxisMapRecord
    {
        public string[] destination_axes_from_source;
        public int determinant;
    }

    [Serializable]
    private sealed class OrientationInput
    {
        public string case_id;
        public float[] canonical_wxyz;
    }

    [Serializable]
    private sealed class OrientationRecord
    {
        public string case_id;
        public float[] canonical_wxyz;
        public float[] native_quaternion;
        public VectorRecord[] native_rotated_positive_basis_vectors_xyz;
    }

    [Serializable]
    private sealed class CompositionRecord
    {
        public string left_case_id;
        public string right_case_id;
        public string product_order;
        public float[] native_product_quaternion;
        public VectorRecord[] native_rotated_positive_basis_vectors_xyz;
    }

    [Serializable]
    private sealed class CanonicalInteropRecord
    {
        public string contract_id;
        public string native_quaternion_order;
        public string[] frame_map_destination_axes_from_source;
        public VectorRecord native_mixed_probe_xyz;
        public OrientationRecord[] orientation_cases;
        public CompositionRecord composition;
        public CompositionRecord reverse_composition;
    }

    [Serializable]
    private sealed class CaptureRecord
    {
        public int schema_version;
        public string evidence_kind;
        public string capture_method;
        public EngineRecord engine;
        public FrameRecord source_frame;
        public FrameRecord unity_world_frame;
        public AxisMapRecord source_to_unity_axis_map;
        public CanonicalInteropRecord canonical_interop;
    }

    private static readonly OrientationInput[] Orientations =
    {
        new() { case_id = "identity", canonical_wxyz = new[] { 1f, 0f, 0f, 0f } },
        new() { case_id = "mixed_a", canonical_wxyz = new[] { 0.8f, 0.2f, -0.4f, 0.4f } },
        new() { case_id = "mixed_b", canonical_wxyz = new[] { 0.7f, -0.3f, 0.2f, 0.6164414002968976f } },
        new() { case_id = "antipode_mixed_a", canonical_wxyz = new[] { -0.8f, -0.2f, 0.4f, -0.4f } },
        new() { case_id = "nonprincipal_halfturn", canonical_wxyz = new[] { 0f, 0.5773502691896258f, 0.5773502691896258f, 0.5773502691896258f } },
    };

    public static void Capture()
    {
        const string captureMethod = "Unity Editor batchmode C# executeMethod";
        var record = new CaptureRecord
        {
            schema_version = 1,
            evidence_kind = "native_unity_editor_runtime_capture",
            capture_method = captureMethod,
            engine = new EngineRecord
            {
                name = "Unity Editor",
                version = Application.unityVersion,
                platform = Application.platform.ToString(),
                capture_method = captureMethod,
                precision = "float32",
            },
            source_frame = new FrameRecord
            {
                handedness = "right-handed",
                positive_x = "right",
                positive_y = "forward",
                positive_z = "up",
            },
            unity_world_frame = new FrameRecord
            {
                handedness = "left-handed",
                positive_x = "right",
                positive_y = "up",
                positive_z = "forward",
            },
            source_to_unity_axis_map = new AxisMapRecord
            {
                destination_axes_from_source = new[] { "x", "z", "y" },
                determinant = -1,
            },
            canonical_interop = CaptureCanonicalInterop(),
        };

        var outputPath = GetOutputPath();
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath));
        File.WriteAllText(
            outputPath,
            JsonUtility.ToJson(record, true) + Environment.NewLine,
            new UTF8Encoding(false)
        );
        Debug.Log($"Captured selected Unity quaternion witnesses to {outputPath} (Unity {Application.unityVersion}).");
    }

    private static CanonicalInteropRecord CaptureCanonicalInterop()
    {
        var probe = new Vector3(0.25f, -1.5f, 2.75f);
        var nativeProbe = MapSourceVector(probe);
        var nativeById = new Dictionary<string, Quaternion>();
        var cases = new OrientationRecord[Orientations.Length];
        for (var index = 0; index < Orientations.Length; index++)
        {
            var input = Orientations[index];
            var native = MapCanonicalQuaternion(input.canonical_wxyz);
            nativeById[input.case_id] = native;
            cases[index] = CaptureOrientation(input, native);
        }

        return new CanonicalInteropRecord
        {
            contract_id = "quatops-canonical-hamilton-wxyz-active-rh-v1",
            native_quaternion_order = "xyzw",
            frame_map_destination_axes_from_source = new[] { "x", "z", "y" },
            native_mixed_probe_xyz = new VectorRecord(nativeById["mixed_a"] * nativeProbe),
            orientation_cases = cases,
            composition = CaptureComposition(
                "mixed_a", "mixed_b", nativeById["mixed_a"] * nativeById["mixed_b"]
            ),
            reverse_composition = CaptureComposition(
                "mixed_b", "mixed_a", nativeById["mixed_b"] * nativeById["mixed_a"]
            ),
        };
    }

    private static OrientationRecord CaptureOrientation(
        OrientationInput input,
        Quaternion native
    ) => new()
    {
        case_id = input.case_id,
        canonical_wxyz = input.canonical_wxyz,
        native_quaternion = QuaternionXyzw(native),
        native_rotated_positive_basis_vectors_xyz = RotateBasis(native),
    };

    private static CompositionRecord CaptureComposition(
        string leftId,
        string rightId,
        Quaternion product
    ) => new()
    {
        left_case_id = leftId,
        right_case_id = rightId,
        product_order = "lhs * rhs; rhs acts first",
        native_product_quaternion = QuaternionXyzw(product),
        native_rotated_positive_basis_vectors_xyz = RotateBasis(product),
    };

    private static Quaternion MapCanonicalQuaternion(float[] wxyz)
    {
        // The declared frame map is (x, z, y), with determinant -1.
        return new Quaternion(-wxyz[1], -wxyz[3], -wxyz[2], wxyz[0]);
    }

    private static Vector3 MapSourceVector(Vector3 value) => new(value.x, value.z, value.y);

    private static float[] QuaternionXyzw(Quaternion value) =>
        new[] { value.x, value.y, value.z, value.w };

    private static VectorRecord[] RotateBasis(Quaternion value) => new[]
    {
        new VectorRecord(value * Vector3.right),
        new VectorRecord(value * Vector3.up),
        new VectorRecord(value * Vector3.forward),
    };

    private static string GetOutputPath()
    {
        foreach (var argument in Environment.GetCommandLineArgs())
        {
            const string prefix = "-QuatOpsCaptureOutput=";
            if (argument.StartsWith(prefix, StringComparison.Ordinal))
                return Path.GetFullPath(argument.Substring(prefix.Length));
        }
        return Path.GetFullPath(Path.Combine(Path.GetTempPath(), "quatops-unity-witnesses.json"));
    }
}
