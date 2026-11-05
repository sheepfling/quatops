#include "Containers/UnrealString.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformProperties.h"
#include "Math/Quat.h"
#include "Math/Vector.h"
#include "Misc/AutomationTest.h"
#include "Misc/CommandLine.h"
#include "Misc/EngineVersion.h"
#include "Misc/FileHelper.h"
#include "Misc/Parse.h"
#include "Misc/Paths.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"

#include <initializer_list>

namespace
{
    using FJsonObjectRef = TSharedPtr<FJsonObject>;
    using FJsonValueRef = TSharedPtr<FJsonValue>;

    struct FCanonicalQuaternionCase
    {
        const TCHAR* CaseId;
        double W;
        double X;
        double Y;
        double Z;
    };

    constexpr FCanonicalQuaternionCase CanonicalQuaternionCases[] = {
        { TEXT("identity"), 1.0, 0.0, 0.0, 0.0 },
        { TEXT("quarter_x"), 0.7071067811865476, 0.7071067811865476, 0.0, 0.0 },
        { TEXT("quarter_y"), 0.7071067811865476, 0.0, 0.7071067811865476, 0.0 },
        { TEXT("quarter_z"), 0.7071067811865476, 0.0, 0.0, 0.7071067811865476 },
        { TEXT("mixed_a"), 0.8, 0.2, -0.4, 0.4 },
        { TEXT("mixed_b"), 0.7, -0.3, 0.2, 0.6164414002968976 },
        { TEXT("antipode_mixed_a"), -0.8, -0.2, 0.4, -0.4 },
        { TEXT("nonprincipal_halfturn"), 0.0, 0.5773502691896258, 0.5773502691896258, 0.5773502691896258 },
    };

    FJsonValueRef Number(double Value)
    {
        return MakeShared<FJsonValueNumber>(Value);
    }

    TArray<FJsonValueRef> NumberArray(std::initializer_list<double> Values)
    {
        TArray<FJsonValueRef> Result;
        Result.Reserve(static_cast<int32>(Values.size()));
        for (double Value : Values)
        {
            Result.Add(Number(Value));
        }
        return Result;
    }

    template<typename T>
    TArray<FJsonValueRef> VectorArray(const UE::Math::TVector<T>& Value)
    {
        return NumberArray({
            static_cast<double>(Value.X),
            static_cast<double>(Value.Y),
            static_cast<double>(Value.Z),
        });
    }

    template<typename T>
    TArray<FJsonValueRef> QuaternionArray(const UE::Math::TQuat<T>& Value)
    {
        return NumberArray({
            static_cast<double>(Value.X),
            static_cast<double>(Value.Y),
            static_cast<double>(Value.Z),
            static_cast<double>(Value.W),
        });
    }

    template<typename T>
    FJsonObjectRef CaptureOrientation(
        const FCanonicalQuaternionCase& Input,
        const UE::Math::TQuat<T>& Native,
        const UE::Math::TVector<T>& NativeProbe)
    {
        using TVector = UE::Math::TVector<T>;
        const TVector Axes[] = { TVector(1, 0, 0), TVector(0, 1, 0), TVector(0, 0, 1) };
        TArray<FJsonValueRef> BasisImages;
        for (const TVector& Axis : Axes)
        {
            BasisImages.Add(MakeShared<FJsonValueArray>(VectorArray(Native.RotateVector(Axis))));
        }

        FJsonObjectRef Result = MakeShared<FJsonObject>();
        Result->SetStringField(TEXT("case_id"), Input.CaseId);
        Result->SetArrayField(TEXT("canonical_wxyz"), NumberArray({ Input.W, Input.X, Input.Y, Input.Z }));
        Result->SetArrayField(TEXT("native_quaternion"), QuaternionArray(Native));
        Result->SetArrayField(TEXT("native_rotated_positive_basis_vectors_xyz"), BasisImages);
        Result->SetArrayField(TEXT("native_rotated_probe_vector_xyz"), VectorArray(Native.RotateVector(NativeProbe)));
        return Result;
    }

    template<typename T>
    FJsonObjectRef CaptureComposition(
        const TCHAR* LeftId,
        const TCHAR* RightId,
        const UE::Math::TQuat<T>& Product,
        const UE::Math::TVector<T>& NativeProbe)
    {
        using TVector = UE::Math::TVector<T>;
        const TVector Axes[] = { TVector(1, 0, 0), TVector(0, 1, 0), TVector(0, 0, 1) };
        TArray<FJsonValueRef> BasisImages;
        for (const TVector& Axis : Axes)
        {
            BasisImages.Add(MakeShared<FJsonValueArray>(VectorArray(Product.RotateVector(Axis))));
        }

        FJsonObjectRef Result = MakeShared<FJsonObject>();
        Result->SetStringField(TEXT("left_case_id"), LeftId);
        Result->SetStringField(TEXT("right_case_id"), RightId);
        Result->SetStringField(TEXT("product_order"), TEXT("lhs * rhs; rhs acts first"));
        Result->SetArrayField(TEXT("native_product_quaternion"), QuaternionArray(Product));
        Result->SetArrayField(TEXT("native_rotated_positive_basis_vectors_xyz"), BasisImages);
        Result->SetArrayField(TEXT("native_rotated_probe_vector_xyz"), VectorArray(Product.RotateVector(NativeProbe)));
        return Result;
    }

    template<typename T>
    FJsonObjectRef CapturePrecisionProfile()
    {
        using TVector = UE::Math::TVector<T>;
        using TQuat = UE::Math::TQuat<T>;

        // Unreal world axes are +X forward, +Y right, +Z up. The explicit
        // source-to-target map is (y, x, z), with determinant -1.
        const auto MapQuaternion = [](const TQuat& Source) -> TQuat
        {
            return TQuat(-Source.Y, -Source.X, -Source.Z, Source.W);
        };
        const auto MapVector = [](const TVector& Source) -> TVector
        {
            return TVector(Source.Y, Source.X, Source.Z);
        };

        const TVector CanonicalProbe(static_cast<T>(0.25), static_cast<T>(-1.5), static_cast<T>(2.75));
        const TVector NativeProbe = MapVector(CanonicalProbe);
        TArray<FJsonValueRef> Orientations;
        TQuat NativeLeft = TQuat::Identity;
        TQuat NativeRight = TQuat::Identity;
        for (const FCanonicalQuaternionCase& Input : CanonicalQuaternionCases)
        {
            const TQuat Source(static_cast<T>(Input.X), static_cast<T>(Input.Y), static_cast<T>(Input.Z), static_cast<T>(Input.W));
            const TQuat Native = MapQuaternion(Source);
            Orientations.Add(MakeShared<FJsonValueObject>(CaptureOrientation(Input, Native, NativeProbe)));
            if (FCString::Strcmp(Input.CaseId, TEXT("mixed_a")) == 0) NativeLeft = Native;
            if (FCString::Strcmp(Input.CaseId, TEXT("mixed_b")) == 0) NativeRight = Native;
        }

        FJsonObjectRef Result = MakeShared<FJsonObject>();
        Result->SetArrayField(TEXT("probe_vector_canonical_xyz"), VectorArray(CanonicalProbe));
        Result->SetArrayField(TEXT("probe_vector_native_xyz"), VectorArray(NativeProbe));
        Result->SetArrayField(TEXT("orientation_cases"), Orientations);
        Result->SetObjectField(TEXT("composition"), CaptureComposition(
            TEXT("mixed_a"), TEXT("mixed_b"), NativeLeft * NativeRight, NativeProbe));
        Result->SetObjectField(TEXT("reverse_composition"), CaptureComposition(
            TEXT("mixed_b"), TEXT("mixed_a"), NativeRight * NativeLeft, NativeProbe));
        return Result;
    }

    FJsonObjectRef CaptureFrame(const TCHAR* Handedness, const TCHAR* X, const TCHAR* Y, const TCHAR* Z)
    {
        FJsonObjectRef Result = MakeShared<FJsonObject>();
        Result->SetStringField(TEXT("handedness"), Handedness);
        Result->SetStringField(TEXT("positive_x"), X);
        Result->SetStringField(TEXT("positive_y"), Y);
        Result->SetStringField(TEXT("positive_z"), Z);
        return Result;
    }

    FJsonObjectRef CaptureFixture()
    {
        const FEngineVersion Version = FEngineVersion::Current();
        FJsonObjectRef Engine = MakeShared<FJsonObject>();
        Engine->SetStringField(TEXT("name"), TEXT("Unreal Engine"));
        Engine->SetStringField(TEXT("version"), Version.ToString());
        Engine->SetNumberField(TEXT("major"), Version.GetMajor());
        Engine->SetNumberField(TEXT("minor"), Version.GetMinor());
        Engine->SetNumberField(TEXT("patch"), Version.GetPatch());
        Engine->SetNumberField(TEXT("changelist"), Version.GetChangelist());
        Engine->SetStringField(TEXT("branch"), Version.GetBranchDescriptor());
        Engine->SetStringField(TEXT("platform"), UTF8_TO_TCHAR(FPlatformProperties::PlatformName()));
        Engine->SetStringField(TEXT("build_target"), TEXT("QuatOpsUnrealCaptureEditor Win64 Development"));
#if PLATFORM_CPU_ARM_FAMILY
        Engine->SetStringField(TEXT("platform_architecture"), TEXT("arm64"));
#else
        Engine->SetStringField(TEXT("platform_architecture"), TEXT("x64"));
#endif
        Engine->SetStringField(TEXT("precision_types"), TEXT("FQuat4f and FQuat4d"));

        FJsonObjectRef NativeSemantics = MakeShared<FJsonObject>();
        NativeSemantics->SetObjectField(TEXT("source_frame"), CaptureFrame(
            TEXT("right-handed"), TEXT("right"), TEXT("forward"), TEXT("up")));
        NativeSemantics->SetObjectField(TEXT("unreal_world_frame"), CaptureFrame(
            TEXT("left-handed"), TEXT("forward"), TEXT("right"), TEXT("up")));
        FJsonObjectRef AxisMap = MakeShared<FJsonObject>();
        AxisMap->SetArrayField(TEXT("destination_axes_from_source"), {
            MakeShared<FJsonValueString>(TEXT("y")),
            MakeShared<FJsonValueString>(TEXT("x")),
            MakeShared<FJsonValueString>(TEXT("z")),
        });
        AxisMap->SetNumberField(TEXT("determinant"), -1);
        NativeSemantics->SetObjectField(TEXT("source_to_unreal_axis_map"), AxisMap);

        FJsonObjectRef CanonicalInterop = MakeShared<FJsonObject>();
        CanonicalInterop->SetStringField(TEXT("contract_id"), TEXT("quatops-canonical-hamilton-wxyz-active-rh-v1"));
        CanonicalInterop->SetStringField(TEXT("native_quaternion_order"), TEXT("xyzw"));
        CanonicalInterop->SetArrayField(TEXT("frame_map_destination_axes_from_source"), {
            MakeShared<FJsonValueString>(TEXT("y")),
            MakeShared<FJsonValueString>(TEXT("x")),
            MakeShared<FJsonValueString>(TEXT("z")),
        });
        FJsonObjectRef PrecisionCaptures = MakeShared<FJsonObject>();
        PrecisionCaptures->SetObjectField(TEXT("float32"), CapturePrecisionProfile<float>());
        PrecisionCaptures->SetObjectField(TEXT("float64"), CapturePrecisionProfile<double>());
        CanonicalInterop->SetObjectField(TEXT("precision_captures"), PrecisionCaptures);

        FJsonObjectRef Root = MakeShared<FJsonObject>();
        Root->SetNumberField(TEXT("schema_version"), 1);
        Root->SetStringField(TEXT("evidence_kind"), TEXT("native_unreal_runtime_capture"));
        Root->SetStringField(TEXT("capture_method"), TEXT("C++ Unreal Editor Automation Test"));
        Root->SetObjectField(TEXT("engine"), Engine);
        Root->SetObjectField(TEXT("native_semantics"), NativeSemantics);
        Root->SetObjectField(TEXT("canonical_interop"), CanonicalInterop);
        return Root;
    }
}

IMPLEMENT_SIMPLE_AUTOMATION_TEST(
    FQuatOpsUnrealFixtureCaptureTest,
    "QuatOps.Unreal.CaptureGoldenFixtures",
    EAutomationTestFlags::EditorContext | EAutomationTestFlags::EngineFilter)

bool FQuatOpsUnrealFixtureCaptureTest::RunTest(const FString& Parameters)
{
    const FJsonObjectRef Fixture = CaptureFixture();
    FString OutputPath;
    if (!FParse::Value(FCommandLine::Get(), TEXT("QuatOpsCaptureOutput="), OutputPath))
    {
        OutputPath = FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("quatops_unreal_witnesses.json"));
    }
    FPaths::NormalizeFilename(OutputPath);
    IFileManager::Get().MakeDirectory(*FPaths::GetPath(OutputPath), true);

    FString Json;
    const TSharedRef<TJsonWriter<TCHAR, TPrettyJsonPrintPolicy<TCHAR>>> Writer =
        TJsonWriterFactory<TCHAR, TPrettyJsonPrintPolicy<TCHAR>>::Create(&Json);
    if (!FJsonSerializer::Serialize(Fixture.ToSharedRef(), Writer))
    {
        AddError(TEXT("Unreal witness JSON serialization failed."));
        return false;
    }
    Json.AppendChar(TEXT('\n'));
    if (!FFileHelper::SaveStringToFile(Json, *OutputPath, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM))
    {
        AddError(FString::Printf(TEXT("Could not write Unreal capture to %s."), *OutputPath));
        return false;
    }
    AddInfo(FString::Printf(TEXT("Captured selected Unreal quaternion witnesses to %s."), *OutputPath));
    return true;
}