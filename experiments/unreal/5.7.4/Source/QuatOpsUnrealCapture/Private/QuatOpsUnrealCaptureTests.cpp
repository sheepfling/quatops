#include "Containers/UnrealString.h"
#include "HAL/FileManager.h"
#include "HAL/PlatformProperties.h"
#include "Math/Matrix.h"
#include "Math/Quat.h"
#include "Math/Rotator.h"
#include "Math/Transform.h"
#include "Math/Vector.h"
#include "Misc/AutomationTest.h"
#include "Misc/CommandLine.h"
#include "Misc/EngineVersion.h"
#include "Misc/FileHelper.h"
#include "Misc/Parse.h"
#include "Misc/Paths.h"
#include "Serialization/JsonSerializer.h"
#include "Serialization/JsonWriter.h"

#include <cmath>

namespace
{
	struct FRotatorCase
	{
		const TCHAR* CaseId;
		double Pitch;
		double Yaw;
		double Roll;
	};

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

	constexpr FRotatorCase Cases[] =
	{
		{ TEXT("identity"), 0.0, 0.0, 0.0 },
		{ TEXT("pitch_positive_90"), 90.0, 0.0, 0.0 },
		{ TEXT("yaw_positive_90"), 0.0, 90.0, 0.0 },
		{ TEXT("roll_positive_90"), 0.0, 0.0, 90.0 },
		{ TEXT("mixed_positive"), 31.41, -49.16, 12.34 },
		{ TEXT("mixed_negative"), -50.0, 28.0, 39.0 },
		{ TEXT("near_gimbal"), 89.9, 27.0, -41.0 },
		{ TEXT("gimbal_threshold_below_lower"), 89.91, 0.0, 0.0 },
		{ TEXT("gimbal_threshold_above_lower"), 89.93, 0.0, 0.0 },
		{ TEXT("gimbal_threshold_below_upper"), 90.07, 0.0, 0.0 },
		{ TEXT("gimbal_threshold_above_upper"), 90.09, 0.0, 0.0 },
		{ TEXT("gimbal"), 90.0, 27.0, -41.0 },
		{ TEXT("pitch_over_90"), 105.0, 23.0, -17.0 },
		{ TEXT("yaw_positive_180"), 0.0, 180.0, 0.0 },
		{ TEXT("yaw_negative_180"), 0.0, -180.0, 0.0 },
	};

	struct FProductCase
	{
		const TCHAR* LhsCaseId;
		const TCHAR* RhsCaseId;
	};

	constexpr FProductCase ProductCases[] =
	{
		{ TEXT("pitch_positive_90"), TEXT("yaw_positive_90") },
		{ TEXT("yaw_positive_90"), TEXT("roll_positive_90") },
		{ TEXT("mixed_positive"), TEXT("mixed_negative") },
	};

	using FJsonObjectRef = TSharedPtr<FJsonObject>;
	using FJsonValueRef = TSharedPtr<FJsonValue>;

	FJsonValueRef Number(double Value)
	{
		return MakeShared<FJsonValueNumber>(Value);
	}

	TArray<FJsonValueRef> NumberArray(std::initializer_list<double> Values)
	{
		TArray<FJsonValueRef> Result;
		Result.Reserve(static_cast<int32>(Values.size()));
		for (const double Value : Values)
		{
			Result.Add(Number(Value));
		}
		return Result;
	}

	template<typename T>
	FJsonObjectRef VectorJson(const UE::Math::TVector<T>& Value)
	{
		FJsonObjectRef Result = MakeShared<FJsonObject>();
		Result->SetNumberField(TEXT("x"), static_cast<double>(Value.X));
		Result->SetNumberField(TEXT("y"), static_cast<double>(Value.Y));
		Result->SetNumberField(TEXT("z"), static_cast<double>(Value.Z));
		return Result;
	}

	template<typename T>
	FJsonObjectRef RotatorJson(const UE::Math::TRotator<T>& Value)
	{
		FJsonObjectRef Result = MakeShared<FJsonObject>();
		Result->SetNumberField(TEXT("pitch"), static_cast<double>(Value.Pitch));
		Result->SetNumberField(TEXT("yaw"), static_cast<double>(Value.Yaw));
		Result->SetNumberField(TEXT("roll"), static_cast<double>(Value.Roll));
		return Result;
	}

	FJsonObjectRef InputRotatorJson(const FRotatorCase& Value)
	{
		FJsonObjectRef Result = MakeShared<FJsonObject>();
		Result->SetNumberField(TEXT("pitch"), Value.Pitch);
		Result->SetNumberField(TEXT("yaw"), Value.Yaw);
		Result->SetNumberField(TEXT("roll"), Value.Roll);
		return Result;
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
	TArray<FJsonValueRef> VectorArray(const UE::Math::TVector<T>& Value)
	{
		return NumberArray({
			static_cast<double>(Value.X),
			static_cast<double>(Value.Y),
			static_cast<double>(Value.Z),
		});
	}

	template<typename T>
	bool SameRotation(const UE::Math::TQuat<T>& A, const UE::Math::TQuat<T>& B, T Tolerance)
	{
		const T Dot = A.X * B.X + A.Y * B.Y + A.Z * B.Z + A.W * B.W;
		return FMath::Abs(FMath::Abs(Dot) - T(1)) <= Tolerance;
	}

	template<typename T>
	bool NearlyEqual(const UE::Math::TVector<T>& A, const UE::Math::TVector<T>& B, T Tolerance)
	{
		return std::abs(A.X - B.X) <= Tolerance
			&& std::abs(A.Y - B.Y) <= Tolerance
			&& std::abs(A.Z - B.Z) <= Tolerance;
	}

	template<typename T>
	FJsonObjectRef CaptureCrossProduct(
		const TCHAR* CaseId,
		const UE::Math::TVector<T>& A,
		const UE::Math::TVector<T>& B,
		const UE::Math::TVector<T>& Expected)
	{
		const UE::Math::TVector<T> Cross = UE::Math::TVector<T>::CrossProduct(A, B);
		const UE::Math::TVector<T> Reverse = UE::Math::TVector<T>::CrossProduct(B, A);
		FJsonObjectRef Result = MakeShared<FJsonObject>();
		Result->SetStringField(TEXT("case_id"), CaseId);
		Result->SetArrayField(TEXT("lhs_xyz"), VectorArray(A));
		Result->SetArrayField(TEXT("rhs_xyz"), VectorArray(B));
		Result->SetArrayField(TEXT("result_xyz"), VectorArray(Cross));
		Result->SetArrayField(TEXT("reversed_result_xyz"), VectorArray(Reverse));
		Result->SetBoolField(TEXT("matches_numeric_basis_identity"), NearlyEqual(Cross, Expected, static_cast<T>(1.e-6)));
		Result->SetBoolField(TEXT("reversed_is_negative"), NearlyEqual(Reverse, -Expected, static_cast<T>(1.e-6)));
		return Result;
	}

	template<typename T>
	TArray<FJsonValueRef> CaptureCrossProducts()
	{
		using TVector = UE::Math::TVector<T>;
		const TVector X(1, 0, 0);
		const TVector Y(0, 1, 0);
		const TVector Z(0, 0, 1);
		TArray<FJsonValueRef> Result;
		Result.Add(MakeShared<FJsonValueObject>(CaptureCrossProduct<T>(TEXT("x_cross_y"), X, Y, Z)));
		Result.Add(MakeShared<FJsonValueObject>(CaptureCrossProduct<T>(TEXT("y_cross_z"), Y, Z, X)));
		Result.Add(MakeShared<FJsonValueObject>(CaptureCrossProduct<T>(TEXT("z_cross_x"), Z, X, Y)));
		return Result;
	}

	template<typename T>
	FJsonObjectRef CaptureQuaternionComposition(const TMap<FString, UE::Math::TQuat<T>>& QuatsByCase)
	{
		using TVector = UE::Math::TVector<T>;
		using TQuat = UE::Math::TQuat<T>;
		const TQuat& Lhs = QuatsByCase.FindChecked(TEXT("pitch_positive_90"));
		const TQuat& Rhs = QuatsByCase.FindChecked(TEXT("yaw_positive_90"));
		const TVector Probe(1.25, -2.5, 3.75);
		const TQuat Product = Lhs * Rhs;
		const TVector ProductImage = Product.RotateVector(Probe);
		const TVector RightThenLeft = Lhs.RotateVector(Rhs.RotateVector(Probe));
		const TVector LeftThenRight = Rhs.RotateVector(Lhs.RotateVector(Probe));

		FJsonObjectRef Result = MakeShared<FJsonObject>();
		Result->SetStringField(TEXT("lhs_case_id"), TEXT("pitch_positive_90"));
		Result->SetStringField(TEXT("rhs_case_id"), TEXT("yaw_positive_90"));
		Result->SetArrayField(TEXT("probe_vector_xyz"), VectorArray(Probe));
		Result->SetArrayField(TEXT("lhs_xyzw"), QuaternionArray(Lhs));
		Result->SetArrayField(TEXT("rhs_xyzw"), QuaternionArray(Rhs));
		Result->SetArrayField(TEXT("lhs_times_rhs_xyzw"), QuaternionArray(Product));
		Result->SetArrayField(TEXT("lhs_times_rhs_rotated_vector_xyz"), VectorArray(ProductImage));
		Result->SetArrayField(TEXT("rhs_then_lhs_rotated_vector_xyz"), VectorArray(RightThenLeft));
		Result->SetArrayField(TEXT("lhs_then_rhs_rotated_vector_xyz"), VectorArray(LeftThenRight));
		Result->SetBoolField(TEXT("product_matches_rhs_then_lhs"), NearlyEqual(ProductImage, RightThenLeft, static_cast<T>(1.e-5)));
		Result->SetBoolField(TEXT("product_matches_lhs_then_rhs"), NearlyEqual(ProductImage, LeftThenRight, static_cast<T>(1.e-5)));
		return Result;
	}

	template<typename T>
	FJsonObjectRef CaptureTransformComposition(const TMap<FString, UE::Math::TQuat<T>>& QuatsByCase)
	{
		using TVector = UE::Math::TVector<T>;
		using TQuat = UE::Math::TQuat<T>;
		using TTransform = UE::Math::TTransform<T>;
		const TTransform Lhs(
			QuatsByCase.FindChecked(TEXT("pitch_positive_90")),
			TVector(3, -2, 5),
			TVector(1, 1, 1));
		const TTransform Rhs(
			QuatsByCase.FindChecked(TEXT("yaw_positive_90")),
			TVector(-7, 11, 13),
			TVector(1, 1, 1));
		const TVector Probe(2, 3, 5);
		const TTransform Product = Lhs * Rhs;
		const TVector ProductImage = Product.TransformPosition(Probe);
		const TVector LhsThenRhs = Rhs.TransformPosition(Lhs.TransformPosition(Probe));
		const TVector RhsThenLhs = Lhs.TransformPosition(Rhs.TransformPosition(Probe));

		FJsonObjectRef Result = MakeShared<FJsonObject>();
		Result->SetStringField(TEXT("lhs_rotation_case_id"), TEXT("pitch_positive_90"));
		Result->SetStringField(TEXT("rhs_rotation_case_id"), TEXT("yaw_positive_90"));
		Result->SetArrayField(TEXT("lhs_translation_xyz"), VectorArray(Lhs.GetTranslation()));
		Result->SetArrayField(TEXT("rhs_translation_xyz"), VectorArray(Rhs.GetTranslation()));
		Result->SetArrayField(TEXT("scale_xyz"), VectorArray(Lhs.GetScale3D()));
		Result->SetArrayField(TEXT("probe_position_xyz"), VectorArray(Probe));
		Result->SetArrayField(TEXT("lhs_times_rhs_rotation_xyzw"), QuaternionArray(Product.GetRotation()));
		Result->SetArrayField(TEXT("lhs_times_rhs_translation_xyz"), VectorArray(Product.GetTranslation()));
		Result->SetArrayField(TEXT("lhs_times_rhs_position_xyz"), VectorArray(ProductImage));
		Result->SetArrayField(TEXT("lhs_then_rhs_position_xyz"), VectorArray(LhsThenRhs));
		Result->SetArrayField(TEXT("rhs_then_lhs_position_xyz"), VectorArray(RhsThenLhs));
		Result->SetBoolField(TEXT("product_matches_lhs_then_rhs"), NearlyEqual(ProductImage, LhsThenRhs, static_cast<T>(1.e-4)));
		Result->SetBoolField(TEXT("product_matches_rhs_then_lhs"), NearlyEqual(ProductImage, RhsThenLhs, static_cast<T>(1.e-4)));
		return Result;
	}

	template<typename T>
	FJsonObjectRef CaptureMappedSourceRotation(const TCHAR* AxisName, const UE::Math::TQuat<T>& SourceQuat)
	{
		using TVector = UE::Math::TVector<T>;
		using TQuat = UE::Math::TQuat<T>;
		// Source RH axes are +X right, +Y forward, +Z up. Unreal's documented
		// world axes are +X forward, +Y right, +Z up, so B swaps X and Y and has det -1.
		const TQuat UnrealQuat(-SourceQuat.Y, -SourceQuat.X, -SourceQuat.Z, SourceQuat.W);
		const auto MapSourceVector = [](const TVector& Value) -> TVector
		{
			return TVector(Value.Y, Value.X, Value.Z);
		};
		const TVector Axes[] = {
			UnrealQuat.RotateVector(TVector(1, 0, 0)),
			UnrealQuat.RotateVector(TVector(0, 1, 0)),
			UnrealQuat.RotateVector(TVector(0, 0, 1)),
		};
		TArray<FJsonValueRef> AxisValues;
		for (const TVector& Axis : Axes)
		{
			AxisValues.Add(MakeShared<FJsonValueArray>(VectorArray(Axis)));
		}
		const TVector SourceBasis[] = { TVector(1, 0, 0), TVector(0, 1, 0), TVector(0, 0, 1) };
		bool PreservesMappedAction = true;
		for (const TVector& Axis : SourceBasis)
		{
			const TVector Expected = MapSourceVector(SourceQuat.RotateVector(Axis));
			const TVector Actual = UnrealQuat.RotateVector(MapSourceVector(Axis));
			PreservesMappedAction = PreservesMappedAction && NearlyEqual(Expected, Actual, static_cast<T>(1.e-6));
		}

		FJsonObjectRef Result = MakeShared<FJsonObject>();
		Result->SetStringField(TEXT("source_rotation_axis"), AxisName);
		Result->SetArrayField(TEXT("source_quaternion_xyzw"), QuaternionArray(SourceQuat));
		Result->SetArrayField(TEXT("unreal_quaternion_xyzw"), QuaternionArray(UnrealQuat));
		Result->SetArrayField(TEXT("unreal_rotated_positive_basis_vectors_xyz"), AxisValues);
		Result->SetObjectField(TEXT("unreal_rotator_readback_degrees"), RotatorJson(UnrealQuat.Rotator()));
		Result->SetBoolField(TEXT("preserves_mapped_rotation_action"), PreservesMappedAction);
		return Result;
	}

	template<typename T>
	TArray<FJsonValueRef> CaptureMappedSourceRotations()
	{
		using TQuat = UE::Math::TQuat<T>;
		const T HalfSqrt = static_cast<T>(std::sqrt(0.5));
		TArray<FJsonValueRef> Result;
		Result.Add(MakeShared<FJsonValueObject>(CaptureMappedSourceRotation<T>(TEXT("+X right"), TQuat(HalfSqrt, 0, 0, HalfSqrt))));
		Result.Add(MakeShared<FJsonValueObject>(CaptureMappedSourceRotation<T>(TEXT("+Y forward"), TQuat(0, HalfSqrt, 0, HalfSqrt))));
		Result.Add(MakeShared<FJsonValueObject>(CaptureMappedSourceRotation<T>(TEXT("+Z up"), TQuat(0, 0, HalfSqrt, HalfSqrt))));
		return Result;
	}

	template<typename T>
	FJsonObjectRef CaptureCanonicalInteropOrientation(
		const TCHAR* CaseId,
		std::initializer_list<double> CanonicalWxyz,
		const UE::Math::TQuat<T>& NativeQuat,
		const UE::Math::TVector<T>& NativeProbe)
	{
		using TVector = UE::Math::TVector<T>;
		FJsonObjectRef Result = MakeShared<FJsonObject>();
		TArray<FJsonValueRef> BasisImages;
		for (const TVector& Axis : { TVector(1, 0, 0), TVector(0, 1, 0), TVector(0, 0, 1) })
		{
			BasisImages.Add(MakeShared<FJsonValueArray>(VectorArray(NativeQuat.RotateVector(Axis))));
		}
		Result->SetStringField(TEXT("case_id"), CaseId);
		Result->SetArrayField(TEXT("canonical_wxyz"), NumberArray(CanonicalWxyz));
		Result->SetArrayField(TEXT("native_quaternion"), QuaternionArray(NativeQuat));
		Result->SetArrayField(TEXT("native_rotated_positive_basis_vectors_xyz"), BasisImages);
		Result->SetArrayField(TEXT("native_rotated_probe_vector_xyz"), VectorArray(NativeQuat.RotateVector(NativeProbe)));
		return Result;
	}

	template<typename T>
	FJsonObjectRef CaptureCanonicalInterop()
	{
		using TVector = UE::Math::TVector<T>;
		using TQuat = UE::Math::TQuat<T>;
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
		TQuat NativeLeft = TQuat::Identity;
		TQuat NativeRight = TQuat::Identity;

		TArray<FJsonValueRef> FrameMap;
		FrameMap.Add(MakeShared<FJsonValueString>(TEXT("y")));
		FrameMap.Add(MakeShared<FJsonValueString>(TEXT("x")));
		FrameMap.Add(MakeShared<FJsonValueString>(TEXT("z")));
		TArray<FJsonValueRef> Orientations;
		for (const FCanonicalQuaternionCase& Case : CanonicalQuaternionCases)
		{
			const TQuat Source(static_cast<T>(Case.X), static_cast<T>(Case.Y), static_cast<T>(Case.Z), static_cast<T>(Case.W));
			const TQuat Native = MapQuaternion(Source);
			Orientations.Add(MakeShared<FJsonValueObject>(CaptureCanonicalInteropOrientation<T>(
				Case.CaseId, { Case.W, Case.X, Case.Y, Case.Z }, Native, NativeProbe)));
			if (FCString::Strcmp(Case.CaseId, TEXT("mixed_a")) == 0) NativeLeft = Native;
			if (FCString::Strcmp(Case.CaseId, TEXT("mixed_b")) == 0) NativeRight = Native;
		}

		const auto MakeComposition = [&NativeProbe](const TCHAR* LeftId, const TCHAR* RightId, const TQuat& Product) -> FJsonObjectRef
		{
			FJsonObjectRef Composition = MakeShared<FJsonObject>();
			TArray<FJsonValueRef> ProductBasisImages;
			for (const TVector& Axis : { TVector(1, 0, 0), TVector(0, 1, 0), TVector(0, 0, 1) })
			{
				ProductBasisImages.Add(MakeShared<FJsonValueArray>(VectorArray(Product.RotateVector(Axis))));
			}
			Composition->SetStringField(TEXT("left_case_id"), LeftId);
			Composition->SetStringField(TEXT("right_case_id"), RightId);
			Composition->SetStringField(TEXT("product_order"), TEXT("lhs * rhs; rhs acts first"));
			Composition->SetArrayField(TEXT("native_product_quaternion"), QuaternionArray(Product));
			Composition->SetArrayField(TEXT("native_rotated_positive_basis_vectors_xyz"), ProductBasisImages);
			Composition->SetArrayField(TEXT("native_rotated_probe_vector_xyz"), VectorArray(Product.RotateVector(NativeProbe)));
			return Composition;
		};

		FJsonObjectRef Result = MakeShared<FJsonObject>();
		Result->SetStringField(TEXT("contract_id"), TEXT("quatops-canonical-hamilton-wxyz-active-rh-v1"));
		Result->SetStringField(TEXT("native_quaternion_order"), TEXT("xyzw"));
		Result->SetArrayField(TEXT("frame_map_destination_axes_from_source"), FrameMap);
		Result->SetArrayField(TEXT("probe_vector_canonical_xyz"), VectorArray(CanonicalProbe));
		Result->SetArrayField(TEXT("probe_vector_native_xyz"), VectorArray(NativeProbe));
		Result->SetArrayField(TEXT("orientation_cases"), Orientations);
		Result->SetObjectField(TEXT("composition"), MakeComposition(TEXT("mixed_a"), TEXT("mixed_b"), NativeLeft * NativeRight));
		Result->SetObjectField(TEXT("reverse_composition"), MakeComposition(TEXT("mixed_b"), TEXT("mixed_a"), NativeRight * NativeLeft));
		return Result;
	}

	FJsonObjectRef CaptureNativeSemantics(
		const TMap<FString, FQuat4f>& Quats32,
		const TMap<FString, FQuat4d>& Quats64)
	{
		FJsonObjectRef SourceFrame = MakeShared<FJsonObject>();
		SourceFrame->SetStringField(TEXT("handedness"), TEXT("right-handed"));
		SourceFrame->SetStringField(TEXT("positive_x"), TEXT("right"));
		SourceFrame->SetStringField(TEXT("positive_y"), TEXT("forward"));
		SourceFrame->SetStringField(TEXT("positive_z"), TEXT("up"));

		FJsonObjectRef UnrealFrame = MakeShared<FJsonObject>();
		UnrealFrame->SetStringField(TEXT("handedness"), TEXT("left-handed"));
		UnrealFrame->SetStringField(TEXT("positive_x"), TEXT("forward"));
		UnrealFrame->SetStringField(TEXT("positive_y"), TEXT("right"));
		UnrealFrame->SetStringField(TEXT("positive_z"), TEXT("up"));

		FJsonObjectRef AxisMap = MakeShared<FJsonObject>();
		AxisMap->SetArrayField(TEXT("destination_axes_from_source"), {
			MakeShared<FJsonValueString>(TEXT("y")),
			MakeShared<FJsonValueString>(TEXT("x")),
			MakeShared<FJsonValueString>(TEXT("z")),
		});
		AxisMap->SetNumberField(TEXT("determinant"), -1);
		AxisMap->SetStringField(TEXT("meaning"), TEXT("Unreal X=source Y, Unreal Y=source X, Unreal Z=source Z"));
		AxisMap->SetArrayField(TEXT("mapped_source_principal_rotations_float32"), CaptureMappedSourceRotations<float>());
		AxisMap->SetArrayField(TEXT("mapped_source_principal_rotations_float64"), CaptureMappedSourceRotations<double>());

		FJsonObjectRef Result = MakeShared<FJsonObject>();
		Result->SetObjectField(TEXT("source_frame"), SourceFrame);
		Result->SetObjectField(TEXT("unreal_world_frame"), UnrealFrame);
		Result->SetObjectField(TEXT("source_to_unreal_axis_map"), AxisMap);
		Result->SetArrayField(TEXT("cross_products_float32"), CaptureCrossProducts<float>());
		Result->SetArrayField(TEXT("cross_products_float64"), CaptureCrossProducts<double>());
		Result->SetObjectField(TEXT("quaternion_composition_float32"), CaptureQuaternionComposition<float>(Quats32));
		Result->SetObjectField(TEXT("quaternion_composition_float64"), CaptureQuaternionComposition<double>(Quats64));
		Result->SetObjectField(TEXT("transform_composition_float32"), CaptureTransformComposition<float>(Quats32));
		Result->SetObjectField(TEXT("transform_composition_float64"), CaptureTransformComposition<double>(Quats64));
		return Result;
	}

	template<typename T>
	FJsonObjectRef CapturePrecision(
		const FRotatorCase& Input,
		const TCHAR* RotatorType,
		const TCHAR* QuatType,
		TMap<FString, UE::Math::TQuat<T>>& QuatsByCase)
	{
		UE::Math::TRotator<T> Rotator;
		// Assign the named fields explicitly to avoid the positional-order trap.
		Rotator.Pitch = static_cast<T>(Input.Pitch);
		Rotator.Yaw = static_cast<T>(Input.Yaw);
		Rotator.Roll = static_cast<T>(Input.Roll);

		const UE::Math::TQuat<T> Quat = Rotator.Quaternion();
		const UE::Math::TVector<T> EulerInput(Rotator.Roll, Rotator.Pitch, Rotator.Yaw);
		const UE::Math::TRotator<T> EulerRotator = UE::Math::TRotator<T>::MakeFromEuler(EulerInput);
		const UE::Math::TQuat<T> EulerRotatorQuat = EulerRotator.Quaternion();
		const UE::Math::TQuat<T> EulerQuat = UE::Math::TQuat<T>::MakeFromEuler(EulerInput);
		QuatsByCase.Add(FString(Input.CaseId), Quat);

		FJsonObjectRef Result = MakeShared<FJsonObject>();
		Result->SetStringField(TEXT("rotator_type"), RotatorType);
		Result->SetStringField(TEXT("quat_type"), QuatType);
		Result->SetArrayField(TEXT("rotator_quaternion_xyzw"), QuaternionArray(Quat));
		Result->SetObjectField(TEXT("quat_rotator_readback_degrees"), RotatorJson(Quat.Rotator()));
		Result->SetArrayField(TEXT("make_from_euler_input_xyz_degrees"), VectorArray(EulerInput));
		Result->SetObjectField(TEXT("rotator_make_from_euler_degrees"), RotatorJson(EulerRotator));
		Result->SetArrayField(TEXT("rotator_make_from_euler_quaternion_xyzw"), QuaternionArray(EulerRotatorQuat));
		Result->SetArrayField(TEXT("quat_make_from_euler_xyzw"), QuaternionArray(EulerQuat));
		Result->SetBoolField(TEXT("rotator_make_from_euler_matches_named_fields"),
			EulerRotator.Pitch == Rotator.Pitch && EulerRotator.Yaw == Rotator.Yaw && EulerRotator.Roll == Rotator.Roll);
		Result->SetBoolField(TEXT("quat_make_from_euler_matches_named_fields"),
			SameRotation(Quat, EulerQuat, static_cast<T>(1.e-5)));

		const UE::Math::TVector<T> Euler = Quat.Euler();
		Result->SetArrayField(TEXT("quat_euler_readback_degrees_xyz"), NumberArray({
			static_cast<double>(Euler.X),
			static_cast<double>(Euler.Y),
			static_cast<double>(Euler.Z),
		}));

		const UE::Math::TVector<T> Axes[] =
		{
			Quat.RotateVector(UE::Math::TVector<T>(1, 0, 0)),
			Quat.RotateVector(UE::Math::TVector<T>(0, 1, 0)),
			Quat.RotateVector(UE::Math::TVector<T>(0, 0, 1)),
		};
		TArray<FJsonValueRef> AxisValues;
		for (const UE::Math::TVector<T>& Axis : Axes)
		{
			AxisValues.Add(MakeShared<FJsonValueArray>(NumberArray({
				static_cast<double>(Axis.X),
				static_cast<double>(Axis.Y),
				static_cast<double>(Axis.Z),
			})));
		}
		Result->SetArrayField(TEXT("rotated_positive_basis_vectors_xyz"), AxisValues);

		const UE::Math::TMatrix<T> Matrix = Quat.ToMatrix();
		TArray<FJsonValueRef> MatrixRows;
		for (int32 Row = 0; Row < 3; ++Row)
		{
			MatrixRows.Add(MakeShared<FJsonValueArray>(NumberArray({
				static_cast<double>(Matrix.M[Row][0]),
				static_cast<double>(Matrix.M[Row][1]),
				static_cast<double>(Matrix.M[Row][2]),
			})));
		}
		Result->SetArrayField(TEXT("rotation_matrix_3x3_rows"), MatrixRows);
		return Result;
	}

	template<typename T>
	FJsonObjectRef CaptureProduct(
		const FProductCase& Input,
		const TMap<FString, UE::Math::TQuat<T>>& QuatsByCase)
	{
		const FString LhsId(Input.LhsCaseId);
		const FString RhsId(Input.RhsCaseId);
		const UE::Math::TQuat<T>& Lhs = QuatsByCase.FindChecked(LhsId);
		const UE::Math::TQuat<T>& Rhs = QuatsByCase.FindChecked(RhsId);
		const UE::Math::TQuat<T> Product = Lhs * Rhs;
		FJsonObjectRef Result = MakeShared<FJsonObject>();
		Result->SetStringField(TEXT("lhs_case_id"), LhsId);
		Result->SetStringField(TEXT("rhs_case_id"), RhsId);
		Result->SetArrayField(TEXT("lhs_xyzw"), QuaternionArray(Lhs));
		Result->SetArrayField(TEXT("rhs_xyzw"), QuaternionArray(Rhs));
		Result->SetArrayField(TEXT("product_xyzw"), QuaternionArray(Product));
		Result->SetArrayField(TEXT("product_xyzw_sign_bits"), {
			MakeShared<FJsonValueBoolean>(std::signbit(Product.X)),
			MakeShared<FJsonValueBoolean>(std::signbit(Product.Y)),
			MakeShared<FJsonValueBoolean>(std::signbit(Product.Z)),
			MakeShared<FJsonValueBoolean>(std::signbit(Product.W)),
		});
		return Result;
	}

	template<typename T>
	TArray<FJsonValueRef> CaptureProducts(const TMap<FString, UE::Math::TQuat<T>>& QuatsByCase)
	{
		TArray<FJsonValueRef> Result;
		for (const FProductCase& ProductCase : ProductCases)
		{
			Result.Add(MakeShared<FJsonValueObject>(CaptureProduct(ProductCase, QuatsByCase)));
		}
		return Result;
	}

	template<typename T>
	TArray<FJsonValueRef> MatrixRows(const UE::Math::TMatrix<T>& Matrix)
	{
		TArray<FJsonValueRef> Rows;
		for (int32 Row = 0; Row < 3; ++Row)
		{
			Rows.Add(MakeShared<FJsonValueArray>(NumberArray({
				static_cast<double>(Matrix.M[Row][0]),
				static_cast<double>(Matrix.M[Row][1]),
				static_cast<double>(Matrix.M[Row][2]),
			})));
		}
		return Rows;
	}

	template<typename T>
	FJsonObjectRef CaptureMatrixToQuaternionCase(const TCHAR* CaseId, const UE::Math::TQuat<T>& Source)
	{
		using TMatrix = UE::Math::TMatrix<T>;
		using TQuat = UE::Math::TQuat<T>;
		const TMatrix Matrix = Source.ToMatrix();
		const TQuat Result(Matrix);
		FJsonObjectRef Value = MakeShared<FJsonObject>();
		Value->SetStringField(TEXT("case_id"), CaseId);
		Value->SetArrayField(TEXT("matrix_rows_native"), MatrixRows(Matrix));
		Value->SetNumberField(TEXT("matrix_trace"), static_cast<double>(Matrix.M[0][0] + Matrix.M[1][1] + Matrix.M[2][2]));
		Value->SetArrayField(TEXT("source_quaternion_xyzw"), QuaternionArray(Source));
		Value->SetArrayField(TEXT("constructed_quaternion_xyzw"), QuaternionArray(Result));
		Value->SetNumberField(TEXT("constructed_norm"), static_cast<double>(Result.Size()));
		Value->SetBoolField(TEXT("same_rotation"), SameRotation(Source, Result, static_cast<T>(1.e-5)));
		Value->SetBoolField(TEXT("diagnostic_only"), false);
		return Value;
	}

	template<typename T>
	TArray<FJsonValueRef> CaptureMatrixToQuaternion()
	{
		using TVector = UE::Math::TVector<T>;
		using TMatrix = UE::Math::TMatrix<T>;
		using TQuat = UE::Math::TQuat<T>;
		const T InvSqrt14 = T(1) / FMath::Sqrt(T(14));
		const T DegreesToRadians = static_cast<T>(std::acos(-1.0) / 180.0);
		const TQuat Identity = TQuat::Identity;
		const TQuat X90(TVector(1, 0, 0), T(90) * DegreesToRadians);
		const TQuat Y90(TVector(0, 1, 0), T(90) * DegreesToRadians);
		const TQuat Z90(TVector(0, 0, 1), T(90) * DegreesToRadians);
		const TQuat X180(TVector(1, 0, 0), static_cast<T>(std::acos(-1.0)));
		const TQuat Y180(TVector(0, 1, 0), static_cast<T>(std::acos(-1.0)));
		const TQuat Z180(TVector(0, 0, 1), static_cast<T>(std::acos(-1.0)));
		const TQuat Mixed(TVector(T(1) * InvSqrt14, T(2) * InvSqrt14, T(3) * InvSqrt14), T(1.234));
		const TQuat NegativeTrace(TVector(T(1) * InvSqrt14, T(2) * InvSqrt14, T(3) * InvSqrt14), T(170) * DegreesToRadians);
		TArray<FJsonValueRef> Result;
		Result.Add(MakeShared<FJsonValueObject>(CaptureMatrixToQuaternionCase<T>(TEXT("identity"), Identity)));
		Result.Add(MakeShared<FJsonValueObject>(CaptureMatrixToQuaternionCase<T>(TEXT("x_90"), X90)));
		Result.Add(MakeShared<FJsonValueObject>(CaptureMatrixToQuaternionCase<T>(TEXT("y_90"), Y90)));
		Result.Add(MakeShared<FJsonValueObject>(CaptureMatrixToQuaternionCase<T>(TEXT("z_90"), Z90)));
		Result.Add(MakeShared<FJsonValueObject>(CaptureMatrixToQuaternionCase<T>(TEXT("x_180"), X180)));
		Result.Add(MakeShared<FJsonValueObject>(CaptureMatrixToQuaternionCase<T>(TEXT("y_180"), Y180)));
		Result.Add(MakeShared<FJsonValueObject>(CaptureMatrixToQuaternionCase<T>(TEXT("z_180"), Z180)));
		Result.Add(MakeShared<FJsonValueObject>(CaptureMatrixToQuaternionCase<T>(TEXT("mixed_axis"), Mixed)));
		Result.Add(MakeShared<FJsonValueObject>(CaptureMatrixToQuaternionCase<T>(TEXT("negative_trace"), NegativeTrace)));

		TMatrix Perturbed = Mixed.ToMatrix();
		Perturbed.M[0][0] += T(1.e-6);
		const TQuat PerturbedResult(Perturbed);
		FJsonObjectRef PerturbedCase = MakeShared<FJsonObject>();
		PerturbedCase->SetStringField(TEXT("case_id"), TEXT("slightly_perturbed_diagnostic"));
		PerturbedCase->SetArrayField(TEXT("matrix_rows_native"), MatrixRows(Perturbed));
		PerturbedCase->SetNumberField(TEXT("matrix_trace"), static_cast<double>(Perturbed.M[0][0] + Perturbed.M[1][1] + Perturbed.M[2][2]));
		PerturbedCase->SetArrayField(TEXT("constructed_quaternion_xyzw"), QuaternionArray(PerturbedResult));
		PerturbedCase->SetNumberField(TEXT("constructed_norm"), static_cast<double>(PerturbedResult.Size()));
		PerturbedCase->SetBoolField(TEXT("diagnostic_only"), true);
		Result.Add(MakeShared<FJsonValueObject>(PerturbedCase));

		TMatrix Reflection = TMatrix::Identity;
		Reflection.M[0][0] = T(-1);
		const TQuat ReflectionResult(Reflection);
		FJsonObjectRef ReflectionCase = MakeShared<FJsonObject>();
		ReflectionCase->SetStringField(TEXT("case_id"), TEXT("reflection_diagnostic"));
		ReflectionCase->SetArrayField(TEXT("matrix_rows_native"), MatrixRows(Reflection));
		ReflectionCase->SetArrayField(TEXT("constructed_quaternion_xyzw"), QuaternionArray(ReflectionResult));
		ReflectionCase->SetNumberField(TEXT("constructed_norm"), static_cast<double>(ReflectionResult.Size()));
		ReflectionCase->SetBoolField(TEXT("diagnostic_only"), true);
		Result.Add(MakeShared<FJsonValueObject>(ReflectionCase));

		TMatrix ZeroColumn = TMatrix::Identity;
		ZeroColumn.M[0][0] = T(0);
		ZeroColumn.M[1][0] = T(0);
		ZeroColumn.M[2][0] = T(0);
		const TQuat ZeroColumnResult(ZeroColumn);
		FJsonObjectRef ZeroColumnCase = MakeShared<FJsonObject>();
		ZeroColumnCase->SetStringField(TEXT("case_id"), TEXT("zero_axis_diagnostic"));
		ZeroColumnCase->SetArrayField(TEXT("matrix_rows_native"), MatrixRows(ZeroColumn));
		ZeroColumnCase->SetArrayField(TEXT("constructed_quaternion_xyzw"), QuaternionArray(ZeroColumnResult));
		ZeroColumnCase->SetNumberField(TEXT("constructed_norm"), static_cast<double>(ZeroColumnResult.Size()));
		ZeroColumnCase->SetBoolField(TEXT("diagnostic_only"), true);
		Result.Add(MakeShared<FJsonValueObject>(ZeroColumnCase));
		return Result;
	}

	struct FAxisAngleInput
	{
		const TCHAR* CaseId;
		double AxisX;
		double AxisY;
		double AxisZ;
		double AngleRadians;
	};

	template<typename T>
	FJsonObjectRef CaptureAxisAngleCase(const FAxisAngleInput& Input)
	{
		using TVector = UE::Math::TVector<T>;
		using TQuat = UE::Math::TQuat<T>;
		const TVector Axis(static_cast<T>(Input.AxisX), static_cast<T>(Input.AxisY), static_cast<T>(Input.AxisZ));
		const TQuat Quat(Axis, static_cast<T>(Input.AngleRadians));
		const TQuat NegativeQuat = -Quat;
		TVector AxisReadback;
		T AngleReadback = T(0);
		TVector NegativeAxisReadback;
		T NegativeAngleReadback = T(0);
		Quat.ToAxisAndAngle(AxisReadback, AngleReadback);
		NegativeQuat.ToAxisAndAngle(NegativeAxisReadback, NegativeAngleReadback);
		FJsonObjectRef Value = MakeShared<FJsonObject>();
		Value->SetStringField(TEXT("case_id"), Input.CaseId);
		Value->SetArrayField(TEXT("input_axis_xyz"), VectorArray(Axis));
		Value->SetNumberField(TEXT("input_angle_radians"), static_cast<double>(Input.AngleRadians));
		Value->SetArrayField(TEXT("constructed_quaternion_xyzw"), QuaternionArray(Quat));
		Value->SetArrayField(TEXT("negative_quaternion_xyzw"), QuaternionArray(NegativeQuat));
		Value->SetArrayField(TEXT("axis_angle_readback_axis_xyz"), VectorArray(AxisReadback));
		Value->SetNumberField(TEXT("axis_angle_readback_angle_radians"), static_cast<double>(AngleReadback));
		Value->SetArrayField(TEXT("negative_axis_angle_readback_axis_xyz"), VectorArray(NegativeAxisReadback));
		Value->SetNumberField(TEXT("negative_axis_angle_readback_angle_radians"), static_cast<double>(NegativeAngleReadback));
		Value->SetArrayField(TEXT("rotation_vector_readback_xyz"), VectorArray(Quat.ToRotationVector()));
		Value->SetArrayField(TEXT("negative_rotation_vector_readback_xyz"), VectorArray(NegativeQuat.ToRotationVector()));
		Value->SetBoolField(TEXT("constructor_norm_is_unit"), FMath::Abs(Quat.Size() - T(1)) <= T(1.e-5));
		return Value;
	}

	template<typename T>
	TArray<FJsonValueRef> CaptureAxisAngleCases()
	{
		const double Pi = std::acos(-1.0);
		const double InvSqrt14 = 1.0 / std::sqrt(14.0);
		const FAxisAngleInput Inputs[] = {
			{ TEXT("zero_angle"), 1.0, 0.0, 0.0, 0.0 },
			{ TEXT("tiny_angle"), InvSqrt14, 2.0 * InvSqrt14, 3.0 * InvSqrt14, 1.e-7 },
			{ TEXT("nonprincipal_angle"), InvSqrt14, 2.0 * InvSqrt14, 3.0 * InvSqrt14, 2.4 },
			{ TEXT("exact_pi"), 0.0, 0.0, 1.0, Pi },
			{ TEXT("just_above_pi"), -2.0 / std::sqrt(14.0), 1.0 / std::sqrt(14.0), 3.0 / std::sqrt(14.0), Pi + 0.2 },
		};
		TArray<FJsonValueRef> Result;
		for (const FAxisAngleInput& Input : Inputs)
		{
			Result.Add(MakeShared<FJsonValueObject>(CaptureAxisAngleCase<T>(Input)));
		}
		return Result;
	}

	struct FRotationVectorInput
	{
		const TCHAR* CaseId;
		double X;
		double Y;
		double Z;
	};

	template<typename T>
	FJsonObjectRef CaptureRotationVectorCase(const FRotationVectorInput& Input)
	{
		using TVector = UE::Math::TVector<T>;
		using TQuat = UE::Math::TQuat<T>;
		const TVector Vector(static_cast<T>(Input.X), static_cast<T>(Input.Y), static_cast<T>(Input.Z));
		const TQuat Quat = TQuat::MakeFromRotationVector(Vector);
		TVector Axis;
		T Angle = T(0);
		Quat.ToAxisAndAngle(Axis, Angle);
		FJsonObjectRef Value = MakeShared<FJsonObject>();
		Value->SetStringField(TEXT("case_id"), Input.CaseId);
		Value->SetArrayField(TEXT("input_rotation_vector_xyz"), VectorArray(Vector));
		Value->SetArrayField(TEXT("constructed_quaternion_xyzw"), QuaternionArray(Quat));
		Value->SetArrayField(TEXT("axis_angle_readback_axis_xyz"), VectorArray(Axis));
		Value->SetNumberField(TEXT("axis_angle_readback_angle_radians"), static_cast<double>(Angle));
		Value->SetArrayField(TEXT("rotation_vector_readback_xyz"), VectorArray(Quat.ToRotationVector()));
		Value->SetBoolField(TEXT("constructor_norm_is_unit"), FMath::Abs(Quat.Size() - T(1)) <= T(1.e-5));
		return Value;
	}

	template<typename T>
	TArray<FJsonValueRef> CaptureRotationVectorCases()
	{
		const double Pi = std::acos(-1.0);
		const double InvSqrt14 = 1.0 / std::sqrt(14.0);
		const FRotationVectorInput Inputs[] = {
			{ TEXT("zero"), 0.0, 0.0, 0.0 },
			{ TEXT("tiny"), 1.e-7, -2.e-7, 3.e-7 },
			{ TEXT("mixed"), 0.3, -0.7, 1.2 },
			{ TEXT("exact_pi"), 0.0, 0.0, Pi },
			{ TEXT("just_above_pi"), -2.0 * InvSqrt14 * (Pi + 0.2), InvSqrt14 * (Pi + 0.2), 3.0 * InvSqrt14 * (Pi + 0.2) },
		};
		TArray<FJsonValueRef> Result;
		for (const FRotationVectorInput& Input : Inputs)
		{
			Result.Add(MakeShared<FJsonValueObject>(CaptureRotationVectorCase<T>(Input)));
		}
		return Result;
	}

	struct FSlerpInput
	{
		const TCHAR* CaseId;
		UE::Math::TQuat<double> Lhs;
		UE::Math::TQuat<double> Rhs;
		double Alpha;
	};

	template<typename T>
	FJsonObjectRef CaptureSlerpCase(const FSlerpInput& Input)
	{
		using TQuat = UE::Math::TQuat<T>;
		const TQuat Lhs(Input.Lhs);
		const TQuat Rhs(Input.Rhs);
		const T Alpha = static_cast<T>(Input.Alpha);
		const TQuat Shortest = TQuat::Slerp(Lhs, Rhs, Alpha);
		const TQuat FullPath = TQuat::SlerpFullPath(Lhs, Rhs, Alpha);
		FJsonObjectRef Value = MakeShared<FJsonObject>();
		Value->SetStringField(TEXT("case_id"), Input.CaseId);
		Value->SetNumberField(TEXT("alpha"), static_cast<double>(Input.Alpha));
		Value->SetArrayField(TEXT("lhs_xyzw"), QuaternionArray(Lhs));
		Value->SetArrayField(TEXT("rhs_xyzw"), QuaternionArray(Rhs));
		Value->SetArrayField(TEXT("slerp_xyzw"), QuaternionArray(Shortest));
		Value->SetNumberField(TEXT("slerp_norm"), static_cast<double>(Shortest.Size()));
		Value->SetArrayField(TEXT("slerp_full_path_xyzw"), QuaternionArray(FullPath));
		Value->SetNumberField(TEXT("slerp_full_path_norm"), static_cast<double>(FullPath.Size()));
		Value->SetBoolField(TEXT("shortest_result_is_unit"), FMath::Abs(Shortest.Size() - T(1)) <= T(1.e-5));
		Value->SetBoolField(TEXT("full_path_result_is_unit"), FMath::Abs(FullPath.Size() - T(1)) <= T(1.e-5));
		return Value;
	}

	template<typename T>
	TArray<FJsonValueRef> CaptureSlerpCases()
	{
		using TVector = UE::Math::TVector<double>;
		using TQuat64 = UE::Math::TQuat<double>;
		const double Pi = std::acos(-1.0);
		const double InvSqrt14 = 1.0 / std::sqrt(14.0);
		const TQuat64 Identity = TQuat64::Identity;
		const TQuat64 X90(TVector(1, 0, 0), Pi / 2.0);
		const TQuat64 Z270(TVector(0, 0, 1), 1.5 * Pi);
		const TQuat64 ArbitraryA(TVector(InvSqrt14, 2.0 * InvSqrt14, 3.0 * InvSqrt14), 1.1);
		const TQuat64 ArbitraryB(TVector(-2.0 * InvSqrt14, InvSqrt14, 3.0 * InvSqrt14), 2.2);
		const TQuat64 NearA(TVector(0, 0, 1), 0.8);
		const TQuat64 NearB(TVector(0, 0, 1), 0.801);
		const TQuat64 NearPi(TVector(0, 0, 1), Pi - 1.e-4);
		const TQuat64 MixedA(TVector(InvSqrt14, 2.0 * InvSqrt14, 3.0 * InvSqrt14), 1.3);
		const TQuat64 MixedB(TVector(-2.0 * InvSqrt14, InvSqrt14, 3.0 * InvSqrt14), 2.0);
		TArray<FSlerpInput> Inputs;
		for (const double Alpha : { 0.0, 0.25, 0.5, 0.75, 1.0 })
		{
			Inputs.Add({ TEXT("identity_to_x90"), Identity, X90, Alpha });
			Inputs.Add({ TEXT("long_path"), Identity, Z270, Alpha });
			Inputs.Add({ TEXT("antipodal"), ArbitraryA, -ArbitraryA, Alpha });
		}
		for (const double Alpha : { 0.0, 0.5, 1.0 })
		{
			Inputs.Add({ TEXT("arbitrary_pair"), ArbitraryA, ArbitraryB, Alpha });
		}
		Inputs.Add({ TEXT("nearly_identical"), NearA, NearB, 0.5 });
		Inputs.Add({ TEXT("near_180_separation"), Identity, NearPi, 0.5 });
		Inputs.Add({ TEXT("extrapolation"), Identity, X90, -0.5 });
		Inputs.Add({ TEXT("extrapolation"), Identity, X90, 1.5 });
		TArray<FJsonValueRef> Result;
		for (const FSlerpInput& Input : Inputs)
		{
			Result.Add(MakeShared<FJsonValueObject>(CaptureSlerpCase<T>(Input)));
		}
		return Result;
	}

	template<typename T>
	FJsonObjectRef CaptureNormalizationCase(const TCHAR* CaseId, const UE::Math::TQuat<T>& Input)
	{
		const UE::Math::TQuat<T> Result = Input.GetNormalized();
		FJsonObjectRef Value = MakeShared<FJsonObject>();
		Value->SetStringField(TEXT("case_id"), CaseId);
		Value->SetArrayField(TEXT("input_xyzw"), QuaternionArray(Input));
		Value->SetNumberField(TEXT("input_norm_squared"), static_cast<double>(Input.SizeSquared()));
		Value->SetArrayField(TEXT("normalized_xyzw"), QuaternionArray(Result));
		Value->SetNumberField(TEXT("normalized_norm"), static_cast<double>(Result.Size()));
		return Value;
	}

	template<typename T>
	TArray<FJsonValueRef> CaptureNormalizationCases()
	{
		using TQuat = UE::Math::TQuat<T>;
		TArray<FJsonValueRef> Result;
		Result.Add(MakeShared<FJsonValueObject>(CaptureNormalizationCase<T>(TEXT("identity"), TQuat::Identity)));
		Result.Add(MakeShared<FJsonValueObject>(CaptureNormalizationCase<T>(TEXT("zero"), TQuat(T(0), T(0), T(0), T(0)))));
		Result.Add(MakeShared<FJsonValueObject>(CaptureNormalizationCase<T>(TEXT("below_default_squared_tolerance"), TQuat(T(5.e-5), T(0), T(0), T(0)))));
		Result.Add(MakeShared<FJsonValueObject>(CaptureNormalizationCase<T>(TEXT("above_default_squared_tolerance"), TQuat(T(2.e-4), T(0), T(0), T(0)))));
		Result.Add(MakeShared<FJsonValueObject>(CaptureNormalizationCase<T>(TEXT("nonunit"), TQuat(T(0.2), T(-0.3), T(0.4), T(0.5)))));
		return Result;
	}

	FJsonObjectRef CaptureExtendedConversions()
	{
		FJsonObjectRef Result = MakeShared<FJsonObject>();
		Result->SetStringField(TEXT("quaternion_algebra"), TEXT("Hamilton"));
		Result->SetStringField(TEXT("rotation_action"), TEXT("active; quaternion coefficient order xyzw"));
		Result->SetNumberField(TEXT("normalization_squared_tolerance"), UE_SMALL_NUMBER);
		Result->SetArrayField(TEXT("matrix_to_quaternion_float32"), CaptureMatrixToQuaternion<float>());
		Result->SetArrayField(TEXT("matrix_to_quaternion_float64"), CaptureMatrixToQuaternion<double>());
		Result->SetArrayField(TEXT("axis_angle_float32"), CaptureAxisAngleCases<float>());
		Result->SetArrayField(TEXT("axis_angle_float64"), CaptureAxisAngleCases<double>());
		Result->SetArrayField(TEXT("rotation_vector_float32"), CaptureRotationVectorCases<float>());
		Result->SetArrayField(TEXT("rotation_vector_float64"), CaptureRotationVectorCases<double>());
		Result->SetArrayField(TEXT("slerp_float32"), CaptureSlerpCases<float>());
		Result->SetArrayField(TEXT("slerp_float64"), CaptureSlerpCases<double>());
		Result->SetArrayField(TEXT("normalization_float32"), CaptureNormalizationCases<float>());
		Result->SetArrayField(TEXT("normalization_float64"), CaptureNormalizationCases<double>());
		return Result;
	}

	TSharedPtr<FJsonObject> CaptureFixture()
	{
		TMap<FString, FQuat4f> Quats32;
		TMap<FString, FQuat4d> Quats64;
		TArray<FJsonValueRef> CaseValues;
		for (const FRotatorCase& Input : Cases)
		{
			FJsonObjectRef CaseJson = MakeShared<FJsonObject>();
			CaseJson->SetStringField(TEXT("case_id"), Input.CaseId);
			CaseJson->SetObjectField(TEXT("input_rotator_degrees"), InputRotatorJson(Input));
			CaseJson->SetObjectField(TEXT("float32"), CapturePrecision<float>(
				Input, TEXT("FRotator3f"), TEXT("FQuat4f"), Quats32));
			CaseJson->SetObjectField(TEXT("float64"), CapturePrecision<double>(
				Input, TEXT("FRotator3d (FRotator)"), TEXT("FQuat4d (FQuat)"), Quats64));
			CaseValues.Add(MakeShared<FJsonValueObject>(CaseJson));
		}

		const FEngineVersion& Version = FEngineVersion::Current();
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

		FJsonObjectRef ProofProfile = MakeShared<FJsonObject>();
		ProofProfile->SetStringField(TEXT("euler_to_quaternion"), TEXT("native FRotator::Quaternion, TRotator::MakeFromEuler, and TQuat::MakeFromEuler"));
		ProofProfile->SetStringField(TEXT("euler_readback_and_threshold_neighbors"), TEXT("native FQuat::Rotator and FQuat::Euler with four samples around the singularity threshold"));
		ProofProfile->SetStringField(TEXT("matrix_to_quaternion"), TEXT("native TQuat(TMatrix)"));
		ProofProfile->SetStringField(TEXT("axis_angle"), TEXT("native FQuat axis-angle constructor and GetAxis/GetAngle"));
		ProofProfile->SetStringField(TEXT("rotation_vector"), TEXT("native FQuat rotation-vector constructor and ToRotationVector"));
		ProofProfile->SetStringField(TEXT("slerp_shortest"), TEXT("native FQuat::Slerp"));
		ProofProfile->SetStringField(TEXT("slerp_full_path"), TEXT("native FQuat::SlerpFullPath"));
		ProofProfile->SetStringField(TEXT("normalization_edges"), TEXT("native FQuat::GetNormalized with zero and squared-norm threshold cases"));
		ProofProfile->SetStringField(TEXT("basis_vector_action"), TEXT("native FQuat::RotateVector and TQuat::ToMatrix"));
		ProofProfile->SetStringField(TEXT("composition_and_frame_map"), TEXT("native FQuat/FTransform composition and explicit source-frame probes"));

		FJsonObjectRef Root = MakeShared<FJsonObject>();
		Root->SetNumberField(TEXT("schema_version"), 1);
		Root->SetStringField(TEXT("evidence_kind"), TEXT("native_unreal_runtime_capture"));
		Root->SetStringField(TEXT("capture_method"), TEXT("C++ Unreal Editor Automation Test"));
		Root->SetStringField(TEXT("quaternion_order"), TEXT("xyzw"));
		Root->SetStringField(TEXT("quaternion_algebra"), TEXT("Hamilton"));
		Root->SetStringField(TEXT("rotation_action"), TEXT("active"));
		Root->SetStringField(TEXT("input_field_order"), TEXT("named Pitch, Yaw, Roll fields; angles in degrees"));
		Root->SetStringField(TEXT("matrix_layout"), TEXT("3x3 rows copied from TQuat::ToMatrix; each row equals the RotateVector image of +X, +Y, or +Z"));
		Root->SetStringField(TEXT("quaternion_multiplication_order"), TEXT("lhs * rhs; Unreal applies rhs first, then lhs"));
		Root->SetObjectField(TEXT("proof_profile"), ProofProfile);
		Root->SetObjectField(TEXT("engine"), Engine);
		Root->SetArrayField(TEXT("cases"), CaseValues);
		Root->SetArrayField(TEXT("multiplication_cases_float32"), CaptureProducts(Quats32));
		Root->SetArrayField(TEXT("multiplication_cases_float64"), CaptureProducts(Quats64));
		Root->SetObjectField(TEXT("native_semantics"), CaptureNativeSemantics(Quats32, Quats64));
		FJsonObjectRef CanonicalInterop = MakeShared<FJsonObject>();
		FJsonObjectRef PrecisionCaptures = MakeShared<FJsonObject>();
		PrecisionCaptures->SetObjectField(TEXT("float32"), CaptureCanonicalInterop<float>());
		PrecisionCaptures->SetObjectField(TEXT("float64"), CaptureCanonicalInterop<double>());
		CanonicalInterop->SetStringField(TEXT("contract_id"), TEXT("quatops-canonical-hamilton-wxyz-active-rh-v1"));
		CanonicalInterop->SetStringField(TEXT("native_quaternion_order"), TEXT("xyzw"));
		CanonicalInterop->SetArrayField(TEXT("frame_map_destination_axes_from_source"), {
			MakeShared<FJsonValueString>(TEXT("y")),
			MakeShared<FJsonValueString>(TEXT("x")),
			MakeShared<FJsonValueString>(TEXT("z")),
		});
		CanonicalInterop->SetObjectField(TEXT("precision_captures"), PrecisionCaptures);
		Root->SetObjectField(TEXT("canonical_interop"), CanonicalInterop);
		Root->SetObjectField(TEXT("extended_conversions"), CaptureExtendedConversions());
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
	const FJsonObjectRef Semantics = Fixture->GetObjectField(TEXT("native_semantics"));
	for (const TCHAR* Precision : { TEXT("float32"), TEXT("float64") })
	{
		const FString Suffix = Precision;
		const FJsonObjectRef QuaternionComposition = Semantics->GetObjectField(
			FString::Printf(TEXT("quaternion_composition_%s"), *Suffix));
		const FJsonObjectRef TransformComposition = Semantics->GetObjectField(
			FString::Printf(TEXT("transform_composition_%s"), *Suffix));
		TestTrue(FString::Printf(TEXT("FQuat %s lhs*rhs applies rhs then lhs"), Precision),
			QuaternionComposition->GetBoolField(TEXT("product_matches_rhs_then_lhs")));
		TestFalse(FString::Printf(TEXT("FQuat %s lhs*rhs applies lhs then rhs"), Precision),
			QuaternionComposition->GetBoolField(TEXT("product_matches_lhs_then_rhs")));
		TestTrue(FString::Printf(TEXT("FTransform %s lhs*rhs applies lhs then rhs"), Precision),
			TransformComposition->GetBoolField(TEXT("product_matches_lhs_then_rhs")));
		TestFalse(FString::Printf(TEXT("FTransform %s lhs*rhs applies rhs then lhs"), Precision),
			TransformComposition->GetBoolField(TEXT("product_matches_rhs_then_lhs")));
		for (const TSharedPtr<FJsonValue>& CaseValue : Fixture->GetArrayField(TEXT("cases")))
		{
			const FJsonObjectRef Case = CaseValue->AsObject();
			const FJsonObjectRef Native = Case->GetObjectField(Suffix);
			TestTrue(FString::Printf(TEXT("TRotator::MakeFromEuler %s matches named fields for %s"),
				Precision, *Case->GetStringField(TEXT("case_id"))),
				Native->GetBoolField(TEXT("rotator_make_from_euler_matches_named_fields")));
			TestTrue(FString::Printf(TEXT("TQuat::MakeFromEuler %s matches FRotator::Quaternion for %s"),
				Precision, *Case->GetStringField(TEXT("case_id"))),
				Native->GetBoolField(TEXT("quat_make_from_euler_matches_named_fields")));
		}
		const TArray<TSharedPtr<FJsonValue>>& CrossProducts = Semantics->GetArrayField(
			FString::Printf(TEXT("cross_products_%s"), *Suffix));
		for (const TSharedPtr<FJsonValue>& CrossValue : CrossProducts)
		{
			const FJsonObjectRef Cross = CrossValue->AsObject();
			TestTrue(FString::Printf(TEXT("Unreal %s cross product %s matches the numeric basis identity"),
				Precision, *Cross->GetStringField(TEXT("case_id"))),
				Cross->GetBoolField(TEXT("matches_numeric_basis_identity")));
			TestTrue(FString::Printf(TEXT("Unreal %s cross product %s reverses sign"),
				Precision, *Cross->GetStringField(TEXT("case_id"))),
				Cross->GetBoolField(TEXT("reversed_is_negative")));
		}
	}
	const FJsonObjectRef AxisMap = Semantics->GetObjectField(TEXT("source_to_unreal_axis_map"));
	for (const TCHAR* Precision : { TEXT("float32"), TEXT("float64") })
	{
		for (const TSharedPtr<FJsonValue>& ProbeValue : AxisMap->GetArrayField(
			FString::Printf(TEXT("mapped_source_principal_rotations_%s"), Precision)))
		{
			const FJsonObjectRef Probe = ProbeValue->AsObject();
			TestTrue(FString::Printf(TEXT("Source-to-Unreal map preserves %s rotation action at %s precision"),
				Precision, *Probe->GetStringField(TEXT("source_rotation_axis"))),
				Probe->GetBoolField(TEXT("preserves_mapped_rotation_action")));
		}
	}

	const FJsonObjectRef Extended = Fixture->GetObjectField(TEXT("extended_conversions"));
	for (const TCHAR* Precision : { TEXT("float32"), TEXT("float64") })
	{
		for (const TSharedPtr<FJsonValue>& MatrixValue : Extended->GetArrayField(
			FString::Printf(TEXT("matrix_to_quaternion_%s"), Precision)))
		{
			const FJsonObjectRef MatrixCase = MatrixValue->AsObject();
			if (!MatrixCase->GetBoolField(TEXT("diagnostic_only")))
			{
				TestTrue(FString::Printf(TEXT("TQuat(matrix) %s %s preserves the rotation"),
					Precision, *MatrixCase->GetStringField(TEXT("case_id"))),
					MatrixCase->GetBoolField(TEXT("same_rotation")));
			}
		}
		for (const TCHAR* Family : { TEXT("axis_angle"), TEXT("rotation_vector") })
		{
			for (const TSharedPtr<FJsonValue>& ConversionValue : Extended->GetArrayField(
				FString::Printf(TEXT("%s_%s"), Family, Precision)))
			{
				const FJsonObjectRef Conversion = ConversionValue->AsObject();
				TestTrue(FString::Printf(TEXT("Unreal %s %s constructor returns a unit quaternion"),
					Precision, *Conversion->GetStringField(TEXT("case_id"))),
					Conversion->GetBoolField(TEXT("constructor_norm_is_unit")));
			}
		}
		for (const TSharedPtr<FJsonValue>& SlerpValue : Extended->GetArrayField(
			FString::Printf(TEXT("slerp_%s"), Precision)))
		{
			const FJsonObjectRef SlerpCase = SlerpValue->AsObject();
			TestTrue(FString::Printf(TEXT("Unreal Slerp %s %s at %g returns unit output"),
				Precision, *SlerpCase->GetStringField(TEXT("case_id")), SlerpCase->GetNumberField(TEXT("alpha"))),
				SlerpCase->GetBoolField(TEXT("shortest_result_is_unit")));
			TestTrue(FString::Printf(TEXT("Unreal SlerpFullPath %s %s at %g returns unit output"),
				Precision, *SlerpCase->GetStringField(TEXT("case_id")), SlerpCase->GetNumberField(TEXT("alpha"))),
				SlerpCase->GetBoolField(TEXT("full_path_result_is_unit")));
		}
		const TArray<TSharedPtr<FJsonValue>>& Normalization = Extended->GetArrayField(
			FString::Printf(TEXT("normalization_%s"), Precision));
		for (const TSharedPtr<FJsonValue>& NormalizationValue : Normalization)
		{
			TestTrue(FString::Printf(TEXT("Unreal normalization %s produces a unit quaternion"), Precision),
				FMath::Abs(NormalizationValue->AsObject()->GetNumberField(TEXT("normalized_norm")) - 1.0) <= 1.e-5);
		}
		const FJsonObjectRef Zero = Normalization[1]->AsObject();
		const FJsonObjectRef Below = Normalization[2]->AsObject();
		const FJsonObjectRef Above = Normalization[3]->AsObject();
		TestTrue(FString::Printf(TEXT("Unreal normalization %s maps zero to identity"), Precision),
			Zero->GetArrayField(TEXT("normalized_xyzw"))[3]->AsNumber() == 1.0);
		TestTrue(FString::Printf(TEXT("Unreal normalization %s maps below-threshold input to identity"), Precision),
			Below->GetArrayField(TEXT("normalized_xyzw"))[3]->AsNumber() == 1.0);
		TestTrue(FString::Printf(TEXT("Unreal normalization %s normalizes above-threshold input"), Precision),
			Above->GetArrayField(TEXT("normalized_xyzw"))[0]->AsNumber() > 0.99);
	}

	FString OutputPath;
	if (!FParse::Value(FCommandLine::Get(), TEXT("QuatOpsCaptureOutput="), OutputPath))
	{
		OutputPath = FPaths::Combine(FPaths::ProjectSavedDir(), TEXT("quatops_unreal_fixtures.json"));
	}
	FPaths::NormalizeFilename(OutputPath);
	IFileManager::Get().MakeDirectory(*FPaths::GetPath(OutputPath), true);

	FString Json;
	const TSharedRef<TJsonWriter< TCHAR, TPrettyJsonPrintPolicy<TCHAR> >> Writer =
		TJsonWriterFactory<TCHAR, TPrettyJsonPrintPolicy<TCHAR>>::Create(&Json);
	if (!FJsonSerializer::Serialize(Fixture.ToSharedRef(), Writer))
	{
		AddError(TEXT("Unreal fixture JSON serialization failed."));
		return false;
	}
	Json.AppendChar(TEXT('\n'));
	if (!FFileHelper::SaveStringToFile(Json, *OutputPath, FFileHelper::EEncodingOptions::ForceUTF8WithoutBOM))
	{
		AddError(FString::Printf(TEXT("Could not write Unreal fixture capture to %s."), *OutputPath));
		return false;
	}
	AddInfo(FString::Printf(TEXT("Captured native Unreal fixtures to %s."), *OutputPath));
	return true;
}
