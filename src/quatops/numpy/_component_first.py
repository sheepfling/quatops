"""Adapters for NumPy APIs whose quaternion component axis is leading."""

from functools import wraps
from inspect import Signature, signature
from types import ModuleType
from typing import Any, Callable

import numpy as np

_OPERATIONS = (
    "acosh", "acos", "add", "align_sign", "align_vectors", "asin", "asinh", "atan",
    "atanh", "boxminus", "boxplus", "canonicalize", "commutator", "conjugate",
    "cos", "cosh", "divide", "dot", "exp", "expm1", "from_axis_angle", "from_euler",
    "from_hamilton", "from_homogeneous_matrix", "from_mrp", "from_rotation_matrix",
    "from_rotation_vector", "from_rotvec", "from_two_vectors", "from_wxyz", "from_xyzw",
    "integrate_angular_velocity", "integrate_angular_velocity_samples", "inverse",
    "left_divide", "log", "log1p", "log10", "log2", "make_sign_continuous",
    "multiply", "negate", "nlerp", "norm", "norm_squared", "normalize", "power",
    "power_quaternion", "pow_real", "quaternion_derivative", "random_quaternion",
    "random_rotation", "rotate", "rotation_covariance", "rotation_distance",
    "rotation_mean", "rotations_equivalent", "right_divide", "scalar_part", "scale", "sin",
    "sinh", "slerp", "so3_adjoint", "squad", "skew_matrix", "sqrt", "swing_twist",
    "so3_left_jacobian",
    "so3_left_jacobian_inverse", "so3_right_jacobian", "so3_right_jacobian_inverse",
    "subtract", "tan", "tanh", "to_axis_angle", "to_euler", "to_hamilton", "to_left_matrix",
    "to_mrp", "to_right_matrix", "to_rotation_matrix", "to_rotation_vector", "to_rotvec",
    "to_homogeneous_matrix", "to_wxyz", "to_xyzw", "vector_part",
)

_COMPONENT_ARGUMENTS = {
    "acosh": {"q"},
    "acos": {"q"},
    "add": {"lhs", "rhs"},
    "align_sign": {"q", "reference"},
    "align_vectors": {"source", "target"},
    "asin": {"q"},
    "asinh": {"q"},
    "atan": {"q"},
    "atanh": {"q"},
    "boxminus": {"q", "reference"},
    "boxplus": {"q", "delta"},
    "conjugate": {"q"},
    "commutator": {"lhs", "rhs"},
    "cos": {"q"},
    "cosh": {"q"},
    "divide": {"lhs", "rhs"},
    "dot": {"lhs", "rhs"},
    "exp": {"q"},
    "expm1": {"q"},
    "from_axis_angle": {"a"},
    "from_euler": {"angles"},
    "from_hamilton": {"q"},
    "from_rotation_matrix": {"matrix"},
    "from_homogeneous_matrix": {"matrix"},
    "from_mrp": {"sigma"},
    "from_rotation_vector": {"r"},
    "from_rotvec": {"r"},
    "from_two_vectors": {"source", "target"},
    "from_wxyz": {"q"},
    "from_xyzw": {"q"},
    "integrate_angular_velocity": {"q", "omega"},
    "integrate_angular_velocity_samples": {"q", "omegas"},
    "inverse": {"q"},
    "left_divide": {"lhs", "rhs"},
    "log": {"q"},
    "log1p": {"q"},
    "log10": {"q"},
    "log2": {"q"},
    "make_sign_continuous": {"quaternions"},
    "multiply": {"lhs", "rhs"},
    "nlerp": {"q0", "q1"},
    "negate": {"q"},
    "norm": {"q"},
    "norm_squared": {"q"},
    "normalize": {"q"},
    "power": {"q"},
    "power_quaternion": {"q", "exponent"},
    "pow_real": {"q"},
    "quaternion_derivative": {"q", "omega"},
    "rotation_mean": {"quaternions"},
    "rotation_covariance": {"quaternions", "mean"},
    "random_quaternion": set(),
    "random_rotation": set(),
    "rotate": {"q", "v"},
    "rotation_distance": {"q0", "q1"},
    "rotations_equivalent": {"q0", "q1"},
    "right_divide": {"lhs", "rhs"},
    "scale": {"q"},
    "scalar_part": {"q"},
    "sin": {"q"},
    "sinh": {"q"},
    "tan": {"q"},
    "slerp": {"q0", "q1"},
    "so3_adjoint": {"q"},
    "squad": {"q0", "q1", "a", "b"},
    "swing_twist": {"q", "axis"},
    "skew_matrix": {"v"},
    "sqrt": {"q"},
    "so3_left_jacobian": {"r"},
    "so3_left_jacobian_inverse": {"r"},
    "so3_right_jacobian": {"r"},
    "so3_right_jacobian_inverse": {"r"},
    "subtract": {"lhs", "rhs"},
    "tanh": {"q"},
    "to_axis_angle": {"q"},
    "to_euler": {"q"},
    "to_homogeneous_matrix": {"q", "translation"},
    "canonicalize": {"q"},
    "to_hamilton": {"q"},
    "to_left_matrix": {"q"},
    "to_right_matrix": {"q"},
    "to_rotation_matrix": {"q"},
    "to_mrp": {"q"},
    "to_rotation_vector": {"q"},
    "to_rotvec": {"q"},
    "to_wxyz": {"q"},
    "to_xyzw": {"q"},
    "vector_part": {"q"},
}

_QUATERNION_RESULTS = {
    "acos", "acosh", "add", "align_sign", "align_vectors", "asin", "asinh", "atan",
    "atanh", "boxplus", "canonicalize", "commutator", "conjugate", "cos", "cosh",
    "divide", "exp", "expm1", "from_axis_angle", "from_euler", "from_hamilton",
    "from_homogeneous_matrix", "from_mrp", "from_rotation_matrix", "from_rotation_vector",
    "from_rotvec", "from_two_vectors", "from_wxyz", "from_xyzw",
    "integrate_angular_velocity", "integrate_angular_velocity_samples", "inverse",
    "left_divide", "log", "log1p", "log10", "log2", "make_sign_continuous",
    "multiply", "negate", "nlerp", "normalize", "power", "power_quaternion", "pow_real",
    "quaternion_derivative", "random_quaternion", "random_rotation", "right_divide",
    "rotation_mean", "scale", "sin", "sinh", "slerp", "squad", "sqrt", "subtract",
    "tan", "tanh", "to_axis_angle",
    "to_hamilton", "to_wxyz", "to_xyzw",
}
_VECTOR_RESULTS = {"boxminus", "rotate", "to_euler", "to_mrp", "to_rotvec", "to_rotation_vector", "vector_part"}
_MATRIX_RESULTS = {
    "rotation_covariance", "skew_matrix", "so3_adjoint", "so3_left_jacobian", "so3_left_jacobian_inverse",
    "so3_right_jacobian", "so3_right_jacobian_inverse", "to_left_matrix",
    "to_homogeneous_matrix", "to_right_matrix", "to_rotation_matrix",
}


def _move_input_to_last(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have its component axis first") from exc
    if array.ndim == 0:
        raise ValueError(f"{name} must have its component axis first")
    return np.moveaxis(array, 0, -1)


def _move_matrix_to_last(value: Any, name: str) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have its matrix axes first") from exc
    if array.ndim < 2:
        raise ValueError(f"{name} must have its matrix axes first")
    return np.moveaxis(array, (0, 1), (-2, -1))


def _move_output_to_first(operation: str, result: Any) -> Any:
    if operation == "swing_twist":
        return tuple(np.moveaxis(item, -1, 0) for item in result)
    if operation in _MATRIX_RESULTS:
        return np.moveaxis(result, (-2, -1), (0, 1))
    if operation in _VECTOR_RESULTS:
        return np.moveaxis(result, -1, 0)
    if operation in _QUATERNION_RESULTS:
        return np.moveaxis(result, -1, 0)
    return result


def bind_component_first(backend: ModuleType, module_name: str) -> dict[str, Any]:
    """Wrap one trailing-component NumPy module with leading-axis adapters."""
    exports: dict[str, Any] = {}
    for name in _OPERATIONS:
        if not hasattr(backend, name):
            continue
        implementation = getattr(backend, name)
        operation_signature = signature(implementation)
        component_arguments = _COMPONENT_ARGUMENTS.get(name, set())

        def make_adapter(
            fn: Callable[..., Any],
            fn_signature: Signature,
            operation: str,
            inputs: set[str],
        ) -> Callable[..., Any]:
            operation_doc = fn.__doc__ or ""
            for trailing_shape, leading_shape in (
                ("(..., 4, 4)", "(4, 4, ...)"),
                ("(..., 3, 3)", "(3, 3, ...)"),
                ("(..., 4)", "(4, ...)"),
                ("(..., 3)", "(3, ...)"),
            ):
                operation_doc = operation_doc.replace(trailing_shape, leading_shape)

            @wraps(fn)
            def adapted(*args: Any, **kwargs: Any) -> Any:
                bound = fn_signature.bind(*args, **kwargs)
                for parameter in inputs:
                    if parameter in bound.arguments:
                        mover = (
                            _move_matrix_to_last
                            if operation in ("from_rotation_matrix", "from_homogeneous_matrix")
                            else _move_input_to_last
                        )
                        if bound.arguments[parameter] is not None:
                            bound.arguments[parameter] = mover(bound.arguments[parameter], parameter)
                result = fn(*bound.args, **bound.kwargs)
                return _move_output_to_first(operation, result)

            adapted.__module__ = module_name
            adapted.__doc__ = (
                f"Run `{operation}` with the component axis first. Quaternion arrays "
                "have shape `(4, ...)`, vector arrays have shape `(3, ...)`, and "
                "matrix outputs put both matrix axes first. Scalar reductions return "
                "their batch shape. Validation and numerical behavior match the "
                "corresponding trailing-axis NumPy operation."
                + (f"\n\n{operation_doc}" if operation_doc else "")
            )
            return adapted

        exports[name] = make_adapter(
            implementation, operation_signature, name, component_arguments
        )
    return exports
