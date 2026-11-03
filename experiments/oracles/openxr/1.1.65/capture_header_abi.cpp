// OpenXR SDK header ABI probe only. It does not create an instance or runtime.
#include <openxr/openxr.h>

#include <cstddef>
#include <iostream>

int main() {
  static_assert(offsetof(XrQuaternionf, x) == 0);
  static_assert(offsetof(XrQuaternionf, y) == sizeof(float));
  static_assert(offsetof(XrQuaternionf, z) == 2 * sizeof(float));
  static_assert(offsetof(XrQuaternionf, w) == 3 * sizeof(float));
  static_assert(offsetof(XrPosef, orientation) == 0);
  static_assert(offsetof(XrPosef, position) == sizeof(XrQuaternionf));

  const XrQuaternionf orientation{0.125F, -0.25F, 0.5F, 0.75F};
  const XrPosef pose{orientation, XrVector3f{1.25F, -2.5F, 4.0F}};

  std::cout << "{\"evidence_kind\":\"native_sdk_header_abi_not_runtime\","
            << "\"sdk_api_version\":["
            << XR_VERSION_MAJOR(XR_CURRENT_API_VERSION) << ','
            << XR_VERSION_MINOR(XR_CURRENT_API_VERSION) << ','
            << XR_VERSION_PATCH(XR_CURRENT_API_VERSION) << "],"
            << "\"sizeof_float\":" << sizeof(float) << ','
            << "\"sizeof_XrQuaternionf\":" << sizeof(XrQuaternionf) << ','
            << "\"XrQuaternionf_offsets_bytes\":{\"x\":"
            << offsetof(XrQuaternionf, x) << ",\"y\":"
            << offsetof(XrQuaternionf, y) << ",\"z\":"
            << offsetof(XrQuaternionf, z) << ",\"w\":"
            << offsetof(XrQuaternionf, w) << "},"
            << "\"sizeof_XrPosef\":" << sizeof(XrPosef) << ','
            << "\"XrPosef_offsets_bytes\":{\"orientation\":"
            << offsetof(XrPosef, orientation) << ",\"position\":"
            << offsetof(XrPosef, position) << "},"
            << "\"assigned_quaternion_xyzw\":[" << orientation.x << ','
            << orientation.y << ',' << orientation.z << ',' << orientation.w
            << "],\"assigned_pose_position_xyz_m\":[" << pose.position.x
            << ',' << pose.position.y << ',' << pose.position.z << "],"
            << "\"runtime_functions_called\":false}\n";
  return 0;
}
