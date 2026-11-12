// OpenXR SDK header ABI probe only. It does not create an instance or runtime.
#include <openxr/openxr.h>

#include <cstddef>
#include <cstdio>
#include <iostream>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

int main() {
#ifdef _WIN32
  _setmode(_fileno(stdout), _O_BINARY);
#endif

  static_assert(offsetof(XrQuaternionf, x) == 0);
  static_assert(offsetof(XrQuaternionf, y) == sizeof(float));
  static_assert(offsetof(XrQuaternionf, z) == 2 * sizeof(float));
  static_assert(offsetof(XrQuaternionf, w) == 3 * sizeof(float));
  static_assert(offsetof(XrPosef, orientation) == 0);
  static_assert(offsetof(XrPosef, position) == sizeof(XrQuaternionf));

  const XrQuaternionf orientation{0.125F, -0.25F, 0.5F, 0.75F};
  const XrPosef pose{orientation, XrVector3f{1.25F, -2.5F, 4.0F}};

  std::cout << "{\n"
            << "  \"evidence_kind\": \"native_sdk_header_abi_not_runtime\",\n"
            << "  \"sdk_api_version\": ["
            << XR_VERSION_MAJOR(XR_CURRENT_API_VERSION) << ", "
            << XR_VERSION_MINOR(XR_CURRENT_API_VERSION) << ", "
            << XR_VERSION_PATCH(XR_CURRENT_API_VERSION) << "],\n"
            << "  \"sizeof_float\": " << sizeof(float) << ",\n"
            << "  \"sizeof_XrQuaternionf\": " << sizeof(XrQuaternionf)
            << ",\n"
            << "  \"XrQuaternionf_offsets_bytes\": {\n"
            << "    \"x\": " << offsetof(XrQuaternionf, x) << ",\n"
            << "    \"y\": " << offsetof(XrQuaternionf, y) << ",\n"
            << "    \"z\": " << offsetof(XrQuaternionf, z) << ",\n"
            << "    \"w\": " << offsetof(XrQuaternionf, w) << "\n"
            << "  },\n"
            << "  \"sizeof_XrPosef\": " << sizeof(XrPosef) << ",\n"
            << "  \"XrPosef_offsets_bytes\": {\n"
            << "    \"orientation\": "
            << offsetof(XrPosef, orientation) << ",\n"
            << "    \"position\": " << offsetof(XrPosef, position)
            << "\n"
            << "  },\n"
            << "  \"assigned_quaternion_xyzw\": [" << orientation.x << ", "
            << orientation.y << ", " << orientation.z << ", "
            << orientation.w << "],\n"
            << "  \"assigned_pose_position_xyz_m\": [" << pose.position.x
            << ", " << pose.position.y << ", " << pose.position.z << "],\n"
            << "  \"runtime_functions_called\": false\n"
            << "}\n";
  return 0;
}
