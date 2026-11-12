// Capture OpenXR space relations from the pinned Khronos test runtime.
// This is a software test runtime, not a headset or vendor runtime.
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <openxr/openxr.h>
#include <openxr/openxr_loader_negotiation.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>

namespace {

void check(XrResult result, const char* operation) {
  if (XR_FAILED(result)) {
    std::cerr << operation << " failed with XrResult " << result << '\n';
    std::exit(EXIT_FAILURE);
  }
}

template <typename Function>
Function load_function(PFN_xrGetInstanceProcAddr get_proc_addr,
                       XrInstance instance, const char* name) {
  PFN_xrVoidFunction function = nullptr;
  check(get_proc_addr(instance, name, &function), name);
  if (function == nullptr) {
    std::cerr << name << " resolved to null\n";
    std::exit(EXIT_FAILURE);
  }
  return reinterpret_cast<Function>(function);
}

struct NamedPose {
  const char* name;
  XrPosef pose;
};

void print_vector(const XrVector3f& vector) {
  std::cout << '[' << vector.x << ", " << vector.y << ", " << vector.z
            << ']';
}

void print_pose(const XrPosef& pose) {
  std::cout << "{\"orientation_xyzw\":[" << pose.orientation.x << ", "
            << pose.orientation.y << ", " << pose.orientation.z << ", "
            << pose.orientation.w << "],\"position_xyz_m\":";
  print_vector(pose.position);
  std::cout << '}';
}

XrSpace create_local_space(PFN_xrCreateReferenceSpace create_reference_space,
                           XrSession session, const XrPosef& pose,
                           XrReferenceSpaceType reference_type) {
  XrReferenceSpaceCreateInfo create_info{
      XR_TYPE_REFERENCE_SPACE_CREATE_INFO};
  create_info.referenceSpaceType = reference_type;
  create_info.poseInReferenceSpace = pose;
  XrSpace space = XR_NULL_HANDLE;
  check(create_reference_space(session, &create_info, &space),
        "xrCreateReferenceSpace");
  return space;
}

XrSpaceLocation locate(PFN_xrLocateSpace locate_space, XrSpace space,
                       XrSpace base_space) {
  XrSpaceLocation location{XR_TYPE_SPACE_LOCATION};
  check(locate_space(space, base_space, 1, &location), "xrLocateSpace");
  if ((location.locationFlags & XR_SPACE_LOCATION_ORIENTATION_VALID_BIT) ==
          0 ||
      (location.locationFlags & XR_SPACE_LOCATION_POSITION_VALID_BIT) == 0) {
    std::cerr << "test runtime returned an invalid space location\n";
    std::exit(EXIT_FAILURE);
  }
  return location;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2 && argc != 3) {
    std::cerr << "usage: capture_openxr_test_runtime.exe <test_runtime.dll> [LOCAL|VIEW|LOCAL_FLOOR]\n";
    return EXIT_FAILURE;
  }

  const std::string reference_name = argc == 3 ? argv[2] : "LOCAL";
  XrReferenceSpaceType reference_type;
  if (reference_name == "LOCAL") reference_type = XR_REFERENCE_SPACE_TYPE_LOCAL;
  else if (reference_name == "VIEW") reference_type = XR_REFERENCE_SPACE_TYPE_VIEW;
  else if (reference_name == "LOCAL_FLOOR") reference_type = XR_REFERENCE_SPACE_TYPE_LOCAL_FLOOR;
  else { std::cerr << "unsupported reference space\n"; return EXIT_FAILURE; }

  HMODULE runtime_module = LoadLibraryA(argv[1]);
  if (runtime_module == nullptr) {
    std::cerr << "could not load the test runtime DLL\n";
    return EXIT_FAILURE;
  }
  auto negotiate = reinterpret_cast<PFN_xrNegotiateLoaderRuntimeInterface>(
      GetProcAddress(runtime_module, "xrNegotiateLoaderRuntimeInterface"));
  if (negotiate == nullptr) {
    std::cerr << "test runtime does not export the loader negotiation entry\n";
    return EXIT_FAILURE;
  }

  XrNegotiateLoaderInfo loader_info{};
  loader_info.structType = XR_LOADER_INTERFACE_STRUCT_LOADER_INFO;
  loader_info.structVersion = XR_LOADER_INFO_STRUCT_VERSION;
  loader_info.structSize = sizeof(loader_info);
  loader_info.minInterfaceVersion = XR_CURRENT_LOADER_RUNTIME_VERSION;
  loader_info.maxInterfaceVersion = XR_CURRENT_LOADER_RUNTIME_VERSION;
  loader_info.minApiVersion = XR_MAKE_VERSION(1, 0, 0);
  loader_info.maxApiVersion = XR_CURRENT_API_VERSION;

  XrNegotiateRuntimeRequest runtime_request{};
  runtime_request.structType = XR_LOADER_INTERFACE_STRUCT_RUNTIME_REQUEST;
  runtime_request.structVersion = XR_RUNTIME_INFO_STRUCT_VERSION;
  runtime_request.structSize = sizeof(runtime_request);
  check(negotiate(&loader_info, &runtime_request),
        "xrNegotiateLoaderRuntimeInterface");

  const auto get_proc_addr = runtime_request.getInstanceProcAddr;
  const auto enumerate_extensions = load_function<
      PFN_xrEnumerateInstanceExtensionProperties>(
      get_proc_addr, XR_NULL_HANDLE, "xrEnumerateInstanceExtensionProperties");
  const auto create_instance = load_function<PFN_xrCreateInstance>(
      get_proc_addr, XR_NULL_HANDLE, "xrCreateInstance");

  uint32_t extension_count = 0;
  check(enumerate_extensions(nullptr, 0, &extension_count, nullptr),
        "xrEnumerateInstanceExtensionProperties(count)");
  std::array<XrExtensionProperties, 16> extensions{};
  if (extension_count > extensions.size()) {
    std::cerr << "unexpected extension count from test runtime\n";
    return EXIT_FAILURE;
  }
  for (auto& extension : extensions) {
    extension.type = XR_TYPE_EXTENSION_PROPERTIES;
  }
  check(enumerate_extensions(nullptr, extension_count, &extension_count,
                             extensions.data()),
        "xrEnumerateInstanceExtensionProperties(values)");
  bool headless_extension_advertised = false;
  for (uint32_t index = 0; index < extension_count; ++index) {
    if (std::string(extensions[index].extensionName) ==
        XR_MND_HEADLESS_EXTENSION_NAME) {
      headless_extension_advertised = true;
    }
  }
  if (!headless_extension_advertised) {
    std::cerr << "test runtime does not advertise XR_MND_headless\n";
    return EXIT_FAILURE;
  }

  const char* enabled_extensions[] = {XR_MND_HEADLESS_EXTENSION_NAME};
  XrInstanceCreateInfo instance_info{XR_TYPE_INSTANCE_CREATE_INFO};
  instance_info.applicationInfo.apiVersion = XR_MAKE_VERSION(1, 1, 0);
  std::string app_name = "QuatOps OpenXR test-runtime capture";
  std::copy(app_name.begin(), app_name.end(),
            instance_info.applicationInfo.applicationName);
  instance_info.applicationInfo.applicationVersion = 1;
  std::string engine_name = "QuatOps evidence capture";
  std::copy(engine_name.begin(), engine_name.end(),
            instance_info.applicationInfo.engineName);
  instance_info.applicationInfo.engineVersion = 1;
  instance_info.enabledExtensionCount = 1;
  instance_info.enabledExtensionNames = enabled_extensions;

  XrInstance instance = XR_NULL_HANDLE;
  check(create_instance(&instance_info, &instance), "xrCreateInstance");

  const auto get_system = load_function<PFN_xrGetSystem>(
      get_proc_addr, instance, "xrGetSystem");
  const auto create_session = load_function<PFN_xrCreateSession>(
      get_proc_addr, instance, "xrCreateSession");
  const auto create_reference_space =
      load_function<PFN_xrCreateReferenceSpace>(
          get_proc_addr, instance, "xrCreateReferenceSpace");
  const auto locate_space = load_function<PFN_xrLocateSpace>(
      get_proc_addr, instance, "xrLocateSpace");
  const auto destroy_space = load_function<PFN_xrDestroySpace>(
      get_proc_addr, instance, "xrDestroySpace");
  const auto destroy_session = load_function<PFN_xrDestroySession>(
      get_proc_addr, instance, "xrDestroySession");
  const auto destroy_instance = load_function<PFN_xrDestroyInstance>(
      get_proc_addr, instance, "xrDestroyInstance");

  XrSystemGetInfo system_info{XR_TYPE_SYSTEM_GET_INFO};
  system_info.formFactor = XR_FORM_FACTOR_HEAD_MOUNTED_DISPLAY;
  XrSystemId system_id = XR_NULL_SYSTEM_ID;
  check(get_system(instance, &system_info, &system_id), "xrGetSystem");

  XrSessionCreateInfo session_info{XR_TYPE_SESSION_CREATE_INFO};
  session_info.systemId = system_id;
  XrSession session = XR_NULL_HANDLE;
  check(create_session(instance, &session_info, &session), "xrCreateSession");

  constexpr float s45 = 0.7071067811865475244F;
  constexpr float s30_over_s2 = 0.3535533905932737622F;
  constexpr float c30 = 0.8660254037844386468F;
  constexpr float s22_5_over_s2 = 0.2705980500730984922F;
  constexpr float c22_5 = 0.9238795325112867561F;
  const std::array<NamedPose, 3> inputs{{
      {"A", {{0.5F, 0.5F, 0.0F, s45}, {1.25F, -2.0F, 0.5F}}},
      {"B", {{0.0F, s30_over_s2, s30_over_s2, c30},
              {-0.75F, 1.5F, 2.25F}}},
      {"C", {{-s22_5_over_s2, 0.0F, -s22_5_over_s2, c22_5},
              {3.0F, 0.25F, -1.0F}}},
  }};

  std::array<XrSpace, 3> spaces{};
  for (size_t index = 0; index < inputs.size(); ++index) {
    spaces[index] = create_local_space(create_reference_space, session,
                                       inputs[index].pose, reference_type);
  }

  // xrLocateSpace(target, base, ...) returns the target pose in base.
  const auto a_from_b = locate(locate_space, spaces[1], spaces[0]);
  const auto b_from_c = locate(locate_space, spaces[2], spaces[1]);
  const auto a_from_c = locate(locate_space, spaces[2], spaces[0]);

  std::cout << std::setprecision(9)
            << "{\n"
            << "  \"evidence_kind\": \"native_openxr_test_runtime_frame_chain\",\n"
            << "  \"runtime\": {\"product\": \"Khronos OpenXR-SDK-Source test_runtime\", \"headless_extension\": \""
            << XR_MND_HEADLESS_EXTENSION_NAME
            << "\", \"headless_extension_advertised\": true, \"hardware_runtime\": false},\n"
            << "  \"api_version\": [" << XR_VERSION_MAJOR(runtime_request.runtimeApiVersion)
            << ", " << XR_VERSION_MINOR(runtime_request.runtimeApiVersion) << ", "
            << XR_VERSION_PATCH(runtime_request.runtimeApiVersion) << "],\n"
            << "  \"locate_time_xr_time\": 1,\n"
            << "  \"reference_space_type\": \"" << reference_name << "\",\n"
            << "  \"inputs_pose_in_reference_space\": {\n";
  for (size_t index = 0; index < inputs.size(); ++index) {
    std::cout << "    \"" << inputs[index].name << "\":";
    print_pose(inputs[index].pose);
    std::cout << (index + 1 == inputs.size() ? "\n" : ",\n");
  }
  std::cout << "  },\n"
            << "  \"observations\": {\n"
            << "    \"A_from_B\": {\"target\": \"B\", \"base\": \"A\", \"location_flags\": "
            << a_from_b.locationFlags << ", \"pose\":";
  print_pose(a_from_b.pose);
  std::cout << "},\n"
            << "    \"B_from_C\": {\"target\": \"C\", \"base\": \"B\", \"location_flags\": "
            << b_from_c.locationFlags << ", \"pose\":";
  print_pose(b_from_c.pose);
  std::cout << "},\n"
            << "    \"A_from_C\": {\"target\": \"C\", \"base\": \"A\", \"location_flags\": "
            << a_from_c.locationFlags << ", \"pose\":";
  print_pose(a_from_c.pose);
  std::cout << "}\n  },\n"
            << "  \"quatops_executed\": false\n"
            << "}\n";

  for (XrSpace space : spaces) {
    check(destroy_space(space), "xrDestroySpace");
  }
  check(destroy_session(session), "xrDestroySession");
  check(destroy_instance(instance), "xrDestroyInstance");
  FreeLibrary(runtime_module);
  return EXIT_SUCCESS;
}
