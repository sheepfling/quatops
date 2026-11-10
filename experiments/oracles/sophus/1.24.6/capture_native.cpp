// Native Sophus SO(3) discovery capture. No QuatOps code is used.
#include <Eigen/Geometry>
#include <sophus/so3.hpp>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

struct QuaternionInput {
  std::string id;
  double w, x, y, z;
};

struct TangentInput {
  std::string id;
  double x, y, z;
};

template <typename Scalar>
using Vector3 = Eigen::Matrix<Scalar, 3, 1>;
template <typename Scalar>
using Rotation = Sophus::SO3<Scalar>;

template <typename Scalar>
void write_vec3(std::ostream& out, const Vector3<Scalar>& value) {
  out << '[' << value.x() << ',' << value.y() << ',' << value.z() << ']';
}

template <typename Scalar>
void write_params(std::ostream& out, const Rotation<Scalar>& rotation) {
  const auto params = rotation.params();
  out << '[' << params[0] << ',' << params[1] << ',' << params[2] << ','
      << params[3] << ']';
}

template <typename Scalar>
void write_actions(std::ostream& out, const Rotation<Scalar>& rotation,
                   const Vector3<Scalar>& probe, bool include_probe) {
  out << ",[";
  write_vec3(out, rotation * Vector3<Scalar>::UnitX());
  out << ',';
  write_vec3(out, rotation * Vector3<Scalar>::UnitY());
  out << ',';
  write_vec3(out, rotation * Vector3<Scalar>::UnitZ());
  out << "],";
  if (include_probe) write_vec3(out, rotation * probe);
  else out << "null";
}

template <typename Scalar>
void write_rotation_row(std::ostream& out, const char* id,
                        const Rotation<Scalar>& rotation,
                        const Vector3<Scalar>& probe, bool include_probe) {
  out << "[\"" << id << "\",";
  write_params(out, rotation);
  write_actions(out, rotation, probe, include_probe);
  out << ']';
}

template <typename Scalar>
void read_inputs(const char* path, Vector3<Scalar>& probe,
                 std::vector<QuaternionInput>& quaternions,
                 std::vector<TangentInput>& tangents) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("cannot open input TSV");

  std::string kind;
  bool found_probe = false;
  while (input >> kind) {
    if (kind == "probe") {
      input >> probe[0] >> probe[1] >> probe[2];
      found_probe = true;
    } else if (kind == "quaternion") {
      QuaternionInput item;
      input >> item.id >> item.w >> item.x >> item.y >> item.z;
      quaternions.push_back(item);
    } else if (kind == "tangent") {
      TangentInput item;
      input >> item.id >> item.x >> item.y >> item.z;
      tangents.push_back(item);
    } else if (!kind.empty() && kind[0] == '#') {
      std::string ignored;
      std::getline(input, ignored);
    } else {
      throw std::runtime_error("unknown input row kind: " + kind);
    }
    if (!input) throw std::runtime_error("malformed input TSV row");
  }
  if (!found_probe || quaternions.empty() || tangents.empty()) {
    throw std::runtime_error("input TSV must define probe, quaternion, tangent");
  }
}

template <typename Scalar>
int capture(const char* input_path) {
  std::cout << std::setprecision(std::numeric_limits<Scalar>::max_digits10);
  Vector3<Scalar> probe;
  std::vector<QuaternionInput> quaternion_inputs;
  std::vector<TangentInput> tangent_inputs;
  read_inputs(input_path, probe, quaternion_inputs, tangent_inputs);

  std::map<std::string, Rotation<Scalar>> rotations;
  std::map<std::string, Vector3<Scalar>> tangents;
  const char* target =
      std::is_same_v<Scalar, float> ? "Sophus SO3f" : "Sophus SO3d";
  std::cout << "{\"capture_schema\":\"quatops-sophus-direct-v1\","
            << "\"target\":\"" << target << "\",\"probe_xyz\":";
  write_vec3(std::cout, probe);
  std::cout << ",\"quaternion_construction\":[";
  for (std::size_t index = 0; index < quaternion_inputs.size(); ++index) {
    const auto& item = quaternion_inputs[index];
    const Eigen::Quaternion<Scalar> input(
        static_cast<Scalar>(item.w), static_cast<Scalar>(item.x),
        static_cast<Scalar>(item.y), static_cast<Scalar>(item.z));
    const Rotation<Scalar> rotation(input);
    rotations.emplace(item.id, rotation);
    if (index != 0) std::cout << ',';
    std::cout << "[\"" << item.id << "\",[" << input.w() << ',' << input.x()
              << ',' << input.y() << ',' << input.z() << "]," << input.norm() << ',';
    write_params(std::cout, rotation);
    std::cout << ",[";
    write_vec3(std::cout, rotation * Vector3<Scalar>::UnitX()); std::cout << ',';
    write_vec3(std::cout, rotation * Vector3<Scalar>::UnitY()); std::cout << ',';
    write_vec3(std::cout, rotation * Vector3<Scalar>::UnitZ()); std::cout << "],";
    write_vec3(std::cout, rotation.log()); std::cout << ',';
    if (item.id == "mixed_unequal") write_vec3(std::cout, rotation * probe);
    else std::cout << "null";
    std::cout << ']';
  }
  std::cout << "],\"exp_log\":[";
  bool first = true;
  for (const auto& item : tangent_inputs) {
    const Vector3<Scalar> tangent(static_cast<Scalar>(item.x),
        static_cast<Scalar>(item.y), static_cast<Scalar>(item.z));
    const Rotation<Scalar> rotation = Rotation<Scalar>::exp(tangent);
    tangents.emplace(item.id, tangent);
    rotations.emplace("exp_" + item.id, rotation);
    if (item.id == "perturbation") continue;
    if (!first) std::cout << ',';
    first = false;
    std::cout << "[\"" << item.id << "\",";
    write_vec3(std::cout, tangent); std::cout << ',';
    write_params(std::cout, rotation); std::cout << ",[";
    write_vec3(std::cout, rotation * Vector3<Scalar>::UnitX()); std::cout << ',';
    write_vec3(std::cout, rotation * Vector3<Scalar>::UnitY()); std::cout << ',';
    write_vec3(std::cout, rotation * Vector3<Scalar>::UnitZ()); std::cout << "],";
    write_vec3(std::cout, rotation.log()); std::cout << ',';
    if (item.id == "mixed_a") write_vec3(std::cout, rotation * probe);
    else std::cout << "null";
    std::cout << ']';
  }
  const Rotation<Scalar>& a = rotations.at("exp_mixed_a");
  const Rotation<Scalar>& b = rotations.at("exp_mixed_b");
  const Rotation<Scalar> delta = Rotation<Scalar>::exp(tangents.at("perturbation"));
  std::cout << "],\"composition\":[";
  write_rotation_row(std::cout, "mixed_a_times_mixed_b", a * b, probe, true);
  std::cout << ',';
  write_rotation_row(std::cout, "mixed_b_times_mixed_a", b * a, probe, true);
  std::cout << "],\"inverse\":";
  write_rotation_row(std::cout, "inverse_mixed_a", a.inverse(), probe, false);
  std::cout << ",\"perturbation_side\":[";
  write_rotation_row(std::cout, "delta_times_mixed_a", delta * a, probe, false);
  std::cout << ',';
  write_rotation_row(std::cout, "mixed_a_times_delta", a * delta, probe, false);
  std::cout << "]}\n";
  return 0;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: capture_native <capture_inputs.tsv> <float32|float64>\n";
    return 2;
  }

  try {
    const std::string profile = argv[2];
    if (profile == "float32") return capture<float>(argv[1]);
    if (profile == "float64") return capture<double>(argv[1]);
    throw std::runtime_error("scalar profile must be float32 or float64");
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
