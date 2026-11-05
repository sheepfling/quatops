// Native Sophus SO(3) discovery capture. No QuatOps code is used.
#include <Eigen/Geometry>
#include <sophus/so3.hpp>

#include <array>
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
using Matrix3 = Eigen::Matrix<Scalar, 3, 3>;
template <typename Scalar>
using Rotation = Sophus::SO3<Scalar>;

template <typename Scalar>
void write_vec3(std::ostream& out, const Vector3<Scalar>& value) {
  out << '[' << value.x() << ',' << value.y() << ',' << value.z() << ']';
}

template <typename Scalar>
void write_quaternion(std::ostream& out,
                      const Eigen::Quaternion<Scalar>& value) {
  const auto coeffs = value.coeffs();
  out << "{\"wxyz\":[" << value.w() << ',' << value.x() << ',' << value.y()
      << ',' << value.z() << "],\"coeffs_xyzw\":[" << coeffs[0] << ','
      << coeffs[1] << ',' << coeffs[2] << ',' << coeffs[3] << "]}";
}

template <typename Scalar>
void write_matrix(std::ostream& out, const Matrix3<Scalar>& value) {
  out << '[';
  for (int row = 0; row < 3; ++row) {
    if (row != 0) out << ',';
    out << '[';
    for (int column = 0; column < 3; ++column) {
      if (column != 0) out << ',';
      out << value(row, column);
    }
    out << ']';
  }
  out << ']';
}

template <typename Scalar>
void write_action(std::ostream& out, const Rotation<Scalar>& rotation,
                  const Vector3<Scalar>& probe) {
  const std::array<Vector3<Scalar>, 4> vectors = {
      Vector3<Scalar>::UnitX(), Vector3<Scalar>::UnitY(),
      Vector3<Scalar>::UnitZ(), probe};
  out << '[';
  for (std::size_t index = 0; index < vectors.size(); ++index) {
    if (index != 0) out << ',';
    write_vec3(out, rotation * vectors[index]);
  }
  out << ']';
}

template <typename Scalar>
void write_rotation(std::ostream& out, const std::string& id,
                    const Rotation<Scalar>& rotation,
                    const Vector3<Scalar>& probe) {
  out << "{\"case_id\":\"" << id << "\",\"unit_quaternion\":";
  write_quaternion(out, rotation.unit_quaternion());
  out << ",\"params_xyzw\":[";
  const auto params = rotation.params();
  for (int index = 0; index < 4; ++index) {
    if (index != 0) out << ',';
    out << params[index];
  }
  out << "],\"matrix_rows\":";
  write_matrix(out, rotation.matrix());
  out << ",\"native_vector_actions_xyz\":";
  write_action(out, rotation, probe);
  out << '}';
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
  std::cout << "{\"capture_schema\":\"quatops-sophus-discovery-v1\","
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
    std::cout << "{\"case_id\":\"" << item.id
              << "\",\"input_wxyz\":[" << input.w() << ',' << input.x()
              << ',' << input.y() << ',' << input.z() << "],\"input_norm\":"
              << input.norm() << ",\"result\":";
    write_rotation(std::cout, item.id, rotation, probe);
    std::cout << ",\"log_xyz\":";
    write_vec3(std::cout, rotation.log());
    std::cout << '}';
  }

  std::cout << "],\"exp_log\":[";
  for (std::size_t index = 0; index < tangent_inputs.size(); ++index) {
    const auto& item = tangent_inputs[index];
    const Vector3<Scalar> tangent(
        static_cast<Scalar>(item.x), static_cast<Scalar>(item.y),
        static_cast<Scalar>(item.z));
    const Rotation<Scalar> rotation = Rotation<Scalar>::exp(tangent);
    tangents.emplace(item.id, tangent);
    rotations.emplace("exp_" + item.id, rotation);
    if (index != 0) std::cout << ',';
    std::cout << "{\"case_id\":\"" << item.id << "\",\"tangent_xyz\":";
    write_vec3(std::cout, tangent);
    std::cout << ",\"exp_result\":";
    write_rotation(std::cout, item.id, rotation, probe);
    std::cout << ",\"log_of_exp_xyz\":";
    write_vec3(std::cout, rotation.log());
    std::cout << '}';
  }

  const auto require_rotation = [&rotations](const std::string& id)
      -> const Rotation<Scalar>& {
    const auto found = rotations.find(id);
    if (found == rotations.end()) {
      throw std::runtime_error("missing rotation: " + id);
    }
    return found->second;
  };
  const auto require_tangent = [&tangents](const std::string& id)
      -> const Vector3<Scalar>& {
    const auto found = tangents.find(id);
    if (found == tangents.end()) {
      throw std::runtime_error("missing tangent: " + id);
    }
    return found->second;
  };

  const Rotation<Scalar>& mixed_a = require_rotation("exp_mixed_a");
  const Rotation<Scalar>& mixed_b = require_rotation("exp_mixed_b");
  const Vector3<Scalar>& perturbation = require_tangent("perturbation");
  const Rotation<Scalar> delta = Rotation<Scalar>::exp(perturbation);
  std::cout << "],\"composition\":[";
  write_rotation(std::cout, "mixed_a_times_mixed_b", mixed_a * mixed_b,
                 probe);
  std::cout << ',';
  write_rotation(std::cout, "mixed_b_times_mixed_a", mixed_b * mixed_a,
                 probe);
  std::cout << "],\"inverse\":";
  write_rotation(std::cout, "inverse_mixed_a", mixed_a.inverse(), probe);
  std::cout << ",\"perturbation_side\":[";
  write_rotation(std::cout, "delta_times_mixed_a", delta * mixed_a, probe);
  std::cout << ',';
  write_rotation(std::cout, "mixed_a_times_delta", mixed_a * delta, probe);
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
