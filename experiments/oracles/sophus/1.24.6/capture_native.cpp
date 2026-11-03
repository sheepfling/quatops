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
#include <vector>

struct QuaternionInput {
  std::string id;
  double w, x, y, z;
};

struct TangentInput {
  std::string id;
  double x, y, z;
};

using Vector3 = Eigen::Vector3d;
using Matrix3 = Eigen::Matrix3d;
using Rotation = Sophus::SO3d;

void write_vec3(std::ostream& out, const Vector3& value) {
  out << '[' << value.x() << ',' << value.y() << ',' << value.z() << ']';
}

void write_quaternion(std::ostream& out, const Eigen::Quaterniond& value) {
  const auto coeffs = value.coeffs();
  out << "{\"wxyz\":[" << value.w() << ',' << value.x() << ',' << value.y()
      << ',' << value.z() << "],\"coeffs_xyzw\":[" << coeffs[0] << ','
      << coeffs[1] << ',' << coeffs[2] << ',' << coeffs[3] << "]}";
}

void write_matrix(std::ostream& out, const Matrix3& value) {
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

void write_action(std::ostream& out, const Rotation& rotation,
                  const Vector3& probe) {
  const std::array<Vector3, 4> vectors = {
      Vector3::UnitX(), Vector3::UnitY(), Vector3::UnitZ(), probe};
  out << '[';
  for (std::size_t index = 0; index < vectors.size(); ++index) {
    if (index != 0) out << ',';
    write_vec3(out, rotation * vectors[index]);
  }
  out << ']';
}

void write_rotation(std::ostream& out, const std::string& id,
                    const Rotation& rotation, const Vector3& probe) {
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

void read_inputs(const char* path, Vector3& probe,
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

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: capture_native <capture_inputs.tsv>\n";
    return 2;
  }

  try {
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    Vector3 probe;
    std::vector<QuaternionInput> quaternion_inputs;
    std::vector<TangentInput> tangent_inputs;
    read_inputs(argv[1], probe, quaternion_inputs, tangent_inputs);

    std::map<std::string, Rotation> rotations;
    std::map<std::string, Vector3> tangents;
    std::cout << "{\"capture_schema\":\"quatops-sophus-discovery-v1\","
              << "\"target\":\"Sophus SO3d\",\"probe_xyz\":";
    write_vec3(std::cout, probe);
    std::cout << ",\"quaternion_construction\":[";
    for (std::size_t index = 0; index < quaternion_inputs.size(); ++index) {
      const auto& item = quaternion_inputs[index];
      const Eigen::Quaterniond input(item.w, item.x, item.y, item.z);
      const Rotation rotation(input);
      rotations.emplace(item.id, rotation);
      if (index != 0) std::cout << ',';
      std::cout << "{\"case_id\":\"" << item.id
                << "\",\"input_wxyz\":[" << item.w << ',' << item.x << ','
                << item.y << ',' << item.z << "],\"input_norm\":"
                << input.norm() << ",\"result\":";
      write_rotation(std::cout, item.id, rotation, probe);
      std::cout << ",\"log_xyz\":";
      write_vec3(std::cout, rotation.log());
      std::cout << '}';
    }

    std::cout << "],\"exp_log\":[";
    for (std::size_t index = 0; index < tangent_inputs.size(); ++index) {
      const auto& item = tangent_inputs[index];
      const Vector3 tangent(item.x, item.y, item.z);
      const Rotation rotation = Rotation::exp(tangent);
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

    const auto require_rotation = [&rotations](const std::string& id) -> const Rotation& {
      const auto found = rotations.find(id);
      if (found == rotations.end()) throw std::runtime_error("missing rotation: " + id);
      return found->second;
    };
    const auto require_tangent = [&tangents](const std::string& id) -> const Vector3& {
      const auto found = tangents.find(id);
      if (found == tangents.end()) throw std::runtime_error("missing tangent: " + id);
      return found->second;
    };

    const Rotation& mixed_a = require_rotation("exp_mixed_a");
    const Rotation& mixed_b = require_rotation("exp_mixed_b");
    const Vector3& perturbation = require_tangent("perturbation");
    const Rotation delta = Rotation::exp(perturbation);
    std::cout << "],\"composition\":[";
    write_rotation(std::cout, "mixed_a_times_mixed_b", mixed_a * mixed_b, probe);
    std::cout << ',';
    write_rotation(std::cout, "mixed_b_times_mixed_a", mixed_b * mixed_a, probe);
    std::cout << "],\"inverse\":";
    write_rotation(std::cout, "inverse_mixed_a", mixed_a.inverse(), probe);
    std::cout << ",\"perturbation_side\":[";
    write_rotation(std::cout, "delta_times_mixed_a", delta * mixed_a, probe);
    std::cout << ',';
    write_rotation(std::cout, "mixed_a_times_delta", mixed_a * delta, probe);
    std::cout << "]}\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
