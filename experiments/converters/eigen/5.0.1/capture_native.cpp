// Native Eigen quaternion boundary capture. No QuatOps code is used.
#include <Eigen/Geometry>

#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct InputCase {
  std::string id;
  double w, x, y, z;
};

struct AxisAngleCase {
  std::string id;
  double x, y, z, angle;
};

template <typename T>
using Quaternion = Eigen::Quaternion<T>;

template <typename T>
using Vector3 = Eigen::Matrix<T, 3, 1>;

template <typename T>
void write_vec3(std::ostream& out, const Vector3<T>& value) {
  out << '[' << value[0] << ',' << value[1] << ',' << value[2] << ']';
}

template <typename T>
void write_quaternion(std::ostream& out, const Quaternion<T>& q,
                      bool include_coeffs = true) {
  out << "{\"wxyz\":[" << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z() << ']';
  if (include_coeffs) {
    const auto coeffs = q.coeffs();
    out << ",\"coeffs_xyzw\":[" << coeffs[0] << ',' << coeffs[1] << ','
        << coeffs[2] << ',' << coeffs[3] << ']';
  }
  out << '}';
}

template <typename T>
void write_basis_actions(std::ostream& out, const Quaternion<T>& q) {
  const std::array<Vector3<T>, 3> vectors = {
      Vector3<T>::UnitX(), Vector3<T>::UnitY(), Vector3<T>::UnitZ()};
  out << '[';
  for (std::size_t i = 0; i < vectors.size(); ++i) {
    if (i) out << ',';
    write_vec3(out, (q * vectors[i]).eval());
  }
  out << ']';
}

template <typename T>
void write_probe_action(std::ostream& out, const Quaternion<T>& q,
                        const Vector3<double>& probe) {
  write_vec3(out, (q * probe.template cast<T>()).eval());
}

template <typename T>
void write_orientation(std::ostream& out, const InputCase& item, const Vector3<double>& probe) {
  const Quaternion<T> q(static_cast<T>(item.w), static_cast<T>(item.x),
                        static_cast<T>(item.y), static_cast<T>(item.z));
  out << "{\"case_id\":\"" << item.id << "\",\"quaternion\":";
  write_quaternion(out, q, item.id == "mixed_a");
  out << ",\"native_quaternion_basis_actions_xyz\":";
  write_basis_actions(out, q);
  if (item.id == "mixed_a") {
    out << ",\"native_quaternion_probe_action_xyz\":";
    write_probe_action(out, q, probe);
  }
  out << '}';
}

template <typename T>
void write_composition(std::ostream& out, const std::string& case_id,
                       const std::string& lhs_id, const std::string& rhs_id,
                       const Quaternion<T>& lhs, const Quaternion<T>& rhs) {
  const Quaternion<T> product = lhs * rhs;
  out << "{\"case_id\":\"" << case_id << "\",\"lhs_case_id\":\""
      << lhs_id << "\",\"rhs_case_id\":\"" << rhs_id
      << "\",\"product_order\":\"lhs * rhs; rhs acts first\",\"quaternion\":";
  write_quaternion(out, product, false);
  out << ",\"native_quaternion_basis_actions_xyz\":";
  write_basis_actions(out, product);
  out << '}';
}

template <typename T>
void write_profile(std::ostream& out, const std::string& dtype,
                   const std::vector<InputCase>& cases, const AxisAngleCase& axis_angle,
                   const Vector3<double>& probe) {
  out << std::setprecision(std::numeric_limits<T>::max_digits10);
  std::vector<Quaternion<T>> quaternions;
  quaternions.reserve(cases.size());
  for (const auto& item : cases) {
    quaternions.emplace_back(static_cast<T>(item.w), static_cast<T>(item.x),
                             static_cast<T>(item.y), static_cast<T>(item.z));
  }

  out << "{\"dtype\":\"" << dtype << "\",\"scalar_bytes\":" << sizeof(T)
      << ",\"orientations\":[";
  for (std::size_t i = 0; i < cases.size(); ++i) {
    if (i) out << ',';
    write_orientation<T>(out, cases[i], probe);
  }
  out << "],\"native_created\":[";
  const Vector3<T> axis(static_cast<T>(axis_angle.x), static_cast<T>(axis_angle.y),
                        static_cast<T>(axis_angle.z));
  const Eigen::AngleAxis<T> angle_axis(static_cast<T>(axis_angle.angle), axis.normalized());
  const Quaternion<T> native_q(angle_axis);
  out << "{\"case_id\":\"" << axis_angle.id
      << "\",\"constructor\":\"Eigen::Quaternion(Eigen::AngleAxis)\",\"quaternion\":";
  write_quaternion(out, native_q, false);
  out << ",\"native_quaternion_basis_actions_xyz\":";
  write_basis_actions(out, native_q);
  out << "}]";

  std::size_t a_index = cases.size(), b_index = cases.size();
  for (std::size_t i = 0; i < cases.size(); ++i) {
    if (cases[i].id == "mixed_a") a_index = i;
    if (cases[i].id == "mixed_b") b_index = i;
  }
  if (a_index == cases.size() || b_index == cases.size()) {
    throw std::runtime_error("required mixed_a and mixed_b inputs are missing");
  }
  out << ",\"composition\":[";
  write_composition(out, "mixed_a_times_mixed_b", "mixed_a", "mixed_b",
                    quaternions[a_index], quaternions[b_index]);
  out << ',';
  write_composition(out, "mixed_b_times_mixed_a", "mixed_b", "mixed_a",
                    quaternions[b_index], quaternions[a_index]);
  out << "]}";
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: capture_native <input.tsv>\n";
    return 2;
  }
  std::ifstream in(argv[1]);
  if (!in) {
    std::cerr << "cannot open native input\n";
    return 2;
  }
  std::size_t count = 0;
  Vector3<double> probe;
  in >> count >> probe[0] >> probe[1] >> probe[2];
  std::vector<InputCase> cases(count);
  for (auto& item : cases) in >> item.id >> item.w >> item.x >> item.y >> item.z;
  AxisAngleCase axis_angle;
  in >> axis_angle.id >> axis_angle.x >> axis_angle.y >> axis_angle.z >> axis_angle.angle;
  if (!in) {
    std::cerr << "malformed native input\n";
    return 2;
  }

  std::cout << "{\"eigen_version\":\"" << EIGEN_VERSION_STRING
            << "\",\"version_macros\":[" << EIGEN_WORLD_VERSION << ','
            << EIGEN_MAJOR_VERSION << ',' << EIGEN_MINOR_VERSION << ','
            << EIGEN_PATCH_VERSION << "],\"profiles\":[";
  write_profile<double>(std::cout, "float64", cases, axis_angle, probe);
  std::cout << ',';
  write_profile<float>(std::cout, "float32", cases, axis_angle, probe);
  std::cout << "]}\n";
  return 0;
}
