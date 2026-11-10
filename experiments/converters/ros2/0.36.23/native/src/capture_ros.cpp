// Native calls only. Built and run against the pinned ROS 2 Jazzy container.
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>
#include <nlohmann/json.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2/LinearMath/Quaternion.hpp>
#include <tf2/LinearMath/Matrix3x3.hpp>
#include <tf2/LinearMath/Transform.hpp>
#include <tf2/buffer_core.hpp>
#include <tf2/time.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

using json = nlohmann::json;
using Native = std::remove_cvref_t<decltype(std::declval<tf2::Quaternion>().x())>;

json qarray(const tf2::Quaternion& q) {return json::array({q.x(),q.y(),q.z(),q.w()});}
json varray(const tf2::Vector3& v) {return json::array({v.x(),v.y(),v.z()});}
json matrix(const tf2::Matrix3x3& m) {
  json rows=json::array();
  for(int r=0;r<3;++r) rows.push_back(json::array({m[r][0],m[r][1],m[r][2]}));
  return rows;
}
json transform(const tf2::Transform& t) {
  return {{"quaternion_xyzw",qarray(t.getRotation())},{"translation_xyz",varray(t.getOrigin())},
          {"matrix_rows",matrix(t.getBasis())}};
}
geometry_msgs::msg::TransformStamped stamped(const std::string& parent,const std::string& child,const tf2::Transform& t, int stamp_sec) {
  geometry_msgs::msg::TransformStamped msg;
  msg.header.frame_id=parent;msg.child_frame_id=child;msg.header.stamp.sec=stamp_sec;msg.header.stamp.nanosec=0;
  msg.transform.rotation=tf2::toMsg(t.getRotation());
  msg.transform.translation.x=t.getOrigin().x();msg.transform.translation.y=t.getOrigin().y();msg.transform.translation.z=t.getOrigin().z();
  return msg;
}
json stamped_message(const geometry_msgs::msg::TransformStamped& msg) {
  return {{"parent_frame",msg.header.frame_id},{"child_frame",msg.child_frame_id},
          {"stamp_sec",msg.header.stamp.sec},{"stamp_nanosec",msg.header.stamp.nanosec},
          {"translation_xyz",json::array({msg.transform.translation.x,msg.transform.translation.y,msg.transform.translation.z})},
          {"rotation_xyzw",json::array({msg.transform.rotation.x,msg.transform.rotation.y,msg.transform.rotation.z,msg.transform.rotation.w})}};
}
int main(int argc,char** argv) {
  try {
    if(argc!=2) throw std::runtime_error("Usage: capture_ros CHAIN_INPUTS.json");
    std::ifstream file(argv[1]);if(!file)throw std::runtime_error("Cannot read chain inputs");
    json corpus;file>>corpus;json result=json::array();
    const auto add=[&](std::string id,std::string op,std::string api,json in,json out){
      result.push_back({{"case_id",id},{"operation_id",op},{"native_entry_point",api},{"native_inputs",in},{"native_outputs",out}});
    };
    const auto& world_input=corpus.at("world_from_base");
    const auto& sensor_input=corpus.at("base_from_sensor");
    const auto& world_rpy=world_input.at("rpy_radians");
    const auto& sensor_rpy=sensor_input.at("rpy_radians");
    const auto& world_xyz=world_input.at("translation_xyz");
    const auto& sensor_xyz=sensor_input.at("translation_xyz");
    const auto& probe_xyz=corpus.at("probe_in_source_xyz");
    const auto stamp_sec=corpus.at("stamp_sec").get<int>();
    tf2::Quaternion qa,qb;
    qa.setRPY(world_rpy[0].get<Native>(),world_rpy[1].get<Native>(),world_rpy[2].get<Native>());
    qb.setRPY(sensor_rpy[0].get<Native>(),sensor_rpy[1].get<Native>(),sensor_rpy[2].get<Native>());
    const tf2::Transform world_base(qa,tf2::Vector3(world_xyz[0].get<Native>(),world_xyz[1].get<Native>(),world_xyz[2].get<Native>()));
    const tf2::Transform base_sensor(qb,tf2::Vector3(sensor_xyz[0].get<Native>(),sensor_xyz[1].get<Native>(),sensor_xyz[2].get<Native>()));
    const tf2::Vector3 probe(probe_xyz[0].get<Native>(),probe_xyz[1].get<Native>(),probe_xyz[2].get<Native>());
    tf2::BufferCore buffer;
    const auto first=stamped("world","base",world_base,stamp_sec),second=stamped("base","sensor",base_sensor,stamp_sec);
    if(!buffer.setTransform(first,"quatops_capture",true)||!buffer.setTransform(second,"quatops_capture",true))
      throw std::runtime_error("BufferCore rejected declared static chain");
    const auto timestamp=tf2::TimePoint(std::chrono::seconds(stamp_sec));
    for(const auto& frames:std::vector<std::pair<std::string,std::string>>{{"world","sensor"},{"sensor","world"}}) {
      const auto message=buffer.lookupTransform(frames.first,frames.second,timestamp);
      tf2::Transform lookup;tf2::fromMsg(message.transform,lookup);
      add(corpus.at("case_id").get<std::string>(), "lookup-"+frames.first+"-from-"+frames.second,"tf2::BufferCore::setTransform; lookupTransform; tf2::Transform::operator*",
          {{"target_frame",frames.first},{"source_frame",frames.second},{"stamp_sec",stamp_sec},
           {"world_from_base",transform(world_base)},{"base_from_sensor",transform(base_sensor)},{"probe_in_source_xyz",varray(probe)},
           {"world_from_base_rpy",world_rpy},{"base_from_sensor_rpy",sensor_rpy}},
          {{"lookup_message",stamped_message(message)},{"lookup",transform(lookup)},
           {"message_roundtrip",transform(lookup)},{"probe_in_target_xyz",varray(lookup*probe)},
           {"direct_world_from_sensor",transform(world_base*base_sensor)},
           {"direct_sensor_from_world",transform((world_base*base_sensor).inverse())},
           {"world_probe_sequential",varray(world_base*(base_sensor*probe))}});
    }
    std::cout<<json({{"native_scalar_bytes",sizeof(Native)},{"compiler",__VERSION__},
      {"cpp_standard",__cplusplus},{"flags","CMake: -Wall -Wextra -Wpedantic -fno-fast-math; C++20"},{"observations",result}}).dump(2)<<'\n';
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
