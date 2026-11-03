// Native calls only. Prepared for ROS 2 Jazzy; compilation/runtime remain unverified here.
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
tf2::Quaternion makeq(const json& wxyz) {
  return {wxyz.at(1).get<Native>(),wxyz.at(2).get<Native>(),wxyz.at(3).get<Native>(),wxyz.at(0).get<Native>()};
}
geometry_msgs::msg::TransformStamped stamped(const std::string& parent,const std::string& child,const tf2::Transform& t) {
  geometry_msgs::msg::TransformStamped msg;
  msg.header.frame_id=parent;msg.child_frame_id=child;msg.header.stamp.sec=42;msg.header.stamp.nanosec=0;
  msg.transform.rotation=tf2::toMsg(t.getRotation());
  msg.transform.translation.x=t.getOrigin().x();msg.transform.translation.y=t.getOrigin().y();msg.transform.translation.z=t.getOrigin().z();
  return msg;
}
int main(int argc,char** argv) {
  try {
    if(argc!=2) throw std::runtime_error("Usage: capture_ros CORPUS.json");
    std::ifstream file(argv[1]);if(!file)throw std::runtime_error("Cannot read corpus");
    json corpus;file>>corpus;json result=json::array();
    const auto add=[&](std::string id,std::string op,std::string api,json in,json out){
      result.push_back({{"case_id",id},{"operation_id",op},{"native_entry_point",api},{"native_inputs",in},{"native_outputs",out}});
    };
    for(const auto& item:corpus.at("euler_inputs")) {
      const auto& xyz=item.at("xyz");
      tf2::Quaternion q;q.setRPY(xyz.at(0).get<Native>(),xyz.at(1).get<Native>(),xyz.at(2).get<Native>());
      const tf2::Matrix3x3 m(q);Native roll{},pitch{},yaw{};m.getRPY(roll,pitch,yaw);
      const auto message=tf2::toMsg(q);tf2::Quaternion back;tf2::fromMsg(message,back);
      json probes=json::array();
      for(const auto& p:corpus.at("vectors_xyz"))probes.push_back(varray(tf2::quatRotate(q,tf2::Vector3(p.at(0).get<Native>(),p.at(1).get<Native>(),p.at(2).get<Native>()))));
      add(item.at("case_id").get<std::string>(),"setRPY-message-action",
          "tf2::Quaternion::setRPY; Matrix3x3::getRPY; tf2::quatRotate; toMsg/fromMsg",
          {{"roll_pitch_yaw_radians",xyz}},{{"quaternion_xyzw",qarray(q)},{"matrix_rows",matrix(m)},
          {"readback_rpy",json::array({roll,pitch,yaw})},{"message_xyzw",json::array({message.x,message.y,message.z,message.w})},
          {"message_roundtrip_xyzw",qarray(back)},{"basis_and_probe_images",probes}});
    }
    for(const auto& pair:corpus.at("multiplication_pairs")) {
      const auto a=makeq(pair.at("a_wxyz")),b=makeq(pair.at("b_wxyz"));
      add(pair.at("case_id").get<std::string>(),"products","tf2::Quaternion::operator*",
          {{"a_xyzw",qarray(a)},{"b_xyzw",qarray(b)}},{{"ab_xyzw",qarray(a*b)},{"ba_xyzw",qarray(b*a)}});
    }
    for(const auto& item:corpus.at("matrix_inputs")) {
      if(item.at("domain")!="proper_rotation")continue;
      const auto& r=item.at("rows");
      tf2::Matrix3x3 m(r[0][0].get<Native>(),r[0][1].get<Native>(),r[0][2].get<Native>(),
                      r[1][0].get<Native>(),r[1][1].get<Native>(),r[1][2].get<Native>(),
                      r[2][0].get<Native>(),r[2][1].get<Native>(),r[2][2].get<Native>());
      tf2::Quaternion q;m.getRotation(q);
      add(item.at("case_id").get<std::string>(),"matrix-to-quaternion","tf2::Matrix3x3::getRotation",
          {{"matrix_rows",matrix(m)}},{{"quaternion_xyzw",qarray(q)}});
    }
    tf2::Quaternion qa,qb;qa.setRPY(.2,-.3,.5);qb.setRPY(-.7,.4,.1);
    const tf2::Transform world_base(qa,tf2::Vector3(3,5,-7));
    const tf2::Transform base_sensor(qb,tf2::Vector3(1.25,-2.5,3.75));
    const tf2::Vector3 probe(2,-3,5);
    tf2::BufferCore buffer;
    const auto first=stamped("world","base",world_base),second=stamped("base","sensor",base_sensor);
    if(!buffer.setTransform(first,"quatops_capture",true)||!buffer.setTransform(second,"quatops_capture",true))
      throw std::runtime_error("BufferCore rejected declared static chain");
    const auto timestamp=tf2::TimePoint(std::chrono::seconds(42));
    for(const auto& frames:std::vector<std::pair<std::string,std::string>>{{"world","sensor"},{"sensor","world"}}) {
      const auto message=buffer.lookupTransform(frames.first,frames.second,timestamp);
      tf2::Transform lookup;tf2::fromMsg(message.transform,lookup);
      add("world-base-sensor", "lookup-"+frames.first+"-from-"+frames.second,"tf2::BufferCore::setTransform; lookupTransform; tf2::Transform::operator*",
          {{"target_frame",frames.first},{"source_frame",frames.second},{"stamp_sec",42},
           {"world_from_base",transform(world_base)},{"base_from_sensor",transform(base_sensor)},{"probe_in_source_xyz",varray(probe)}},
          {{"lookup",transform(lookup)},{"probe_in_target_xyz",varray(lookup*probe)},
           {"direct_world_from_sensor",transform(world_base*base_sensor)},
           {"direct_sensor_from_world",transform((world_base*base_sensor).inverse())},
           {"world_probe_sequential",varray(world_base*(base_sensor*probe))}});
    }
    std::cout<<json({{"native_scalar_bytes",sizeof(Native)},{"compiler",__VERSION__},
      {"cpp_standard",__cplusplus},{"flags","-fno-fast-math; see retained build log"},{"observations",result}}).dump(2)<<'\n';
    return 0;
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
