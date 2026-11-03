/** Actual Cesium calls only. Local HPR and geographic/ECEF orientation are separate profiles. */
import * as C from 'cesium';
import { fileURLToPath } from 'node:url';
import { args, capture, read, numeric as n, contract } from './capture_io.mjs';

const config=args(), corpus=read(config.corpus), geographic=read(config.geographic);
const cap=capture({id:'cesium',version:C.VERSION,entry:fileURLToPath(import.meta.resolve('cesium')),config,
  limitations:['No Model/glTF import, rendering, CZML, ICRF or time-dependent EOP transforms.',
    'Exact-pole ENU values are native fallback evidence, not a universal geographic east direction.',
    'This exporter was syntax-checked only in the delivery environment; new manifest proves actual execution.']});
cap.asset(config.geographic);
const qv=q=>[q.x,q.y,q.z,q.w], cv=p=>[p.x,p.y,p.z];
const quat=a=>new C.Quaternion(a[1],a[2],a[3],a[0]);
const rows3=m=>Array.from({length:3},(_,r)=>Array.from({length:3},(_,c)=>m[c*3+r]));
const rows4=m=>Array.from({length:4},(_,r)=>Array.from({length:4},(_,c)=>m[c*4+r]));
const basis=m=>corpus.vectors_xyz.map(p=>cv(C.Matrix3.multiplyByVector(m,C.Cartesian3.fromArray(p),new C.Cartesian3())));
for(const item of corpus.euler_inputs) {
  const [x,y,z]=item.xyz;
  const hpr=new C.HeadingPitchRoll(z,y,x);
  const q=C.Quaternion.fromHeadingPitchRoll(hpr,new C.Quaternion());
  const m=C.Matrix3.fromQuaternion(q,new C.Matrix3());
  const back=C.HeadingPitchRoll.fromQuaternion(q,new C.HeadingPitchRoll());
  cap.add(item.case_id,'local-hpr','Quaternion.fromHeadingPitchRoll; HeadingPitchRoll.fromQuaternion; Matrix3.fromQuaternion',
    {heading:n(hpr.heading),pitch:n(hpr.pitch),roll:n(hpr.roll)},
    {quaternion_xyzw:n(qv(q)),matrix_rows:n(rows3(m)),matrix_native_column_major:n(C.Matrix3.toArray(m)),
     basis_and_probe_images:n(basis(m)),readback_hpr:n([back.heading,back.pitch,back.roll])});
}
for(const pair of corpus.multiplication_pairs) for(const [suffix,a,b] of [['ab',pair.a_wxyz,pair.b_wxyz],['ba',pair.b_wxyz,pair.a_wxyz]]) {
  const qa=quat(a),qb=quat(b),result=C.Quaternion.multiply(qa,qb,new C.Quaternion());
  cap.add(pair.case_id,`multiply-${suffix}`,'Quaternion.multiply',{lhs_xyzw:n(qv(qa)),rhs_xyzw:n(qv(qb))},{product_xyzw:n(qv(result))},contract({units:'dimensionless; general algebra'}));
}
for(const item of corpus.matrix_inputs.filter(x=>x.domain==='proper_rotation')) {
  const m=C.Matrix3.fromRowMajorArray(item.rows.flat(),new C.Matrix3());
  const q=C.Quaternion.fromRotationMatrix(m,new C.Quaternion());
  cap.add(item.case_id,'matrix-to-quaternion','Matrix3.fromRowMajorArray; Quaternion.fromRotationMatrix',
    {matrix_rows:n(rows3(m))},{quaternion_xyzw:n(qv(q)),reconstructed_matrix_rows:n(rows3(C.Matrix3.fromQuaternion(q,new C.Matrix3())))});
}
for(const pair of corpus.interpolation_pairs) for(const t of corpus.interpolation_weights.filter(t=>t>=0&&t<=1)) {
  const a=quat(pair.a_wxyz),b=quat(pair.b_wxyz),q=C.Quaternion.slerp(a,b,t,new C.Quaternion());
  cap.add(pair.case_id,`slerp-${t}`,'Quaternion.slerp',{a_xyzw:n(qv(a)),b_xyzw:n(qv(b)),t:n(t)},{quaternion_xyzw:n(qv(q))});
}
for(const location of geographic.locations) {
  const origin=C.Cartesian3.fromDegrees(location.longitude_degrees,location.latitude_degrees,location.height_m,C.Ellipsoid.WGS84);
  const enu=C.Transforms.eastNorthUpToFixedFrame(origin,C.Ellipsoid.WGS84,new C.Matrix4());
  for(const [i,angles] of geographic.heading_pitch_roll_radians.entries()) {
    const hpr=new C.HeadingPitchRoll(...angles);
    const world=C.Transforms.headingPitchRollToFixedFrame(origin,hpr,C.Ellipsoid.WGS84,C.Transforms.eastNorthUpToFixedFrame,new C.Matrix4());
    const q=C.Transforms.headingPitchRollQuaternion(origin,hpr,C.Ellipsoid.WGS84,C.Transforms.eastNorthUpToFixedFrame,new C.Quaternion());
    const point=C.Matrix4.multiplyByPoint(world,C.Cartesian3.fromArray(geographic.local_point_xyz_m),new C.Cartesian3());
    const direction=C.Matrix4.multiplyByPointAsVector(world,C.Cartesian3.fromArray(geographic.local_direction_xyz),new C.Cartesian3());
    const inverse=C.Matrix4.inverseTransformation(world,new C.Matrix4());
    const back=C.Matrix4.multiplyByPoint(inverse,point,new C.Cartesian3());
    cap.add(location.case_id,`geographic-hpr-${i}`,'Cartesian3.fromDegrees; Transforms.eastNorthUpToFixedFrame; headingPitchRollToFixedFrame; headingPitchRollQuaternion',
      {longitude_degrees:n(location.longitude_degrees),latitude_degrees:n(location.latitude_degrees),height_m:n(location.height_m),hpr_radians:n(angles),
       local_point_xyz_m:n(geographic.local_point_xyz_m),local_direction_xyz:n(geographic.local_direction_xyz)},
      {origin_ecef_m:n(cv(origin)),enu_to_ecef_matrix_rows:n(rows4(enu)),model_to_ecef_matrix_rows:n(rows4(world)),
       quaternion_ecef_xyzw:n(qv(q)),world_point_xyz_m:n(cv(point)),world_direction_xyz:n(cv(direction)),local_point_readback:n(cv(back))},
      contract({units:'angles radians except geodetic degrees; positions metres',frame_direction:'model -> local ENU -> WGS84 ECEF'}));
  }
}
cap.finish(['EULER','ALGEBRA','MATRIX','ACTION','INTERPOLATION','FRAME']);
