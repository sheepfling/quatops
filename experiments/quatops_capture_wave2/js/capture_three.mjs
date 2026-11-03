/** Three.js math and actual GLTFLoader/AnimationMixer captures (no renderer). */
import * as T from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { fileURLToPath } from 'node:url';
import fs from 'node:fs';
import path from 'node:path';
import { ROOT, args, capture, read, numeric as n, contract } from './capture_io.mjs';

// Node lacks the browser progress-event class. This only adapts I/O notifications.
if(typeof globalThis.ProgressEvent==='undefined') globalThis.ProgressEvent=class extends Event {
  constructor(type,init={}) {super(type);Object.assign(this,init);}
};
const config=args(),corpus=read(config.corpus);
const cap=capture({id:'three_and_gltfloader',version:T.REVISION,entry:fileURLToPath(import.meta.resolve('three')),config,family:'three',
  limitations:['No renderer, GPU, compressed textures, skins or morphs.',
    'GLTFLoader and Three.js math are one implementation family, not independent votes.']});
const rows=m=>Array.from({length:4},(_,r)=>Array.from({length:4},(_,c)=>m.elements[c*4+r]));
for(const item of corpus.euler_inputs) for(const order of corpus.euler_orders) {
  const e=new T.Euler(...item.xyz,order),q=new T.Quaternion().setFromEuler(e),m=new T.Matrix4().makeRotationFromQuaternion(q);
  const back=new T.Euler().setFromQuaternion(q,order);
  cap.add(item.case_id,`euler-${order}`,'Euler; Quaternion.setFromEuler; Euler.setFromQuaternion; Matrix4.makeRotationFromQuaternion',
    {axis_labelled_xyz_radians:n([e.x,e.y,e.z]),order},
    {quaternion_xyzw:n(q.toArray()),matrix_rows:n(rows(m)),matrix_native_column_major:n(m.elements),
     readback_xyz_radians:n([back.x,back.y,back.z]),
     basis_and_probe_images:n(corpus.vectors_xyz.map(v=>new T.Vector3(...v).applyQuaternion(q).toArray()))});
}
for(const pair of corpus.multiplication_pairs) {
  const a=new T.Quaternion(pair.a_wxyz[1],pair.a_wxyz[2],pair.a_wxyz[3],pair.a_wxyz[0]);
  const b=new T.Quaternion(pair.b_wxyz[1],pair.b_wxyz[2],pair.b_wxyz[3],pair.b_wxyz[0]);
  cap.add(pair.case_id,'multiply-and-premultiply','Quaternion.multiply; Quaternion.premultiply',
    {a_xyzw:n(a.toArray()),b_xyzw:n(b.toArray())},
    {ab_xyzw:n(a.clone().multiply(b).toArray()),ba_xyzw:n(a.clone().premultiply(b).toArray())});
}
const times=read(path.join(ROOT,'assets/sampling.json')).times_seconds;
for(const filename of fs.readdirSync(config.assets).filter(x=>/\.(gltf|glb)$/.test(x)).sort()) {
  const asset=path.join(config.assets,filename),raw=fs.readFileSync(asset);
  const data=filename.endsWith('.glb')?raw.buffer.slice(raw.byteOffset,raw.byteOffset+raw.byteLength):raw.toString('utf8');
  const gltf=await new GLTFLoader().parseAsync(data,'');
  const snapshot=()=>{
    gltf.scene.updateMatrixWorld(true);const objects=[];
    gltf.scene.traverse(o=>{if(o.name)objects.push({name:o.name,quaternion_xyzw:n(o.quaternion.toArray()),
      local_matrix_rows:n(rows(o.matrix)),world_matrix_rows:n(rows(o.matrixWorld))});});return objects;
  };
  cap.asset(asset);
  cap.add(filename,'gltf-static','GLTFLoader.parseAsync; Object3D.updateMatrixWorld',{asset:filename},{objects:snapshot()},contract({units:'metres',frame_direction:'node local to scene/world'}));
  for(const [index,clip] of gltf.animations.entries()) {
    const mixer=new T.AnimationMixer(gltf.scene),action=mixer.clipAction(clip);
    action.setLoop(T.LoopOnce,1);action.clampWhenFinished=true;action.play();
    for(const time of times) {
      mixer.setTime(time);
      cap.add(filename,`gltf-animation-${index}-t${time.toFixed(9)}`,'GLTFLoader animation tracks; AnimationMixer.setTime',
        {asset:filename,animation:index,time_seconds:n(time),loop:'LoopOnce',clamp_when_finished:true},
        {objects:snapshot()},contract({units:'metres, seconds',frame_direction:'node local to scene/world'}));
    }
    mixer.stopAllAction();mixer.uncacheRoot(gltf.scene);
  }
}
cap.finish(['EULER','ALGEBRA','ACTION','MATRIX','FORMAT','TRAJECTORY']);
