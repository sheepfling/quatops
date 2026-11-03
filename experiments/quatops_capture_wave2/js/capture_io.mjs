/** Raw evidence I/O. No quaternion formulas or expected outputs. */
import fs from 'node:fs';
import path from 'node:path';
import os from 'node:os';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

export const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const sha = filename => crypto.createHash('sha256').update(fs.readFileSync(filename)).digest('hex');
export const read = filename => JSON.parse(fs.readFileSync(filename, 'utf8'));
export const contract = (extra = {}) => ({units:'radians',quaternion_order:'xyzw',matrix_action:'column vectors',frame_direction:'active rotation in declared numeric frame',...extra});
export function numeric(values, dtype='binary64') {
  const bits=[];
  const visit = x => {
    if (Array.isArray(x)) return x.map(visit);
    if (typeof x !== 'number') throw new TypeError('Numeric scalar required');
    const buffer=Buffer.alloc(dtype==='binary32'?4:8);
    if(dtype==='binary32') buffer.writeFloatBE(x); else buffer.writeDoubleBE(x);
    bits.push(buffer.toString('hex'));
    const native = dtype==='binary32'?buffer.readFloatBE():buffer.readDoubleBE();
    return Number.isFinite(native)?native:{nonfinite:Number.isNaN(native)?'nan':native>0?'+inf':'-inf'};
  };
  const shape = x => !Array.isArray(x)?[]:[x.length,...(x.length?shape(x[0]):[])];
  return {values:visit(values),dtype,shape:shape(values),ieee754_hex_be:bits};
}
export function args() {
  const out={corpus:path.join(ROOT,'corpus/orientation_inputs_v1.json'),assets:path.join(ROOT,'assets/gltf'),geographic:path.join(ROOT,'corpus/cesium_geographic_inputs.json')};
  for(let i=2;i<process.argv.length;i+=2) {
    const key=process.argv[i].replace(/^--/,'');
    if(!['corpus','assets','geographic','output'].includes(key)||!process.argv[i+1]) throw new Error('Expected --corpus/--assets/--geographic/--output PATH');
    out[key]=path.resolve(process.argv[i+1]);
  }
  if(!out.output) throw new Error('--output NEW_DIRECTORY is required');
  return out;
}
export function capture(options) {
  const {id,version,entry,config,limitations=[],family=id}=options;
  if(fs.existsSync(config.output)&&fs.readdirSync(config.output).length) throw new Error('Refusing to overwrite capture');
  const lock=path.join(ROOT,'js/package-lock.json');
  if(!fs.existsSync(lock)) throw new Error('Resolve dependencies and retain package-lock.json before capturing');
  fs.mkdirSync(config.output,{recursive:true});
  const observations=[]; const artifacts=[];
  const add=(caseId,operation,entryPoint,inputs,outputs,nativeContract=contract()) => observations.push({schema_version:'1.0',case_id:caseId,operation_id:operation,native_entry_point:entryPoint,status:'ok',contract:nativeContract,native_inputs:inputs,native_outputs:outputs,diagnostic:null});
  const asset=filename=>artifacts.push({path:filename,sha256:sha(filename),role:'input_asset'});
  const finish=profiles=>{
    const outfile=path.join(config.output,'observations.jsonl');
    fs.writeFileSync(outfile,observations.map(x=>JSON.stringify(x)).join('\n')+'\n');
    const npmGraphCommand = process.platform === 'win32' ? (process.env.ComSpec || 'cmd.exe') : 'npm';
    const npmGraphArgs = process.platform === 'win32'
      ? ['/d', '/s', '/c', 'npm ls --all --json']
      : ['ls', '--all', '--json'];
    const graph=execFileSync(npmGraphCommand,npmGraphArgs,{cwd:path.join(ROOT,'js'),encoding:'utf8'});
    fs.writeFileSync(path.join(config.output,'npm-resolved.json'),graph);
    const exporter=process.argv[1].startsWith('file:')?fileURLToPath(process.argv[1]):path.resolve(process.argv[1]);
    const manifest={schema_version:'1.0',capture_id:path.basename(config.output),oracle_id:id,oracle_family:family,evidence_kind:'native_runtime',
      product:{name:id,version:String(version),source_revision:null,binary_sha256:sha(entry),binary_identity_note:'Hash is resolved JavaScript entry; dependency graph and npm integrity are recorded separately',dependencies:[{name:'npm lock',sha256:sha(lock)},{name:'Node',version:process.version},{name:'V8',version:process.versions.v8}]},
      execution:{was_run:true,utc:new Date().toISOString(),argv:process.argv,exit_code:0,platform:os.platform(),architecture:os.arch(),runtime:process.version,compiler_flags:[]},
      precision:{input:'binary64 JavaScript Number; format buffers separately labelled',arithmetic:'JavaScript Number; no GPU',output:'binary64 API values; internal buffer types not inferred',serialized:'JSON plus IEEE bits'},
      exporter:{path:exporter,sha256:sha(exporter),commit:null},corpus:{path:config.corpus,sha256:sha(config.corpus)},
      artifacts:[{path:'observations.jsonl',sha256:sha(outfile),role:'raw_native_observations'},{path:'npm-resolved.json',sha256:sha(path.join(config.output,'npm-resolved.json')),role:'resolved_dependencies'},...artifacts],
      coverage:{profiles,independent_input_count:new Set(observations.map(x=>x.case_id)).size,native_observation_count:observations.length,quatops_replay_count:0},
      review:{state:'captured_not_reviewed',reviewer:null,limitations}};
    fs.writeFileSync(path.join(config.output,'manifest.json'),JSON.stringify(manifest,null,2)+'\n');
    console.log(JSON.stringify({output:config.output,observations:observations.length}));
  };
  return {add,asset,finish};
}
