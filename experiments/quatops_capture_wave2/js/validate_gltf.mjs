/** Khronos validator report, separately identified from library schema checks. */
import fs from 'node:fs';
import path from 'node:path';
import validator from 'gltf-validator';
import { args, sha } from './capture_io.mjs';
const config=args();
if(fs.existsSync(config.output)&&fs.readdirSync(config.output).length)throw new Error('Refusing to overwrite');
fs.mkdirSync(config.output,{recursive:true});
let errors=0;
for(const name of fs.readdirSync(config.assets).filter(x=>/\.(gltf|glb)$/.test(x))) {
  const file=path.join(config.assets,name),bytes=new Uint8Array(fs.readFileSync(file));
  const report=await validator.validateBytes(bytes,{uri:name,maxIssues:1000});
  errors+=report.issues.numErrors;
  fs.writeFileSync(path.join(config.output,name+'.validation.json'),JSON.stringify({asset_sha256:sha(file),validator_version:validator.version(),report},null,2)+'\n');
}
console.log(JSON.stringify({errors}));process.exitCode=errors?1:0;
