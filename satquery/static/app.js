const form = document.querySelector('#query-form');
const mode = document.querySelector('#mode');
const filesInput = document.querySelector('#files');
const dropzone = document.querySelector('#dropzone');
const fileHint = document.querySelector('#file-hint');
const fileList = document.querySelector('#file-list');
const question = document.querySelector('#question');
const analyzeButton = document.querySelector('#analyze');
const views = {empty: document.querySelector('#empty-state'), loading: document.querySelector('#loading'), result: document.querySelector('#result'), error: document.querySelector('#error')};
let selectedFiles = [];
let previewUrls = [];

const expectedCount = () => ['temporal_pair', 'optical_sar_pair'].includes(mode.value) ? 2 : 1;
function updateHint() {
  const labels = {single_optical:'one optical image required',single_sar:'one SAR image required',temporal_pair:'two images: before, then after',optical_sar_pair:'two images: optical, then SAR'};
  fileHint.textContent = `PNG or JPEG · ${labels[mode.value]}`;
  selectedFiles = selectedFiles.slice(0, expectedCount());
  renderFiles();
}
function renderFiles() {
  fileList.innerHTML = selectedFiles.map((file, i) => `<div class="file-item"><span>${i + 1}. ${escapeHtml(file.name)}</span><b>${(file.size/1024).toFixed(0)} KB</b></div>`).join('');
}
function escapeHtml(value) { const div=document.createElement('div'); div.textContent=value; return div.innerHTML; }
function acceptFiles(list) {
  const images = [...list].filter(file => ['image/png','image/jpeg'].includes(file.type));
  const room = expectedCount() - selectedFiles.length;
  selectedFiles = [...selectedFiles, ...images.slice(0, room)];
  filesInput.value = '';
  renderFiles();
}
dropzone.addEventListener('click', () => filesInput.click());
dropzone.addEventListener('keydown', e => {if (e.key==='Enter'||e.key===' ') filesInput.click();});
filesInput.addEventListener('change', () => acceptFiles(filesInput.files));
['dragenter','dragover'].forEach(name => dropzone.addEventListener(name, e => {e.preventDefault();dropzone.classList.add('drag');}));
['dragleave','drop'].forEach(name => dropzone.addEventListener(name, e => {e.preventDefault();dropzone.classList.remove('drag');}));
dropzone.addEventListener('drop', e => acceptFiles(e.dataTransfer.files));
mode.addEventListener('change', updateHint);
document.querySelectorAll('[data-query]').forEach(button => button.addEventListener('click', () => {
  question.value=button.dataset.query;
  if(button.dataset.mode){mode.value=button.dataset.mode;updateHint();}
  question.focus();
}));
function show(name){Object.entries(views).forEach(([key,el])=>el.classList.toggle('hidden',key!==name));}
function toDataUrl(file){return new Promise((resolve,reject)=>{const reader=new FileReader();reader.onload=()=>resolve(reader.result);reader.onerror=reject;reader.readAsDataURL(file);});}
function title(value){return value.replaceAll('_',' ').replace(/\b\w/g,c=>c.toUpperCase());}
function flattenMetrics(obj,prefix=''){const rows=[];for(const [key,value] of Object.entries(obj||{})){const label=prefix?`${prefix} · ${key}`:key;if(value&&typeof value==='object'&&!Array.isArray(value))rows.push(...flattenMetrics(value,label));else rows.push([label,String(value)]);}return rows;}
function renderResult(data){
  document.querySelector('#task-name').textContent=title(data.task);
  document.querySelector('#answer-text').textContent=data.answer;
  const badge=document.querySelector('#confidence-badge'); const label=data.confidence?.label||'not scored';
  badge.textContent=data.confidence?`${label} · ${Math.round(data.confidence.score*100)}%`:'not scored'; badge.className=`confidence ${label}`;
  const evidence=document.querySelector('#evidence'); const noEvidence=document.querySelector('#no-evidence');
  if(data.evidence_image){evidence.src=data.evidence_image;evidence.classList.add('visible');noEvidence.classList.add('hidden');}else{evidence.removeAttribute('src');evidence.classList.remove('visible');noEvidence.classList.remove('hidden');}
  const original=document.querySelector('#original-image');
  const secondOriginal=document.querySelector('#second-original-image');
  const secondFigure=document.querySelector('#second-original-figure');
  const comparison=document.querySelector('#evidence-comparison');
  previewUrls.forEach(url=>URL.revokeObjectURL(url));
  previewUrls=selectedFiles.map(file=>URL.createObjectURL(file));
  original.src=previewUrls[0]||'';
  if(previewUrls.length>1){
    secondOriginal.src=previewUrls[1];
    secondFigure.classList.remove('hidden');
    document.querySelector('#original-figure figcaption').textContent=mode.value==='temporal_pair'?'Before image':'Optical image';
    document.querySelector('#second-original-label').textContent=mode.value==='temporal_pair'?'After image':'SAR image';
  }else{
    secondOriginal.removeAttribute('src');
    secondFigure.classList.add('hidden');
    document.querySelector('#original-figure figcaption').textContent='Original image';
  }
  comparison.classList.toggle('three-up',previewUrls.length>1);
  const metrics={...data.metrics};if(data.input_images)metrics.detected_inputs=Object.fromEntries(data.input_images.map((item,i)=>[`image_${i+1}`,`${item.label} (${Math.round(item.confidence*100)}%)`]));if(data.confidence)metrics.confidence_factors=data.confidence.factors;metrics.latency_ms=data.latency_ms;
  document.querySelector('#metrics').innerHTML=flattenMetrics(metrics).map(([k,v])=>`<div class="metric-row"><span>${escapeHtml(title(k))}</span><b>${escapeHtml(v)}</b></div>`).join('')||'<p>No measurements.</p>';
  document.querySelector('#trace').innerHTML=data.trace.map(item=>`<li><b>${escapeHtml(title(item.step))}</b><br>${escapeHtml(item.detail)}</li>`).join('');
  document.querySelector('#warnings').innerHTML=data.warnings.map(item=>`<li>${escapeHtml(item)}</li>`).join('');
  document.querySelector('#warnings-box').classList.toggle('hidden',!data.warnings.length);
}
form.addEventListener('submit',async e=>{
  e.preventDefault(); document.querySelector('#error-text').textContent='';
  if(selectedFiles.length!==expectedCount()){document.querySelector('#error-text').textContent=`Please select exactly ${expectedCount()} image(s) for this mode.`;show('error');return;}
  analyzeButton.disabled=true;show('loading');
  try{
    const images=await Promise.all(selectedFiles.map(async file=>({name:file.name,data:await toDataUrl(file)})));
    const response=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:question.value,mode:mode.value,images})});
    const data=await response.json();if(!response.ok)throw new Error(data.error||'Analysis failed.');renderResult(data);show('result');
  }catch(error){document.querySelector('#error-text').textContent=error.message;show('error');}finally{analyzeButton.disabled=false;}
});
updateHint();

fetch('/api/health').then(response=>response.json()).then(data=>{
  const scene=data.scene_model||{};
  document.querySelector('#service-status').textContent=scene.available?'Local model ready':'Local baselines ready';
  document.querySelector('#model-summary').textContent=scene.available
    ? `Scene model: ${scene.training_images} training images · ${scene.classes.join(', ')} · protocol: ${scene.training_protocol.replaceAll('_',' ')}`
    : 'Scene classifier is not trained. Grounding, change and fusion baselines remain available.';
}).catch(()=>{document.querySelector('#service-status').textContent='Local server';});
