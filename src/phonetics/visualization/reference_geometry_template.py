"""Standalone interactive reference geometry report template."""

PAGE = r"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mandarin syllable distance maps</title>
<style>
*{box-sizing:border-box}body{font:15px Arial,sans-serif;color:#181818;margin:24px}h1{font-size:23px}h2{font-size:17px;margin:10px 0}.toolbar{display:flex;gap:10px;align-items:center;flex-wrap:wrap}button{font:inherit;padding:7px 12px;background:white;border:1px solid #aaa;border-radius:6px;cursor:pointer}button[aria-pressed=true]{background:#222;color:white}.scroll{overflow-x:auto}.panels{display:grid;grid-template-columns:repeat(3,minmax(250px,1fr));gap:14px;min-width:780px}canvas{display:block;width:100%;aspect-ratio:1;border:1px solid #aaa;touch-action:none;cursor:grab;image-rendering:pixelated}canvas:active{cursor:grabbing}.legend{height:9px;background:linear-gradient(90deg,white,black);border:1px solid #bbb;margin-top:8px}.limits{display:flex;justify-content:space-between;font-size:12px}.hint,#readout{color:#444}#readout{min-height:2em;font-variant-numeric:tabular-nums}.extent{font-size:12px;color:#444;min-height:30px}
table{border-collapse:collapse;margin:8px 0 18px;font-variant-numeric:tabular-nums}th,td{padding:6px 12px;border-bottom:1px solid #ddd;text-align:right}th:first-child{text-align:left}caption{text-align:left;font-weight:bold}#stats-note{color:#444;font-size:13px}
:root{--surface:#fff;--ink:#182631;--muted:#51606d;--line:#dbe2e8;--accent:#234b6b;--soft:#edf3f8}
body{margin:0 auto;padding:30px clamp(14px,3vw,44px) 40px;max-width:1740px;background:#f5f7f9;color:var(--ink);line-height:1.5}
.report-head{padding:22px 26px;background:var(--surface);border:1px solid var(--line);border-radius:14px}
.eyebrow{margin:0;color:var(--accent);font-size:12px;font-weight:700;letter-spacing:.11em}
h1{font-size:clamp(24px,3vw,34px);margin:5px 0 10px;line-height:1.15}
h2{font-size:17px;color:var(--ink)}
.badges{display:flex;flex-wrap:wrap;gap:8px}
.badges span,#zoom{padding:3px 9px;border-radius:20px;background:var(--soft);font-size:12px;color:var(--accent);font-variant-numeric:tabular-nums}
.hint,#stats-note{color:var(--muted);max-width:105ch}
.toolbar{gap:7px;margin:12px 0;padding:9px 11px;border:1px solid var(--line);border-radius:11px;background:var(--surface)}
.toolbar>span:first-child{font-size:12px;color:var(--muted);font-weight:700}
button,select{font:inherit;font-size:13px;color:var(--accent);background:#f8fafc;border:1px solid #cbd7e0;border-radius:7px;padding:7px 11px;min-height:34px}
button:hover,select:hover{background:var(--soft);border-color:#95aabd}
button[aria-pressed=true]{background:var(--accent);border-color:var(--accent);color:white}
:focus-visible{outline:3px solid #79b6e5;outline-offset:2px}
button:disabled{opacity:.55;cursor:default}
fieldset{border:1px solid var(--line)!important;background:var(--surface);border-radius:9px;padding:7px 13px 10px}
fieldset legend{color:var(--muted);font-weight:700;font-size:12px;padding:0 4px}
fieldset label{display:inline-flex;gap:5px;align-items:center;font-size:13px}
input[type=checkbox]{accent-color:var(--accent);width:15px;height:15px}
.panels{gap:12px}.panels>section{background:var(--surface);border:1px solid var(--line);border-radius:11px;padding:12px}
.panels>section h2{margin:0 0 7px}canvas{border-color:#cbd7e0;touch-action:none}
#readout{background:var(--soft);border-radius:8px;padding:8px 12px;min-height:40px;overflow-wrap:anywhere}
#map-results>section,#profile-results,details{background:var(--surface);border:1px solid var(--line);border-radius:11px;padding:12px 17px;margin-top:12px}
#map-results>section{overflow-x:auto}
#profile-results>h2:not(:first-child){margin-top:22px}
table{max-width:100%;width:100%;font-size:13px}thead th{background:#edf3f8;color:var(--accent)}
tbody tr:nth-child(even){background:#f8fafc}summary{color:var(--accent);font-weight:700;cursor:pointer}
#agreement-filter{margin:10px 0 14px} [hidden]{display:none!important}
@media(max-width:700px){body{padding:12px}.report-head{padding:15px}.toolbar{gap:5px}button{padding:6px 8px}}
</style>
<header class="report-head"><p class="eyebrow">REFERENCE GEOMETRY</p><h1>Mandarin syllable distance maps</h1>
<div class="badges"><span>@@syllables@@ syllables</span><span>@@pairs@@ unique pairs</span><span>3 systems</span></div></header>
<p class="hint">This exploratory analysis describes representation geometry using a specified component-matching criterion and DIMSIM as a computational reference. Component comparisons are selected from syllable labels independently of system distances. Scores measure structural consistency and reference agreement.</p>
<div class="toolbar"><span>Analysis</span><button id="hier" aria-pressed="true">Component structure</button><button id="rank" aria-pressed="false">DIMSIM neighborhood</button><button id="agreement" aria-pressed="false">Agreement with DIMSIM</button><button id="mask-profile" aria-pressed="false">Mask profile</button><button id="minus" aria-label="Zoom out">−</button><button id="plus" aria-label="Zoom in">+</button><button id="reset">Reset view</button><span id="zoom"></span></div>
<p class="hint">Scroll to zoom, drag to pan, hover for distances. All panels share the view. Rows are anchors. In hierarchical order, checked components must all match. Other cells, including self-pairs, are shown in light gray. Highlighted distances use the 2nd–98th percentile color range when it has nonzero width.</p>
<fieldset id="filters" style="margin:0 0 14px;border:1px solid #bbb;display:flex;gap:16px;flex-wrap:wrap"><legend>Highlight matching pairs</legend><label><input type="checkbox"> Same initial</label><label><input type="checkbox"> Same final</label><label><input type="checkbox"> Same tone</label></fieldset><fieldset id="win-tests" hidden style="margin:0 0 14px;border:1px solid #bbb;gap:16px;flex-wrap:wrap"><legend>Components allowed to change</legend><label><input type="checkbox" value="0"> Initial</label> <label><input type="checkbox" value="1" checked> Final</label> <label><input type="checkbox" value="2"> Tone</label></fieldset><fieldset id="transition-groups" hidden style="margin:0 0 14px;border:1px solid #bbb;gap:16px;flex-wrap:wrap"><legend>Transition groups</legend><label><input type="checkbox" checked> One component</label><label><input type="checkbox"> Two components</label><label><input type="checkbox"> Full match (111)</label></fieldset><div class="toolbar" id="color-mode" style="margin-bottom:12px"><span>Cell color</span><button id="distance-color" aria-pressed="true">Distance</button><button id="final-color" aria-pressed="false">Mean effect</button><button id="win-color" aria-pressed="false">Win rate</button></div><div id="agreement-filter" hidden><p class="hint">Agreement compares model and DIMSIM order on the selected triples. DIMSIM supplies the reference direction; ties in DIMSIM are excluded. Controlled match selects triples using component labels, independently of DIMSIM values. These scores describe agreement with this reference.</p><label>Comparisons <select id="agreement-slice"><option value="same">Same mask</option><option value="trade">Trade-off</option><option value="controlled">Controlled match</option></select></label></div><div id="map-results"><div class="scroll"><div id="panels" class="panels"></div></div><p id="readout" aria-live="off"></p><section aria-label="Summary statistics"><p id="stats-note"></p><table id="stats"></table><div id="anchor-distribution" hidden></div></section>
<p class="hint" id="effect-note" hidden>Blue indicates a positive effect or a majority of comparisons satisfying the specified component-matching criterion. Orange indicates a negative effect or a win rate below 50%. White marks zero mean difference or 50% win rate. Gray marks cells outside the comparison. Mean effect colors clip at the 98th percentile of absolute effects. Ties count as half a win. Each transition requires all its changing components to be checked. Each comparison is displayed in the candidate cell predicted to be nearer by the component-matching criterion. Each cell shows its lowest available mean effect or win rate. Self-distances are zero for the current matrices. Full-match comparisons are reported separately by transition. The criterion is a declared structural hypothesis; additivity is not required.</p></div><section id="profile-results" hidden><h2>Mean similarity by overlap mask</h2><p class="hint">Masks are ordered by mean negative DIMSIM distance for display. Mask means, centered R², J and residuals describe each system independently. The two scales use independent units.</p><div class="toolbar" id="profile-legend"></div><div id="profile-chart"></div><h2>Explained variance</h2><table id="profile-r2"></table><h2>Component gains and the final step</h2><p class="hint">Each gain is a descriptive difference between means of different pair populations. Absolute J is the final gain minus the mean of the three ordinary gains. Mask 111 uses self-pairs, with cosine 1 and DIMSIM distance 0.</p><label>Model <select id="profile-step-model"><option value="0">DIMSIM</option><option value="1" selected>wav2vec2 zh-cn</option><option value="2">wav2vec2 base</option></select></label><div id="profile-steps"></div><h2>Top positive residual pairs</h2><p class="hint">Top 50 distinct unordered syllable pairs per system. Residual equals similarity minus its overlap-mask mean. Similarity is negative distance for DIMSIM and cosine for wav2vec2. Ranks are independent across columns; residuals use different units.</p><div class="scroll"><table id="profile-residuals"></table></div></section><details><summary>Coverage and report files</summary><div class='scroll'><table><thead><tr><th>Population</th><th>Count</th></tr></thead><tbody>@@coverage@@</tbody></table></div><p><a href='manifest.json'>Manifest</a> · <a href='tables/controlled_match.csv'>Controlled match</a> · <a href='tables/dimsim_agreement.csv'>Agreement</a> · <a href='tables/mask_profiles.csv'>Mask profiles</a> · <a href='figures/dimsim-distances.png'>DIMSIM distances</a> · <a href='figures/base-cosine-distances.png'>Base cosine distances</a> · <a href='figures/zh-cn-cosine-distances.png'>Zh-cn cosine distances</a></p></details>
<details><summary>Protocol and sources</summary><p>Evaluation input <code>@@evaluation_id@@</code>. The inventory uses the shared test split with strict nonempty initials and @@excluded@@ configured exclusions. The report uses the original DIMSIM reference and cosine distances from matching audio for both wav2vec2 systems. Comparison tolerance is @@tolerance@@.</p><p>Component matching predicts that replacing selected components with the anchor values, while holding all other candidate components fixed, does not increase dissimilarity. This structural criterion motivates controlled comparisons; it is not an empirically established universal law of Mandarin perception. One-step, two-step and full-match transitions remain separately selectable. Main summaries weight each unique triple equally. Anchor histograms summarize within-anchor win rates.</p><p>Sources: <a href="https://doi.org/10.1037/0033-295X.84.4.327">Tversky (1977), Features of Similarity</a> motivates feature-based monotonicity. <a href="https://doi.org/10.1080/01690960701728261">Tong, Francis &amp; Gandour (2008)</a> documents interactions among Mandarin segmental and tonal processing. <a href="https://aclanthology.org/K18-1043/">Li et al. (2018), DIMSIM</a> supplies the computational reference.</p><p>The exploratory protocol was developed while inspecting these wav2vec2 systems. Interpretation is conditional on the selected inventory, stored representations and reference implementation.</p></details><p class="hint">@@syllables@@ shared tone-marked syllables after applying the evaluation inventory filter. In Distance mode, lighter cells indicate smaller distances. The two wav2vec2 panels share a cosine distance scale.</p>
<script>

const residualTopPairs=@@residualTopPairs@@;

const profileData=@@profileData@@;

const data=@@data@@;

const decode=(s,T)=>{
 const binary=atob(s), bytes=new Uint8Array(binary.length);
 for(let i=0;i<binary.length;i++) bytes[i]=binary.charCodeAt(i);
 return new T(bytes.buffer);
};
const prepared=@@prepared@@;

const preparedCells=decode(prepared.cells,Uint16Array), preparedEffects=decode(prepared.effects,Float64Array);
const preparedWins=decode(prepared.wins,Uint16Array), preparedAgreement=decode(prepared.agree,Uint16Array);

const labels=data.labels,N=labels.length,order=decode(data.rank,Uint16Array),mat=data.maps.map(s=>decode(s,Float32Array));
const components=@@components@@;

const filters=document.querySelector("#filters");
const boxes=Array.from(filters.querySelectorAll("input"));
const rankStats=@@rankStats@@;

const rankGroupSize=@@rankGroupSize@@;

const maxima=[data.maxima[0],data.maxima[1],data.maxima[1]];
let effect=false, winMode=false, effectData=null;
const effectCache=new Map();
let profiling=false;
let agreeing=false, agreementMaps=null, agreementSupport=null;
const agreementCache=new Map();
let ranked=false,x=0,y=0,span=N,drag=null,hover=null;
const panels=document.querySelector('#panels'),readout=document.querySelector('#readout');
const views=['DIMSIM','wav2vec2 zh-cn','wav2vec2 base'].map((name,k)=>{
 const div=document.createElement('section');
 div.innerHTML='<h2>'+name+'</h2><canvas aria-label="'+name+' distance heatmap"></canvas><div class="extent"></div><div class="legend"></div><div class="limits"><span>0</span><span>'+ (k?'Cosine distance':'DIMSIM distance')+'</span><span>'+maxima[k].toFixed(3)+'</span></div>';
 panels.append(div);
 const source=document.createElement('canvas');source.width=source.height=N;
 return {panel:div,canvas:div.querySelector('canvas'),extent:div.querySelector('.extent'),limits:div.querySelector('.limits'),legend:div.querySelector('.legend'),source};
});
function candidate(row,col){return ranked?order[row*N+col]:col}
const winTests=Array.from(document.querySelector('#win-tests').querySelectorAll('input'));
const groupBox=document.querySelector('#transition-groups');
const groupTests=Array.from(groupBox.querySelectorAll('input'));
const changedComponents=mask=>components.map((_,k)=>k).filter(k=>mask&(1<<k));
function buildEffects(mask){
 const delta=mat.map(()=>new Float64Array(N*N));
 const counts=new Uint16Array(N*N);
 const wins=mat.map(()=>new Float64Array(N*N));
 const ties=mat.map(()=>new Uint16Array(N*N));
 for(let i=0;i<preparedCells.length/5;i++){
  const off=i*5;
  if(preparedCells[off]!==mask)continue;
  const cell=preparedCells[off+1]*N+preparedCells[off+2];
  counts[cell]=preparedCells[off+3];
  for(let k=0;k<3;k++){
   delta[k][cell]=preparedEffects[i*3+k];
   wins[k][cell]=preparedWins[i*3+k]/2;
  }
 }
 return {delta,counts,wins,ties};
}
function getEffect(mask){
 if(!effectCache.has(mask))effectCache.set(mask,buildEffects(mask));
 return effectCache.get(mask);
}
function eligibleCell(mask,j){
 if(j%(N+1)===0)return groupTests[2].checked;
 const size=changedComponents(mask).length;
 return size<3&&groupTests[size-1].checked;
}
function combinedEffectData(){
 const allowed=winTests.reduce((m,box,k)=>box.checked?m|(1<<k):m,0);
 const tests=[];
 for(let mask=1;mask<8;mask++){
  const size=changedComponents(mask).length;
  if((mask&allowed)===mask&&(groupTests[2].checked||(size<3&&groupTests[size-1].checked)))
   tests.push([mask,getEffect(mask)]);
 }
 const counts=new Uint16Array(N*N),testCounts=new Uint8Array(N*N);
 const scores=mat.map(()=>new Float64Array(N*N).fill(1));
 const delta=mat.map(()=>new Float32Array(N*N).fill(Infinity));
 const perfect=mat.map(()=>new Uint8Array(N*N).fill(1));
 for(const [mask,data] of tests)for(let j=0;j<counts.length;j++){
  const n=data.counts[j];if(!n||!eligibleCell(mask,j))continue;
  counts[j]+=n;testCounts[j]++;
  for(let k=0;k<mat.length;k++){
   delta[k][j]=Math.min(delta[k][j],data.delta[k][j]);
   scores[k][j]=Math.min(scores[k][j],(data.wins[k][j]+.5*data.ties[k][j])/n);
   if(data.wins[k][j]!==n)perfect[k][j]=0;
  }
 }
 const samples=[[],[]];
 if(!winMode)for(let j=0;j<counts.length;j++)if(counts[j])
  delta.forEach((m,k)=>samples[k===0?0:1].push(Math.abs(m[j])));
 const limits=samples.map(v=>{v.sort((a,b)=>a-b);return v[Math.ceil(.98*(v.length-1))]||1;});
 return {counts,testCounts,scores,perfect,tests,delta,limits};
}
function rebuildEffect(){
 effectData=combinedEffectData();
 const {delta,counts,scores,limits}=effectData;
 views.forEach(({source,legend,limits:labels},k)=>{
  const cap=limits[k===0?0:1],ctx=source.getContext('2d'),img=ctx.createImageData(N,N);
  legend.style.background='linear-gradient(90deg,#bc4b16,#fff,#175ca5)';
  labels.firstElementChild.textContent=winMode?'0%':(-cap).toPrecision(4);
  labels.children[1].textContent=winMode?'Criterion win rate · midpoint 50%':k?'Cosine distance difference':'DIMSIM distance difference';
  labels.lastElementChild.textContent=winMode?'100%':'+'+cap.toPrecision(4);
  for(let j=0;j<counts.length;j++){
   const signed=counts[j]?(winMode?2*scores[k][j]-1:delta[k][j]/cap):0;
   const rgb=counts[j]?(signed>=0?[23,92,165]:[188,75,22]):[226,226,226];
   const t=counts[j]?Math.min(1,Math.abs(signed)):1;
   for(let c=0;c<3;c++)img.data[4*j+c]=255+(rgb[c]-255)*t;
   img.data[4*j+3]=255;
  }
  ctx.putImageData(img,0,0);
 });
 draw();
}
function renderAnchorDistribution(wins,counts,names){
 const target=document.querySelector('#anchor-distribution');target.hidden=false;
 const active=Array.from(counts.keys()).filter(a=>counts[a]>0);
 if(!active.length){target.innerHTML='<p>No eligible comparisons.</p>';return;}
 const values=wins.map(w=>active.map(a=>w[a]/counts[a]));
 const bins=values.map(v=>{
  const b=Array(10).fill(0);for(const x of v)b[Math.min(9,Math.floor(x*10))]++;return b;
 });
 const max=Math.max(1,...bins.flat()),colors=['#333','#175ca5','#bc4b16'];
 const text=(x,y,t,anchor='middle')=>'<text x="'+x+'" y="'+y+'" text-anchor="'+anchor+'" font-size="12">'+t+'</text>';
 const bars=bins.map((b,k)=>b.map((n,j)=>{
  const x=65+j*65+k*(48/names.length),height=150*n/max;
  return '<rect x="'+x+'" y="'+(180-height)+'" width="'+(46/names.length)+'" height="'+height+'" fill="'+colors[names.length===2?k+1:k]+'"><title>'+names[k]+' · '+(j*10)+'–'+((j+1)*10)+'% · '+n+' anchors</title></rect>';
 }).join('')).join('');
 const percentile=(sorted,q)=>{const t=(sorted.length-1)*q;return sorted[Math.floor(t)]+(sorted[Math.ceil(t)]-sorted[Math.floor(t)])*(t-Math.floor(t));};
 target.innerHTML='<h3>Win rate distribution across anchors</h3><p class="hint">Each anchor contributes one value: successful comparisons divided by its eligible comparisons, with half credit for model ties. The histogram gives anchors equal weight. The main summary gives triples equal weight.</p><p>'+names.map((name,k)=>'<span style="color:'+colors[names.length===2?k+1:k]+'">'+name+'</span>').join(' · ')+'</p><svg role="img" aria-label="Anchor win rate histogram" viewBox="0 0 760 225" style="max-width:900px;width:100%">'+text(15,20,'Anchors','start')+text(45,35,max,'end')+'<line x1="60" x2="720" y1="180" y2="180" stroke="#777"/>'+bars+Array.from({length:11},(_,j)=>text(60+j*65,200,j*10+'%')).join('')+'</svg><table><thead><tr><th>Anchor statistic</th>'+names.map(n=>'<th>'+n+'</th>').join('')+'</tr></thead><tbody>'+[[.25,'25th percentile'],[.5,'Median'],[.75,'75th percentile']].map(([q,label])=>'<tr><th scope="row">'+label+'</th>'+values.map(v=>'<td>'+(100*percentile([...v].sort((a,b)=>a-b),q)).toFixed(1)+'%</td>').join('')+'</tr>').join('')+'</tbody></table>';
}
function updateStats(active=[]){
 const table=document.querySelector('#stats'),note=document.querySelector('#stats-note');
 const rows=[];
 document.querySelector('#anchor-distribution').hidden=true;
 const pct=x=>(100*x).toFixed(1)+'%';
 const num=x=>Number(x.toPrecision(4)).toString();
 const count=x=>x.toLocaleString('en-US');
 const quantile=(a,q)=>{
  const k=(a.length-1)*q,j=Math.floor(k);
  return a[j]+(a[Math.ceil(k)]-a[j])*(k-j);
 };
 if(ranked){
  note.textContent='All anchors · self-pairs excluded · lexical tie-break · near and far groups contain '+rankGroupSize+' candidates each (10% per tail, rounded down).';
  rows.push(['Mean top-'+Math.min(10,N-1)+' overlap with DIMSIM',...rankStats[0].map(v=>v===null?'—':pct(v))],
   ['Global rank agreement with DIMSIM (Spearman)',...rankStats[1].map(v=>v===null?'—':num(v))],
   ['Near candidate wins against far candidate',...rankStats[2].map(v=>v===null?'—':pct(v))]);
 }else if(effect){
  const totalWins=mat.map(()=>new Float64Array(N)),totalEffects=mat.map(()=>new Float64Array(N));
  const anchorCounts=new Float64Array(N),byTransition=new Map();
  for(const [bits,data] of effectData.tests){
   for(let j=0;j<data.counts.length;j++){
    const n=data.counts[j];if(!n||!eligibleCell(bits,j))continue;
    const a=Math.floor(j/N),b=j%N,high=overlapCode(a,b);
    const key=maskLabel(high&~bits)+' → '+maskLabel(high);
    if(!byTransition.has(key))byTransition.set(key,{n:0,w:mat.map(()=>0),d:mat.map(()=>0)});
    const entry=byTransition.get(key);entry.n+=n;anchorCounts[a]+=n;
    mat.forEach((_,k)=>{
     const win=data.wins[k][j]+.5*data.ties[k][j],value=n*data.delta[k][j];
     entry.w[k]+=win;entry.d[k]+=value;totalWins[k][a]+=win;totalEffects[k][a]+=value;
    });
   }
  }
  const triples=anchorCounts.reduce((a,b)=>a+b,0);
  const cells=effectData.counts.filter(n=>n>0).length;
  note.textContent=count(triples)+' unique controlled triples · '+count(anchorCounts.filter(n=>n>0).length)+' anchors · '+count(cells)+' nearer map cells. Each triple has equal weight in the summary. Map colors retain the minimum across available transitions. Expected direction is defined by the component-matching criterion, independently of DIMSIM.';
  const values=mat.map((_,k)=>[
   triples?pct(totalWins[k].reduce((a,b)=>a+b,0)/triples):'—',
   triples?num(totalEffects[k].reduce((a,b)=>a+b,0)/triples):'—'
  ]);
  ['Component-matching win rate · equal triple weight','Overall mean effect · equal triple weight'].forEach((label,j)=>rows.push([label,...values.map(v=>v[j])]));
  for(const [key,entry] of [...byTransition].sort()){
   rows.push([key+' · '+count(entry.n)+' triples · win rate',...entry.w.map(v=>pct(v/entry.n))]);
   rows.push([key+' · mean effect',...entry.d.map(v=>num(v/entry.n))]);
  }
  renderAnchorDistribution(totalWins,anchorCounts,['DIMSIM','wav2vec2 zh-cn','wav2vec2 base']);
 }else{
  note.textContent='Distinct unordered pairs · self-pairs excluded · statistics cover the full slice.';
  const groups=[[],[]];
  for(let a=0;a<N;a++)for(let b=a+1;b<N;b++)groups[active.every(c=>c[a]===c[b])?0:1].push(a*N+b);
  groups.forEach((g,k)=>{
   if(k&&!active.length)return;
   const label=active.length?(k?'Other':'Highlighted'):'All';
   rows.push([label+' pairs',...mat.map(()=>count(g.length))]);
   rows.push([label+' median [25th, 75th percentile]',...mat.map(m=>{
    if(!g.length)return '—';
    const v=Float32Array.from(g,j=>m[j]);v.sort();
    return num(quantile(v,.5))+' ['+num(quantile(v,.25))+', '+num(quantile(v,.75))+']';
   })]);
  });
 }
 table.innerHTML='<caption>Summary statistics</caption><thead><tr><th scope="col">Measure</th><th scope="col">DIMSIM</th><th scope="col">wav2vec2 zh-cn</th><th scope="col">wav2vec2 base</th></tr></thead><tbody>'+
 rows.map(([label,...v])=>'<tr><th scope="row">'+label+'</th>'+v.map(x=>'<td>'+x+'</td>').join('')+'</tr>').join('')+'</tbody>';
}
function calculateAgreement(slice){
 let support, points, ties=0;
 if(slice==='controlled'){
  support=new Uint16Array(N*N);points=[new Uint16Array(N*N),new Uint16Array(N*N)];
  const allowed=winTests.reduce((bits,box,k)=>box.checked?bits|(1<<k):bits,0);
  for(let i=0;i<preparedCells.length/5;i++){
   const off=i*5, mask=preparedCells[off], cell=preparedCells[off+1]*N+preparedCells[off+2];
   if((allowed&mask)!==mask||!eligibleCell(mask,cell))continue;
   const count=preparedCells[off+4];
   support[cell]+=count;ties+=preparedCells[off+3]-count;
   for(let k=0;k<2;k++)points[k][cell]+=preparedAgreement[i*2+k];
  }
 }else{
  const index=slice==='trade'?0:1;
  support=decode(prepared.support[index],Uint16Array);
  points=prepared.points[index].map(value=>decode(value,Uint16Array));
  ties=prepared.ties[index];
 }
 const copies=slice==='controlled'?1:2, totals=[0,0];
 const anchorCounts=new Float64Array(N),anchorWins=[new Float64Array(N),new Float64Array(N)];
 for(let cell=0;cell<support.length;cell++){
  const a=Math.floor(cell/N);
  anchorCounts[a]+=support[cell]/copies;
  for(let k=0;k<2;k++){
   totals[k]+=points[k][cell]/copies;
   anchorWins[k][a]+=points[k][cell]/(2*copies);
  }
 }
 const maps=points.map(scores=>Float32Array.from(scores,(v,j)=>support[j]?v/(2*support[j]):NaN));
 return {maps,support,comparisons:anchorCounts.reduce((a,b)=>a+b,0),ties,totals,
         anchorCount:anchorCounts.filter(n=>n>0).length,anchorCounts,anchorWins};
}
function rebuildAgreement(){
 const slice=document.querySelector('#agreement-slice').value;
 if(slice!=='controlled'&&!agreementCache.has(slice))agreementCache.set(slice,calculateAgreement(slice));
 const {maps,support,comparisons,ties,totals,anchorCount,anchorCounts,anchorWins}=slice==='controlled'?calculateAgreement(slice):agreementCache.get(slice);
 agreementMaps=maps;agreementSupport=support;
 views.slice(1).forEach(({source,legend,limits},k)=>{
  const ctx=source.getContext('2d'),img=ctx.createImageData(N,N);
  legend.style.background='linear-gradient(90deg,#bc4b16,#fff,#175ca5)';
  limits.firstElementChild.textContent='0%';limits.children[1].textContent='Order agreement · midpoint 50%';limits.lastElementChild.textContent='100%';
  agreementMaps[k].forEach((v,j)=>{
   const rgb=!Number.isFinite(v)?[226,226,226]:v>=.5?[23,92,165]:[188,75,22];
   const t=Number.isFinite(v)?Math.abs(2*v-1):1;
   for(let c=0;c<3;c++)img.data[4*j+c]=255+(rgb[c]-255)*t;
   img.data[4*j+3]=255;
  });
  ctx.putImageData(img,0,0);
 });
 const fmt=n=>n.toLocaleString('en-US');
 const cells=support.reduce((sum,n)=>sum+(n>0),0);
 const label={same:'Same mask',trade:'Trade-off',controlled:'Controlled match'}[slice];
 document.querySelector('#stats-note').textContent=fmt(anchorCount)+' anchors · '+fmt(cells)+' map cells · '+fmt(ties)+' DIMSIM ties excluded. Each comparison counts once in the table. '+(slice==='controlled'?'The map shows the candidate predicted to be nearer by the component-matching criterion.':'The map shows both candidate cells.')+' wav2vec2 ties count as half agreement.';
 document.querySelector('#stats').innerHTML='<caption>Agreement with DIMSIM</caption><thead><tr><th scope="col">Comparisons</th><th scope="col">Unique triples</th><th scope="col">wav2vec2 zh-cn</th><th scope="col">wav2vec2 base</th></tr></thead><tbody><tr><th scope="row">'+label+'</th><td>'+fmt(comparisons)+'</td>'+totals.map(v=>'<td>'+(comparisons?(50*v/comparisons).toFixed(1)+'%':'—')+'</td>').join('')+'</tr></tbody>';
 renderAnchorDistribution(anchorWins,anchorCounts,['wav2vec2 zh-cn','wav2vec2 base']);
 draw();
}
function overlapCode(a,b){return components.reduce((m,v,k)=>m|(v[a]===v[b]?1<<k:0),0);}
const maskLabel=bits=>components.map((_,k)=>bits&(1<<k)?'1':'0').join('');
function rebuild(){
 document.querySelector('#anchor-distribution').hidden=true;
 if(profiling){renderProfile();return;}
 if(agreeing){rebuildAgreement();return;}
 if(effect&&!ranked){rebuildEffect();updateStats();return;}
 const active=ranked?[]:components.filter((_,k)=>boxes[k].checked);
 const mask=new Uint8Array(N*N);
 let count=0;
 if(active.length){
  for(let row=0;row<N;row++)for(let col=0;col<N;col++){
   if(row!==col && active.every(c=>c[row]===c[col])){mask[row*N+col]=1;count++;}
  }
 }
 function bounds(keys){
  if(!count)return [0,maxima[keys[0]]];
  const values=new Float32Array(count*keys.length);
  let n=0;
  for(const k of keys)for(let j=0;j<mask.length;j++)if(mask[j])values[n++]=mat[k][j];
  values.sort();
  const lo=values[Math.floor((n-1)*.02)],hi=values[Math.ceil((n-1)*.98)];
  return hi>lo?[lo,hi]:[0,maxima[keys[0]]];
 }
 const acoustic=bounds([1,2]), scales=[bounds([0]),acoustic,acoustic];
 views.forEach(({source,limits,legend},k)=>{
  legend.style.background='linear-gradient(90deg,white,black)';
  limits.children[1].textContent=k?'Cosine distance':'DIMSIM distance';
  const [lo,hi]=scales[k];
  const format=v=>Number(v.toPrecision(4)).toString();
  limits.firstElementChild.textContent=format(lo);
  limits.lastElementChild.textContent=format(hi);
  const ctx=source.getContext('2d'),img=ctx.createImageData(N,N);
  for(let row=0;row<N;row++)for(let col=0;col<N;col++){
   const p=4*(row*N+col),j=candidate(row,col);
   const match=!active.length||mask[row*N+j];
   const v=match?255-Math.round(255*clamp((mat[k][row*N+j]-lo)/(hi-lo),0,1)):238;
   img.data[p]=img.data[p+1]=img.data[p+2]=v;img.data[p+3]=255;
  }
  ctx.putImageData(img,0,0);
 });draw();updateStats(active);
}
const clamp=(v,lo,hi)=>Math.min(hi,Math.max(lo,v));
function draw(){
 x=clamp(x,0,N-span);y=clamp(y,0,N-span);
 const lo=v=>clamp(Math.floor(v),0,N-1),hi=v=>clamp(Math.ceil(v+span)-1,0,N-1);
 views.forEach(({canvas,extent,source})=>{
  const size=Math.round(canvas.clientWidth*devicePixelRatio);if(!size)return;
  if(canvas.width!==size||canvas.height!==size){canvas.width=canvas.height=size;}
  const ctx=canvas.getContext('2d');ctx.imageSmoothingEnabled=false;
  ctx.drawImage(source,x,y,span,span,0,0,size,size);
  if(hover){const [row,col]=hover;ctx.strokeStyle='#c33';ctx.lineWidth=1;ctx.strokeRect((col-x)*size/span,(row-y)*size/span,Math.max(2,size/span),Math.max(2,size/span));}
  extent.textContent='Rows '+labels[lo(y)]+' – '+labels[hi(y)]+' · '+(ranked?'Ranks '+(lo(x)+1)+'–'+(hi(x)+1):'Columns '+labels[lo(x)]+' – '+labels[hi(x)]);
 });
 document.querySelector('#zoom').textContent=(N/span).toFixed(1)+'×';
}
function zoom(f,u=.5,v=.5){const next=clamp(span*f,Math.min(5,N),N);x+=(span-next)*u;y+=(span-next)*v;span=next;clear();draw();}
function clear(){hover=null;readout.textContent='';}
function pos(e,c){const b=c.getBoundingClientRect();return [(e.clientX-b.left)/b.width,(e.clientY-b.top)/b.height];}
for(const {canvas:c} of views){
 c.addEventListener('wheel',e=>{e.preventDefault();zoom(Math.exp(clamp(e.deltaY,-240,240)*.002),...pos(e,c));},{passive:false});
 c.addEventListener('pointerdown',e=>{drag=pos(e,c);c.setPointerCapture(e.pointerId);});
 c.addEventListener('pointermove',e=>{
  const [u,v]=pos(e,c);
  if(drag){x+=(drag[0]-u)*span;y+=(drag[1]-v)*span;drag=[u,v];clear();}
  else {const row=clamp(Math.floor(y+v*span),0,N-1),col=clamp(Math.floor(x+u*span),0,N-1),j=candidate(row,col);
   hover=[row,col];
   if(agreeing){
    const cell=row*N+j,n=agreementSupport[cell];
    const pct=k=>(100*agreementMaps[k][cell]).toFixed(1)+'%';
    readout.textContent=labels[row]+' → '+labels[j]+(!n?' · No comparison':' · '+n+' alternatives · zh-cn '+pct(0)+' · base '+pct(1));
    draw();return;
   }
   if(effect&&!ranked){
    const cell=row*N+j,systems=['DIMSIM','zh-cn','base'],detail=[];
    const own=overlapCode(row,j);
    function add(low,high,role,support,measure){
     const line=systems.map((name,k)=>{
      const [delta,w,t]=measure(k);
      if(!winMode)return name+' '+(delta>=0?'+':'')+delta.toFixed(5)+' (n='+support+')';
      return name+' '+w+' win credit / '+support+' ('+(100*(w+.5*t)/support).toFixed(1)+'%)';
     });
     detail.push(maskLabel(low)+' → '+maskLabel(high)+' · '+role+': '+line.join(', '));
    }
    for(const [mask,d] of effectData.tests){
     if(d.counts[cell]&&eligibleCell(mask,cell))
      add(own&~mask,own,'criterion-predicted nearer',d.counts[cell],k=>[d.delta[k][cell],d.wins[k][cell],d.ties[k][cell]]);
    }
    readout.textContent=labels[row]+' → '+labels[j]+' · '+(detail.join(' | ')||'No matched comparison');
    draw();return;
   } else readout.textContent=labels[row]+' → '+labels[j]+(ranked?' · rank '+(col+1):'')+' · DIMSIM '+mat[0][row*N+j].toFixed(5)+' · zh-cn '+mat[1][row*N+j].toFixed(5)+' · base '+mat[2][row*N+j].toFixed(5);
  }draw();
 });
 c.addEventListener('lostpointercapture',()=>{drag=null;});
 c.addEventListener('pointerleave',()=>{if(!drag){clear();draw();}});
}
for(const box of boxes)box.addEventListener("change",rebuild);
function syncControls(){
 const showMatchingPairs=!ranked&&!agreeing&&!profiling&&!effect;
 filters.disabled=!showMatchingPairs;filters.hidden=!showMatchingPairs;
 filters.style.display=showMatchingPairs?'flex':'none';
 const showCellColor=!ranked&&!agreeing&&!profiling;
 document.querySelector('#color-mode').hidden=!showCellColor;
 document.querySelector('#color-mode').style.display=showCellColor?'flex':'none';
 document.querySelector('#distance-color').setAttribute('aria-pressed',!effect);
 document.querySelector('#final-color').setAttribute('aria-pressed',effect&&!winMode);
 document.querySelector('#win-color').setAttribute('aria-pressed',effect&&winMode);
 const showTests=agreeing?document.querySelector('#agreement-slice').value==='controlled':!ranked&&effect;
 document.querySelector('#win-tests').hidden=!showTests;
 document.querySelector('#win-tests').style.display=showTests?'flex':'none';
 groupBox.hidden=!showTests;groupBox.style.display=showTests?'flex':'none';
 groupTests[2].disabled=false;
 document.querySelector('#agreement-filter').hidden=!agreeing;
 document.querySelector('#effect-note').hidden=ranked||!effect;
 panels.style.gridTemplateColumns=agreeing?'repeat(2,minmax(250px,1fr))':'';
}
function color(value,rate=false){effect=value;winMode=rate;syncControls();clear();rebuild();}
document.querySelector('#distance-color').onclick=()=>color(false);
document.querySelector('#final-color').onclick=()=>color(true);
document.querySelector('#win-color').onclick=()=>color(true,true);
[...winTests,...groupTests].forEach(box=>box.addEventListener('change',()=>{clear();rebuild();}));
function mode(value){
 profiling=value==='profile';agreeing=value==='agreement';ranked=value===true;syncControls();
 document.querySelector('#map-results').hidden=profiling;
 document.querySelector('#profile-results').hidden=!profiling;
 document.querySelector('#mask-profile').setAttribute('aria-pressed',profiling);
 for(const id of ['minus','plus','reset','zoom'])document.querySelector('#'+id).hidden=profiling;
 if(profiling)for(const id of ['filters','win-tests','transition-groups','color-mode'])document.querySelector('#'+id).style.display='none';
 views[0].panel.hidden=agreeing;
 if(agreeing){
  document.querySelector('#filters').style.display='none';
  document.querySelector('#effect-note').hidden=true;
 }
 document.querySelector('#agreement').setAttribute('aria-pressed',agreeing);x=y=0;span=N;clear();document.querySelector('#hier').setAttribute('aria-pressed',!ranked&&!agreeing&&!profiling);document.querySelector('#rank').setAttribute('aria-pressed',ranked);rebuild();}
document.querySelector('#hier').onclick=()=>mode(false);
document.querySelector('#rank').onclick=()=>mode(true);
document.querySelector('#agreement').onclick=()=>mode('agreement');
document.querySelector('#mask-profile').onclick=()=>mode('profile');
document.querySelector('#agreement-slice').onchange=()=>{syncControls();clear();rebuild();};
document.querySelector('#minus').onclick=()=>zoom(1.4);
document.querySelector('#plus').onclick=()=>zoom(1/1.4);
document.querySelector('#reset').onclick=()=>{x=y=0;span=N;clear();draw();};
const profileColors=['#333','#175ca5','#bc4b16'],profileVisible=[true,true,true];
const maskOrder=profileData[0].beta.map((_,k)=>k).sort((a,b)=>(profileData[0].beta[a]??Infinity)-(profileData[0].beta[b]??Infinity));
function renderProfile(){
 const x=j=>70+95*j, y=v=>220-175*v;
 const refMin=Math.min(-45,...profileData[0].beta.filter(Number.isFinite));
 const wavMin=Math.min(.65,...profileData.slice(1).flatMap(m=>m.beta.filter(Number.isFinite)));
 const dim=v=>(v-refMin)/(-refMin), acoustic=v=>(v-wavMin)/(1-wavMin);
 const txt=(x,y,t,anchor='middle')=>'<text x="'+x+'" y="'+y+'" text-anchor="'+anchor+'" font-size="13">'+t+'</text>';
 const grid=Array.from({length:6},(_,j)=>'<line x1="70" y1="'+y(j/5)+'" x2="735" y2="'+y(j/5)+'" stroke="#ddd"/>'+txt(63,y(j/5)+4,(refMin*(1-j/5)).toFixed(0),'end')+txt(743,y(j/5)+4,(wavMin+(1-wavMin)*j/5).toFixed(2),'start')).join('');
 const lines=profileData.map((m,k)=>{
  if(!profileVisible[k])return '';
  const visible=maskOrder.map((a,j)=>({a,j})).filter(({a})=>m.beta[a]!==null);
   const points=visible.map(({a,j})=>[x(j),y((k?acoustic:dim)(m.beta[a]))]);
  return '<polyline fill="none" stroke="'+profileColors[k]+'" stroke-width="2.5" points="'+points.map(p=>p.join(',')).join(' ')+'"/>'+points.map(([x,y],j)=>'<circle cx="'+x+'" cy="'+y+'" r="4" fill="'+profileColors[k]+'"><title>'+m.name+' · '+m.beta[visible[j].a].toFixed(5)+'</title></circle>').join('');
 }).join('');
 document.querySelector('#profile-chart').innerHTML='<svg role="img" aria-label="Mask means by system" viewBox="0 0 825 260" style="width:100%;max-width:1000px">'+txt(63,23,'−DIMSIM distance','start')+txt(735,23,'Mean cosine','end')+grid+lines+maskOrder.map((k,j)=>txt(x(j),244,k.toString(2).padStart(3,'0'))).join('')+'</svg>';
 const legend=document.querySelector('#profile-legend');
 legend.innerHTML=profileData.map((m,k)=>'<button aria-pressed="'+profileVisible[k]+'" style="border-color:'+profileColors[k]+'">'+m.name+'</button>').join('');
 legend.querySelectorAll('button').forEach((b,k)=>b.onclick=()=>{profileVisible[k]=!profileVisible[k];renderProfile();});
 const headers=profileData.map(m=>'<th scope="col">'+m.name+'</th>').join('');
 const row=(name,values)=>'<tr><th scope="row">'+name+'</th>'+values.map(v=>'<td>'+v+'</td>').join('')+'</tr>';
 document.querySelector('#profile-r2').innerHTML='<thead><tr><th>Pair set</th>'+headers+'</tr></thead><tbody>'+['Distinct syllables','Including self-pairs (111)'].map((name,j)=>row(name,profileData.map(m=>m.r2[j]===null?'—':(100*m.r2[j]).toFixed(2)+'%'))).join('')+'</tbody>';
 document.querySelector('#profile-residuals').innerHTML='<thead><tr><th scope="col">Rank</th>'+profileData.map(m=>'<th scope="col">'+m.name+'</th>').join('')+'</tr></thead><tbody>'+
 Array.from({length:50},(_,rank)=>'<tr><th scope="row">'+(rank+1)+'</th>'+residualTopPairs.map((pairs,k)=>{
  const p=pairs[rank],digits=k===0?3:4; if(!p)return '<td>—</td>';
  return '<td style="text-align:left;white-space:nowrap"><strong>'+p.a+' · '+p.b+'</strong><br><span class="hint">'+p.mask+' · residual +'+p.residual.toFixed(digits)+'</span></td>';
 }).join('')+'</tr>').join('')+'</tbody>';
 renderProfileSteps();
}
function renderProfileSteps(){
 const k=Number(document.querySelector('#profile-step-model').value),model=profileData[k];
 const values=model.steps.flatMap(t=>[...t.values,t.baseline,t.absolute]).filter(Number.isFinite);
 const lo=Math.min(0,...values),hi=Math.max(0,...values);
 const x=v=>270+500*(v-lo)/(hi-lo||1);
 const text=(x,y,t,anchor='start')=>'<text x="'+x+'" y="'+y+'" text-anchor="'+anchor+'" font-size="13">'+t+'</text>';
 const names=['Initial','Final','Tone'];
 const groups=model.steps.map((step,i)=>{
  const labels=[...step.labels.slice(0,3),'Ordinary mean',step.labels[3],'Absolute J'];
  const vals=[...step.values.slice(0,3),step.baseline,step.values[3],step.absolute];
  const top=50+i*162;
  return text(10,top-10,names[i])+labels.map((label,j)=>{
   const v=vals[j],y=top+j*22;if(v===null)return text(15,y+11,label+' · unavailable');const end=x(v);
   const fill=j===5?'#333':j>=3?profileColors[k]:'#999';
   const number=(v>0?'+':'')+v.toFixed(k===0?3:4);
   return text(15,y+11,label)+'<rect x="'+Math.min(x(0),end)+'" y="'+y+'" width="'+Math.abs(end-x(0))+'" height="13" fill="'+fill+'"/>'+text(end+(v<0?-5:5),y+11,number,v<0?'end':'start');
  }).join('');
 }).join('');
 document.querySelector('#profile-steps').innerHTML='<svg role="img" aria-label="Component gains and absolute J for '+model.name+'" viewBox="0 0 850 520" style="display:block;width:100%;max-width:1000px">'+text(270,22,k===0?'Change in negative DIMSIM distance':'Change in mean cosine')+'<line x1="'+x(0)+'" x2="'+x(0)+'" y1="35" y2="504" stroke="#aaa"/>'+groups+'</svg>';
}
document.querySelector('#profile-step-model').onchange=renderProfileSteps;
window.addEventListener('resize',draw);rebuild();
</script></html>"""
