"""Self-contained offline HTML audit dashboard for Step 8C."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def _track_frames(video, track_id):
    rows=[]
    for frame in video.get("frames",[]):
        frame_id=int(frame.get("frame_index",len(rows)))
        for obj in frame.get("objects",[]):
            if int(obj.get("track_id",-1))==int(track_id):
                rows.append({"frame_id":frame_id,"image_path":str(frame.get("image_path","")),
                    "bbox":list(obj.get("bbox",obj.get("box",[]))),"label":str(obj.get("frame_label",obj.get("label","unknown")))})
                break
    return rows


def _video_path(video):
    for key in ("video_path","input_video_path","source_video_path","output_video_path"):
        if video.get(key):return str(video[key])
    return ""


def _llm_audit(audit_root):
    rows=[];root=Path(audit_root)
    if not root.exists():return rows
    for path in sorted(root.glob("*/*.json")):
        try: payload=json.loads(path.read_text(encoding="utf-8"))
        except (OSError,json.JSONDecodeError):continue
        rows.append({"kind":payload.get("kind",path.parent.name),"request_id":payload.get("request_id",path.stem),
                     "prompt":payload.get("prompt",""),"response":payload.get("response",{}),"source_path":str(path)})
    return rows


def _ablations(records):
    operations=defaultdict(lambda:{"total":0,"passed":0,"selected":0})
    disagreements=unresolved=hard_rejected=0;baseline_present=0;no_llm_agreement=0
    llm_called=llm_skipped=cache_hits=escalated=0
    for record in records:
        disagreements+=str(record.get("LLM_preferred_pattern"))!=str(record.get("validated_pattern"))
        llm=dict(record.get("llm_processing",{}));llm_called+=bool(llm.get("llm_called"));llm_skipped+=bool(llm.get("llm_skipped"));cache_hits+=bool(llm.get("cache_hit"));escalated+=bool(llm.get("escalated_to_single"))
        unresolved+=str(record.get("resolution_status"))!="validated"
        candidates=list(record.get("candidate_repairs",[]));selected=str(record.get("selected_candidate",{}).get("candidate_id",""))
        passing=[row for row in candidates if row.get("symbolic_verdict")=="pass"]
        hard_rejected+=sum(row.get("symbolic_verdict")=="reject" for row in candidates)
        baseline=[row for row in candidates if "minimum_residual_baseline" in row.get("pattern_hypothesis",{}).get("selection_sources",[])]
        baseline_present+=bool(baseline)
        if passing:
            no_llm=max(passing,key=lambda row:.5*float(row.get("residual_improvement",0))+.3*float(row.get("statistical_success_prior",0))+.2*float(row.get("calibration_confidence",0)))
            no_llm_agreement+=no_llm.get("pattern_id")==record.get("validated_pattern")
        for row in candidates:
            op=str(row.get("repair_operation","unknown"));operations[op]["total"]+=1
            operations[op]["passed"]+=row.get("symbolic_verdict")=="pass"
            operations[op]["selected"]+=row.get("candidate_id")==selected
    total=max(1,len(records))
    return {"track_count":len(records),"validated_count":len(records)-unresolved,"unresolved_count":unresolved,
      "llm_validated_disagreements":disagreements,"llm_validated_agreement_rate":(len(records)-disagreements)/total,
      "non_llm_baseline_coverage":baseline_present/total,"no_llm_counterfactual_agreement_rate":no_llm_agreement/total,
      "hard_rejected_candidate_count":hard_rejected,"llm_called":llm_called,"llm_skipped":llm_skipped,
      "cache_hits":cache_hits,"escalated_to_single":escalated,"repair_operations":[{"operation":key,**value} for key,value in sorted(operations.items())],
      "note":"Ablations are read-only counterfactual summaries and never modify pipeline decisions."}


def _dashboard_data(state,audit_root):
    raw={str(row.get("video_id","")):row for row in state.get("pre_pattern_relative_object_motion",[])}
    repaired={str(row.get("video_id","")):row for row in state.get("relative_object_motion",[])}
    records=[]
    for source in state.get("trajectory_pattern_records",[]):
        record=json.loads(json.dumps(source));video_id=str(record.get("video_id",""));track_id=int(record.get("track_id",-1))
        record["media"]={"raw_video_path":_video_path(raw.get(video_id,{})),"repaired_video_path":_video_path(repaired.get(video_id,{})),
          "raw_frames":_track_frames(raw.get(video_id,{}),track_id),"repaired_frames":_track_frames(repaired.get(video_id,{}),track_id)}
        records.append(record)
    return {"schema_version":1,"read_only":True,"records":records,
      "manifest":state.get("trajectory_pattern_manifest",{}),"promotion":state.get("trajectory_pattern_statistics_promotion",{}),
      "statistics_candidate":state.get("trajectory_pattern_statistics_candidate",{}),
      "statistics_review":state.get("trajectory_pattern_statistics_review",{}),"llm_audit":_llm_audit(audit_root),
      "runtime_monitor":state.get("trajectory_pattern_runtime_monitor",{}),
      "ablations":_ablations(records)}


_HTML="""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Step 8C Audit Dashboard</title><style>
:root{--bg:#11151b;--panel:#1b222c;--line:#344050;--text:#edf2f7;--muted:#9ba8b8;--good:#54d98c;--bad:#ff6b76;--warn:#ffc857;--blue:#62b5ff}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font:14px system-ui,sans-serif}header{position:sticky;top:0;z-index:5;background:#151b23;padding:12px 18px;border-bottom:1px solid var(--line)}h1{font-size:20px;margin:0 0 10px}.filters{display:flex;gap:8px;flex-wrap:wrap}select,button,input{background:#252e3a;color:var(--text);border:1px solid #435064;border-radius:5px;padding:7px}button{cursor:pointer}.layout{display:grid;grid-template-columns:280px 1fr;min-height:calc(100vh - 100px)}aside{border-right:1px solid var(--line);padding:12px;overflow:auto}.track{padding:8px;border-bottom:1px solid var(--line);cursor:pointer}.track.active{background:#263346}.badge{display:inline-block;padding:2px 6px;border-radius:10px;margin:2px;font-size:12px}.good{color:var(--good)}.bad{color:var(--bad)}.warn{color:var(--warn)}.badge.good{background:#173c2b}.badge.bad{background:#4a2027}.badge.warn{background:#4b3b18}main{padding:14px;overflow:auto}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;margin-bottom:12px}.wide{grid-column:1/-1}h2{font-size:16px;margin:0 0 10px}video,.frame{width:100%;height:300px;background:#090b0e;object-fit:contain}.controls{display:flex;gap:8px;align-items:center;margin-top:8px}.controls input{flex:1}canvas{width:100%;height:280px;background:#12171e;border:1px solid var(--line)}table{width:100%;border-collapse:collapse}th,td{text-align:left;padding:6px;border-bottom:1px solid var(--line);vertical-align:top}tr.click{cursor:pointer}tr.selected{background:#263346}pre{white-space:pre-wrap;word-break:break-word;background:#10151b;padding:10px;max-height:360px;overflow:auto}.alert{border-left:4px solid var(--warn);padding:8px;background:#292718;margin:7px 0}.alert.bad{border-color:var(--bad);background:#331d22}.tabs button.active{border-color:var(--blue);color:var(--blue)}.hidden{display:none}@media(max-width:900px){.layout{grid-template-columns:1fr}aside{max-height:240px;border-right:0}.grid{grid-template-columns:1fr}.wide{grid-column:auto}}</style></head>
<body><header><h1>Step 8C Static Audit Dashboard <span class='badge warn'>READ ONLY</span></h1><div class='filters'>
<select id='videoFilter'></select><select id='trackFilter'></select><select id='caseFilter'><option value='all'>All cases</option><option value='disagreement'>LLM disagreement</option><option value='unresolved'>Unresolved</option><option value='hard_failure'>Hard-constraint failure</option></select>
<button id='prev'>Previous</button><button id='next'>Next</button><span id='position'></span></div></header><div class='layout'><aside id='trackList'></aside><main>
<div id='alerts'></div><div class='grid'><section class='panel'><h2>Raw trajectory playback</h2><video id='rawVideo' controls></video><img id='rawFrame' class='frame'><div id='rawLabel' class='good'></div></section>
<section class='panel'><h2>Repaired trajectory playback</h2><video id='repairedVideo' controls></video><img id='repairedFrame' class='frame'><div id='repairedLabel' class='good'></div></section>
<section class='panel wide'><div class='controls'><button id='playFrames'>Play/Pause</button><input id='frameSlider' type='range' min='0' max='0' value='0'><span id='frameValue'></span></div></section>
<section class='panel wide'><h2>Interactive signals</h2><select id='signal'><option value='depth'>Depth</option><option value='vx'>Vx</option><option value='vz'>Vz</option><option value='position_x'>Position X</option><option value='confidence'>Confidence</option></select><canvas id='signalPlot' width='1500' height='300'></canvas></section>
<section class='panel wide'><h2>Pattern and residual comparison</h2><canvas id='residualPlot' width='1500' height='320'></canvas><div id='patternTable'></div></section>
<section class='panel wide'><h2>Repair candidates</h2><div id='candidateTable'></div><div id='candidateDetail'></div></section>
<section class='panel'><h2>Symbolic validation</h2><div id='symbolic'></div></section><section class='panel'><h2>Final decision and provenance</h2><div id='decision'></div></section>
<section class='panel wide'><h2>LLM audit records</h2><select id='llmKind'></select><div id='llmAudit'></div></section>
<section class='panel wide'><h2>Dataset-level ablation summary</h2><div id='ablations'></div></section><section class='panel wide'><h2>Statistics promotion</h2><div id='promotion'></div></section></div></main></div>
<script id='audit-data' type='application/json'>__DATA__</script><script>
const D=JSON.parse(document.getElementById('audit-data').textContent),S={rows:[],index:0,timer:null,candidate:0};const $=id=>document.getElementById(id), esc=x=>String(x??'').replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
function unique(a){return [...new Set(a)]}function init(){const vids=unique(D.records.map(r=>r.video_id));$('videoFilter').innerHTML='<option value="all">All videos</option>'+vids.map(v=>`<option>${esc(v)}</option>`).join('');fillTracks();['videoFilter','trackFilter','caseFilter'].forEach(id=>$(id).onchange=filter);$('prev').onclick=()=>nav(-1);$('next').onclick=()=>nav(1);$('signal').onchange=renderSignal;$('playFrames').onclick=toggleFrames;$('frameSlider').oninput=renderFrame;$('rawVideo').ontimeupdate=syncVideo;$('rawVideo').onplay=()=>syncPlayback('play');$('rawVideo').onpause=()=>syncPlayback('pause');$('rawVideo').onratechange=()=>{$('repairedVideo').playbackRate=$('rawVideo').playbackRate};$('llmKind').onchange=renderLLM;filter();renderAblations();renderPromotion()}
function fillTracks(){const ids=unique(D.records.map(r=>r.track_id));$('trackFilter').innerHTML='<option value="all">All tracks</option>'+ids.map(v=>`<option>${v}</option>`).join('')}
function isCase(r,c){if(c==='disagreement')return r.LLM_preferred_pattern!==r.validated_pattern;if(c==='unresolved')return r.resolution_status!=='validated';if(c==='hard_failure')return r.candidate_repairs.some(x=>x.symbolic_verdict==='reject');return true}
function filter(){const v=$('videoFilter').value,t=$('trackFilter').value,c=$('caseFilter').value;S.rows=D.records.filter(r=>(v==='all'||r.video_id===v)&&(t==='all'||String(r.track_id)===t)&&isCase(r,c));S.index=Math.min(S.index,Math.max(0,S.rows.length-1));renderList();render()}
function nav(d){if(!S.rows.length)return;S.index=(S.index+d+S.rows.length)%S.rows.length;renderList();render()}
function renderList(){$('trackList').innerHTML=S.rows.map((r,i)=>`<div class="track ${i===S.index?'active':''}" onclick="S.index=${i};renderList();render()"><b>${esc(r.video_id)} / track ${r.track_id}</b><br><span class="badge ${r.resolution_status==='validated'?'good':'warn'}">${esc(r.validated_pattern)}</span>${r.LLM_preferred_pattern!==r.validated_pattern?'<span class="badge bad">LLM ≠ validated</span>':''}</div>`).join('')||'No matching tracks'}
function current(){return S.rows[S.index]}function render(){const r=current();$('position').textContent=S.rows.length?`${S.index+1}/${S.rows.length}`:'0/0';if(!r)return;renderAlerts(r);renderMedia(r);renderSignal();renderResidual(r);renderCandidates(r);renderSymbolic(r);renderDecision(r);renderLLM()}
function renderAlerts(r){let a=[];if(r.LLM_preferred_pattern!==r.validated_pattern)a.push(`<div class="alert bad">LLM preferred <b>${esc(r.LLM_preferred_pattern)}</b>, validated pattern is <b>${esc(r.validated_pattern)}</b>.</div>`);if(r.resolution_status!=='validated')a.push('<div class="alert">No candidate passed all hard constraints; original trajectory was preserved.</div>');const sel=r.selected_candidate||{},ss=sum((sel.post_repair_pattern_scores||{})[sel.pattern_id]||{});if(r.candidate_repairs.some(x=>x.symbolic_verdict==='reject'&&sum((x.post_repair_pattern_scores||{})[x.pattern_id]||{})<ss))a.push('<div class="alert bad">A lower-residual candidate was rejected by symbolic constraints.</div>');$('alerts').innerHTML=a.join('')}
function setupMedia(el,img,path,frames){if(path){el.src=path;el.classList.remove('hidden');img.classList.add('hidden')}else{el.removeAttribute('src');el.classList.add('hidden');img.classList.remove('hidden')}return frames||[]}
function renderMedia(r){S.raw=setupMedia($('rawVideo'),$('rawFrame'),r.media.raw_video_path,r.media.raw_frames);S.repaired=setupMedia($('repairedVideo'),$('repairedFrame'),r.media.repaired_video_path,r.media.repaired_frames);$('frameSlider').max=Math.max(0,Math.max(S.raw.length,S.repaired.length)-1);$('frameSlider').value=0;renderFrame()}
function syncPlayback(action){const b=$('repairedVideo');if(b.classList.contains('hidden'))return;if(action==='play')b.play().catch(()=>{});else b.pause()}
function syncVideo(){const a=$('rawVideo'),b=$('repairedVideo');if(!b.classList.contains('hidden')&&Math.abs(a.currentTime-b.currentTime)>.15)b.currentTime=a.currentTime}
function renderFrame(){const i=+$('frameSlider').value;for(const [name,rows] of [['raw',S.raw],['repaired',S.repaired]]){const row=rows[Math.min(i,Math.max(0,rows.length-1))]||{};$(name+'Frame').src=row.image_path||'';$(name+'Label').textContent=row.frame_id===undefined?'media unavailable':`frame ${row.frame_id} | bbox ${JSON.stringify(row.bbox||[])}`} $('frameValue').textContent=i}
function toggleFrames(){if(S.timer){clearInterval(S.timer);S.timer=null}else S.timer=setInterval(()=>{let n=(+$('frameSlider').value+1)%(+$('frameSlider').max+1);$('frameSlider').value=n;renderFrame()},120)}
function sig(row,key){if(key==='position_x')return (row.position||[])[0]??0;return +row[key]||0}function plot(canvas,series){const c=canvas.getContext('2d'),W=canvas.width,H=canvas.height;c.clearRect(0,0,W,H);c.strokeStyle='#344050';c.strokeRect(45,15,W-65,H-45);let vals=series.flatMap(s=>s.rows.map(x=>x.y));let lo=Math.min(...vals,0),hi=Math.max(...vals,1),span=hi-lo||1;series.forEach(s=>{c.beginPath();c.strokeStyle=s.color;c.lineWidth=3;s.rows.forEach((p,i)=>{let x=50+i*Math.max(1,(W-75)/Math.max(1,s.rows.length-1)),y=H-35-(p.y-lo)/span*(H-60);i?c.lineTo(x,y):c.moveTo(x,y)});c.stroke();c.fillStyle=s.color;c.fillText(s.name,60+series.indexOf(s)*170,32)})}
function renderSignal(){const r=current();if(!r)return;const s=r.selected_candidate||r.candidate_repairs[0]||{},key=$('signal').value;plot($('signalPlot'),[{name:'raw',color:'#62b5ff',rows:(s.signals_before||[]).map(x=>({y:sig(x,key)}))},{name:'repaired',color:'#54d98c',rows:(s.signals_after||[]).map(x=>({y:sig(x,key)}))}])}
function sum(o){return Object.values(o||{}).reduce((a,b)=>a+(+b||0),0)}function renderResidual(r){const pre=r.pattern_candidates.map(x=>({id:x.pattern_id,v:sum(x.residual_vector)})),post=(r.selected_candidate.post_repair_pattern_scores||{});const c=$('residualPlot').getContext('2d'),W=1500,H=320;c.clearRect(0,0,W,H);let max=Math.max(1,...pre.flatMap(x=>[x.v,sum(post[x.id])]));pre.forEach((x,i)=>{let px=40+i*140;c.fillStyle='#62b5ff';c.fillRect(px,H-45-x.v/max*220,42,x.v/max*220);c.fillStyle='#54d98c';let pv=sum(post[x.id]);c.fillRect(px+45,H-45-pv/max*220,42,pv/max*220);c.fillStyle='#edf2f7';c.fillText(x.id,px,H-25)});$('patternTable').innerHTML='<span class="good">green=post-repair</span> <span style="color:#62b5ff">blue=pre-pattern</span>'}
function renderCandidates(r){let rows=[...r.candidate_repairs].sort((a,b)=>(b.final_score??-1e9)-(a.final_score??-1e9));$('candidateTable').innerHTML='<table><tr><th>candidate</th><th>hypothesis source</th><th>repair</th><th>verdict</th><th>improvement</th><th>score</th><th>reason</th></tr>'+rows.map((x,i)=>`<tr class="click ${i===S.candidate?'selected':''}" onclick="S.candidate=${i};renderCandidateDetail()"><td>${esc(x.candidate_id)}</td><td>${esc((x.pattern_hypothesis.selection_sources||[]).join(', '))}</td><td>${esc(x.repair_hypothesis.operation)}</td><td class="${x.symbolic_verdict==='pass'?'good':'bad'}">${esc(x.symbolic_verdict)}</td><td>${(+x.residual_improvement).toFixed(3)}</td><td>${x.final_score==null?'—':(+x.final_score).toFixed(3)}</td><td>${esc(x.final_selection_reason)}</td></tr>`).join('')+'</table>';S.candidates=rows;renderCandidateDetail()}
function renderCandidateDetail(){const x=(S.candidates||[])[S.candidate]||{};$('candidateDetail').innerHTML=`<h3>${esc(x.candidate_id||'')}</h3><pre>${esc(JSON.stringify(x,null,2))}</pre>`}
function renderSymbolic(r){const x=r.selected_candidate||{};$('symbolic').innerHTML=Object.entries(x.hard_constraint_results||{}).map(([k,v])=>`<div><span class="badge ${v?'good':'bad'}">${v?'PASS':'FAIL'}</span> ${esc(k)}</div>`).join('')||'<span class="warn">No validated candidate</span>'}
function renderDecision(r){$('decision').innerHTML=`<b>LLM prior:</b> ${esc(r.LLM_preferred_pattern)}<br><b>Validated:</b> <span class="good">${esc(r.validated_pattern)}</span><br><b>Status:</b> ${esc(r.resolution_status)}<br><b>Reason:</b> ${esc(r.final_selection_reason)}<h3>LLM batch processing</h3><pre>${esc(JSON.stringify(r.llm_processing||{},null,2))}</pre><h3>Provenance</h3><pre>${esc(JSON.stringify(r.provenance,null,2))}</pre>`}
function renderLLM(){const r=current(),k=$('llmKind').value||'all';let kinds=unique(D.llm_audit.map(x=>x.kind));$('llmKind').innerHTML='<option value="all">All kinds</option>'+kinds.map(x=>`<option ${x===k?'selected':''}>${esc(x)}</option>`).join('');let rows=D.llm_audit.filter(x=>(k==='all'||x.kind===k)&&(!r||x.kind!=='residual_interpretation'||x.prompt.includes(`\"track_id\":${r.track_id}`)&&x.prompt.includes(r.video_id)));$('llmAudit').innerHTML=rows.map(x=>`<details><summary>${esc(x.kind)} | ${esc(x.request_id)}</summary><h4>Prompt</h4><pre>${esc(x.prompt)}</pre><h4>Response</h4><pre>${esc(JSON.stringify(x.response,null,2))}</pre></details>`).join('')||'No matching LLM audit record'}
function renderAblations(){const a=D.ablations;$('ablations').innerHTML=`<div class="grid"><div>Tracks: <b>${a.track_count}</b></div><div>Validated: <b>${a.validated_count}</b></div><div>Unresolved: <b>${a.unresolved_count}</b></div><div>LLM/validated agreement: <b>${(100*a.llm_validated_agreement_rate).toFixed(1)}%</b></div><div>Non-LLM baseline coverage: <b>${(100*a.non_llm_baseline_coverage).toFixed(1)}%</b></div><div>No-LLM counterfactual agreement: <b>${(100*a.no_llm_counterfactual_agreement_rate).toFixed(1)}%</b></div><div>LLM called / skipped: <b>${a.llm_called} / ${a.llm_skipped}</b></div><div>Cache hits / single escalations: <b>${a.cache_hits} / ${a.escalated_to_single}</b></div></div><table><tr><th>repair</th><th>candidates</th><th>hard pass</th><th>selected</th></tr>${a.repair_operations.map(x=>`<tr><td>${esc(x.operation)}</td><td>${x.total}</td><td>${x.passed}</td><td>${x.selected}</td></tr>`).join('')}</table><p class="warn">${esc(a.note)}</p>`}
function renderPromotion(){const p=D.promotion,e=D.manifest.batch_evaluation||{};$('promotion').innerHTML=`<span class="badge ${p.decision==='accept'?'good':p.reason==='validation_regression'?'bad':'warn'}">${esc(p.decision)}</span> ${esc(p.reason)}<pre>${esc(JSON.stringify(p,null,2))}</pre><h3>Batch-size evaluation</h3>${e.results?`<table><tr><th>batch</th><th>pattern agreement</th><th>repair agreement</th><th>validation success</th><th>malformed</th><th>latency</th><th>token cost</th></tr>${e.results.map(x=>`<tr><td>${x.batch_size}</td><td>${(+x.pattern_ranking_agreement).toFixed(3)}</td><td>${(+x.repair_advice_agreement).toFixed(3)}</td><td>${(+x.symbolic_validation_success_rate).toFixed(3)}</td><td>${(+x.malformed_output_rate).toFixed(3)}</td><td>${(+x.latency).toFixed(3)}s</td><td>${(+x.token_cost).toFixed(0)}</td></tr>`).join('')}</table><p>Chosen batch size: <b>${e.chosen_batch_size}</b></p>`:'Evaluation mode has not been run.'}` }init();
</script></body></html>"""


def build_trajectory_pattern_dashboard(state,output_root,audit_root):
    """Write a read-only, dependency-free dashboard after all decisions are complete."""
    output_root=Path(output_root);output_root.mkdir(parents=True,exist_ok=True)
    data=_dashboard_data(state,audit_root);encoded=json.dumps(data,separators=(",",":"),ensure_ascii=False).replace("</","<\\/")
    path=output_root/"index.html";path.write_text(_HTML.replace("__DATA__",encoded),encoding="utf-8")
    manifest={"version":1,"read_only":True,"self_contained":True,"dashboard_path":str(path),
              "num_videos":len({row["video_id"] for row in data["records"]}),"num_tracks":len(data["records"]),
              "num_llm_audit_records":len(data["llm_audit"])}
    (output_root/"dashboard_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    return {**state,"trajectory_pattern_dashboard":manifest,"trajectory_pattern_dashboard_path":path,
            "trajectory_pattern_dashboard_output_root":output_root}