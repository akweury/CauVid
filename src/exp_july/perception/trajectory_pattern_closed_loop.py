"""Step 8C: reproducible trajectory-pattern recognition and repair loop."""

from __future__ import annotations
import copy, hashlib, json, math, os, statistics, time, urllib.error, urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from tqdm import tqdm

from src.exp_july.perception.adaptive_motion_repair import (
    _apply_strategy, _evaluate, _issue_cost, _materialize_repaired_relative_video,
    _modified_frames, _recompute_motion, _snapshot,
)

VERSION = 3
PATTERNS = ("stationary","same_direction","opposite_direction","approaching","receding",
            "crossing","turning","lane_entry","overtaking","unknown")
RESIDUALS = ("position","direction","speed","acceleration","path_intersection","ttc",
             "continuity","depth_consistency","ego_motion_consistency")
REPAIRS = ("interpolation","outlier_removal","kalman_smoothing","segment_split",
           "fragment_merge","depth_refinement","ego_motion_refinement","motion_recomputation")
EXECUTORS = {
    "interpolation": ("gap_interpolation", {"maximum_gap": 12}),
    "outlier_removal": ("outlier_removal", {"median_radius": 2, "mad_scale": 3.0}),
    "kalman_smoothing": ("kalman_smoothing", {"alpha": 0.55}),
    "segment_split": ("track_split", {"minimum_segment_length": 2}),
    "fragment_merge": ("fragment_reassociation", {"normalize_to_dominant_label": True}),
    "depth_refinement": ("depth_reestimation", {"median_radius": 2, "mad_scale": 2.5}),
    "motion_recomputation": ("multi_frame_velocity_recomputation", {"window": 3}),
}
DEFAULT_REPAIRS = {
    "stationary": ("kalman_smoothing","depth_refinement","motion_recomputation"),
    "same_direction": ("kalman_smoothing","motion_recomputation","ego_motion_refinement"),
    "opposite_direction": ("outlier_removal","motion_recomputation"),
    "approaching": ("depth_refinement","motion_recomputation","kalman_smoothing"),
    "receding": ("depth_refinement","motion_recomputation","kalman_smoothing"),
    "crossing": ("fragment_merge","kalman_smoothing","motion_recomputation"),
    "turning": ("kalman_smoothing","motion_recomputation"),
    "lane_entry": ("fragment_merge","interpolation","kalman_smoothing"),
    "overtaking": ("fragment_merge","motion_recomputation","kalman_smoothing"),
    "unknown": ("outlier_removal","interpolation","motion_recomputation"),
}

def f(v, d=0.0):
    try:
        x=float(v); return x if math.isfinite(x) else d
    except (TypeError,ValueError): return d

def mean(v): return sum(map(f,v))/max(1,len(v))
def median(v):
    a=sorted(map(f,v)); n=len(a)
    return 0.0 if not a else a[n//2] if n%2 else (a[n//2-1]+a[n//2])/2
def quantile(v,q):
    a=sorted(map(f,v))
    if not a:return 0.0
    p=max(0,min(1,q))*(len(a)-1); lo=int(p); hi=math.ceil(p); r=p-lo
    return a[lo]*(1-r)+a[hi]*r

def http_llm(prompt,return_metadata=False):
    started=time.perf_counter();timeout_occurred=False
    key=os.environ.get("OPENAI_API_KEY","").strip()
    if not key: raise RuntimeError("OPENAI_API_KEY is not configured")
    base=os.environ.get("OPENAI_BASE_URL","https://api.openai.com/v1").rstrip("/")
    url=base if base.endswith("/chat/completions") else base+"/chat/completions"
    model=os.environ.get("CAUVID_STEP8_PATTERN_LLM_MODEL",os.environ.get("OPENAI_MODEL","gpt-4.1-mini"))
    request_body={"model":model,"temperature":0,"response_format":{"type":"json_object"},"messages":[
        {"role":"system","content":"Return auditable JSON only; never generate corrected values or thresholds."},
        {"role":"user","content":prompt}]}
    req=urllib.request.Request(url,data=json.dumps(request_body).encode(),headers={
        "Authorization":f"Bearer {key}","Content-Type":"application/json"},method="POST")
    timeout=max(1.0,float(os.environ.get("CAUVID_STEP8C_LLM_TIMEOUT_SECONDS","120")))
    max_attempts=max(1,int(os.environ.get("CAUVID_STEP8C_LLM_MAX_ATTEMPTS","3")))
    backoff=max(0.0,float(os.environ.get("CAUVID_STEP8C_LLM_RETRY_BACKOFF_SECONDS","2")))
    payload=None;last_error=None
    for attempt in range(1,max_attempts+1):
        try:
            with urllib.request.urlopen(req,timeout=timeout) as res: payload=json.loads(res.read().decode())
            break
        except urllib.error.HTTPError as exc:
            try: detail=exc.read().decode("utf-8",errors="replace")[:2000]
            except Exception: detail="response body unavailable"
            last_error=exc
            transient=exc.code in {408,409,429} or 500<=exc.code<600
            if not transient or attempt>=max_attempts:
                raise RuntimeError(
                    f"LLM request failed after {attempt} attempt(s): HTTP {exc.code}; "
                    f"model={model}; endpoint={url}; prompt_chars={len(prompt)}; response={detail}"
                ) from exc
        except (urllib.error.URLError,TimeoutError) as exc:
            timeout_occurred=timeout_occurred or isinstance(exc,TimeoutError) or "timed out" in str(exc).lower()
            last_error=exc
            if attempt>=max_attempts:
                raise RuntimeError(
                    f"LLM request failed after {attempt} attempt(s): model={model}; "
                    f"endpoint={url}; timeout={timeout}s; error={exc}"
                ) from exc
        print(
            f"[step 8c] transient LLM failure; retry={attempt + 1}/{max_attempts} "
            f"backoff={backoff * attempt:.1f}s error={last_error}"
        )
        if backoff:time.sleep(backoff*attempt)
    text=str(payload["choices"][0]["message"]["content"]).strip()
    fence=chr(96)*3
    if text.startswith(fence):
        text=text.removeprefix(fence+"json").removeprefix(fence).removesuffix(fence).strip()
    out=json.loads(text)
    if not isinstance(out,dict): raise ValueError("LLM response must be an object")
    usage=payload.get("usage",{}) if isinstance(payload,dict) else {}
    metadata={"real_llm_call":True,"backend":"openai_chat_completions","model":model,
      "heuristic_fallback":False,"timeout_occurred":timeout_occurred,"attempt_count":attempt,
      "prompt_tokens":int(usage.get("prompt_tokens",0) or 0),
      "completion_tokens":int(usage.get("completion_tokens",0) or 0),
      "total_tokens":int(usage.get("total_tokens",0) or 0),
      "true_latency_seconds":time.perf_counter()-started}
    return (out,metadata) if return_metadata else out

def llm_call(kind,prompt,root,generator):
    rid=hashlib.sha256((kind+"\n"+prompt).encode()).hexdigest()
    path=root/kind/f"{rid}.json"
    expose_metadata=kind.startswith("batch_") or kind.endswith("_individual")
    if path.exists():
        cached=json.loads(path.read_text());response=dict(cached["response"]);prior=dict(cached.get("llm_call_metadata",{}))
        if expose_metadata:
            response["__llm_call_metadata__"]={**prior,"real_llm_call":False,"audit_cache_hit":True,
              "backend":"llm_audit_cache","true_latency_seconds":0.0,"timeout_occurred":False}
        return response
    if generator is None:
        response,call_metadata=http_llm(prompt,return_metadata=True)
    else:
        started=time.perf_counter()
        try: response=generator(kind,prompt)
        except TypeError: response=generator(prompt)
        supplied=response.pop("__llm_call_metadata__",{}) if isinstance(response,dict) else {}
        call_metadata={"real_llm_call":False,"backend":"callable_generator",
          "model":getattr(generator,"__name__",type(generator).__name__),"heuristic_fallback":True,
          "timeout_occurred":False,"prompt_tokens":0,"completion_tokens":0,"total_tokens":0,
          "true_latency_seconds":time.perf_counter()-started,**supplied}
    if not isinstance(response,dict): raise ValueError(f"invalid {kind} response")
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({"version":VERSION,"kind":kind,"request_id":rid,
                                "prompt":prompt,"response":response,"llm_call_metadata":call_metadata},indent=2))
    return {**response,"__llm_call_metadata__":call_metadata} if expose_metadata else response

def symbolic_tracks(evidence):
    out=[]
    for video in evidence:
        vid=str(video.get("video_id","")); nf=int(video.get("num_frames",0))
        for tr in video.get("trajectory_motion_evidence",[]):
            obs=sorted(copy.deepcopy(tr.get("trajectory_observations",[])),
                       key=lambda x:int(x.get("frame_index",-1)))
            stats=dict(tr.get("trajectory_statistics",{})); unc=dict(tr.get("uncertainty",{}))
            start=list(obs[0].get("position_3d",[])) if obs else []
            end=list(obs[-1].get("position_3d",[])) if obs else []
            dx=(f(end[0])-f(start[0])) if len(start)>=3 and len(end)>=3 else 0
            dz=(f(end[2])-f(start[2])) if len(start)>=3 and len(end)>=3 else 0
            direction=("rightward" if dx>0 else "leftward") if abs(dx)>abs(dz) else ("receding" if dz>0 else "approaching")
            out.append({"video_id":vid,"track_id":int(tr.get("track_id",-1)),
                "object_class":str(tr.get("primary_label","unknown")),
                "label_counts":dict(stats.get("label_counts",{})),
                "position":{"start":start,"end":end,"path_length_xz":f(stats.get("path_length_xz",0))},
                "bbox_size":dict(stats.get("bbox_area",{})),
                "relative_motion":dict(stats.get("rel_speed",{})),"direction":direction,
                "persistence":f(stats.get("temporal_coverage_in_video",len(obs)/max(1,nf))),
                "confidence":f(unc.get("confidence_score",0)),
                "provenance":copy.deepcopy(tr.get("provenance",{})),"observations":obs,
                "source_validation":copy.deepcopy(tr.get("causal_motion_fact_validation",{})),
                "source_decision":str(tr.get("fact_decision_status",""))})
    return out

def statistics_summary(table, object_class=None):
    rows=[row for row in table.get("rows",[]) if object_class is None or row.get("object_class")==object_class]
    buckets=defaultdict(lambda:{"means":[],"success":[],"samples":0})
    for row in rows:
        key=(str(row.get("trajectory_pattern","")),str(row.get("residual_type","")))
        buckets[key]["means"].append(f(row.get("mean",0)))
        buckets[key]["success"].append(f(row.get("repair_success_rate",0)))
        buckets[key]["samples"]+=int(row.get("sample_count",0))
    compact=[{"trajectory_pattern":key[0],"residual_type":key[1],
              "mean":mean(value["means"]),"repair_success_rate":mean(value["success"]),
              "sample_count":value["samples"]} for key,value in sorted(buckets.items())]
    return {"version":table.get("version",0),"validation_metrics":table.get("validation_metrics",{}),
            "object_class":object_class,"aggregates":compact[:120]}

def pattern_prompt(tracks,table):
    class_counts=Counter(str(track.get("object_class","unknown")) for track in tracks)
    return ("Define all trajectory patterns using only IDs "+",".join(PATTERNS)+
      " and required_metrics from "+",".join(RESIDUALS)+
      ". Give qualitative_constraints and justification; no numerical thresholds or decisions. "
      "Patterns are offline definitions and must not depend on individual tracks. "
      'JSON schema: {"patterns":[{"pattern_id":"stationary","required_metrics":["speed"],'
      '"qualitative_constraints":["..."],"justification":"..."}]}. statistics_summary='+
      json.dumps(statistics_summary(table),separators=(",",":"))+
      " object_class_counts="+json.dumps(class_counts,separators=(",",":")))

def validate_patterns(raw):
    supplied={str(x.get("pattern_id","")):x for x in raw.get("patterns",[]) if isinstance(x,dict)}
    out=[]
    for pid in PATTERNS:
        row=supplied.get(pid,{})
        metrics=[str(x) for x in row.get("required_metrics",[]) if str(x) in RESIDUALS]
        out.append({"pattern_id":pid,"required_metrics":metrics or list(RESIDUALS),
          "qualitative_constraints":[str(x)[:200] for x in row.get("qualitative_constraints",[])],
          "justification":str(row.get("justification","deterministic completion"))[:400],
          "source":"llm_validated" if row else "deterministic_completion"})
    return out

def series(track):
    xs=[];zs=[];speeds=[];angles=[];egores=[]
    for o in track["observations"]:
        p=list(o.get("position_3d",[])); m=dict(o.get("motion",{}))
        if len(p)>=3: xs.append(f(p[0]));zs.append(f(p[2]))
        vx=f(m.get("rel_vx",0));vz=f(m.get("rel_vz",0));s=math.hypot(vx,vz)
        speeds.append(s)
        if s>1e-6: angles.append(math.atan2(vx,vz))
        ox=f(m.get("obj_vx",0));oz=f(m.get("obj_vz",0));ex=f(m.get("ego_vx",0));ez=f(m.get("ego_vz",0))
        egores.append(min(math.hypot(ox-ex,oz-ez),math.hypot(ox+ex,oz+ez)))
    return xs,zs,speeds,angles,egores

def linear_rmse(values):
    if len(values)<3:return 0.0
    xx=list(range(len(values)));xm=mean(xx);ym=mean(values)
    slope=sum((x-xm)*(y-ym) for x,y in zip(xx,values))/max(1e-9,sum((x-xm)**2 for x in xx))
    return math.sqrt(mean([(y-(ym+slope*(x-xm)))**2 for x,y in zip(xx,values)]))

def residual(pid,track):
    xs,zs,speeds,angles,egores=series(track); obs=track["observations"]
    disp_x=abs(xs[-1]-xs[0]) if len(xs)>1 else 0; disp_z=zs[-1]-zs[0] if len(zs)>1 else 0
    ds=0 if not angles else 1-math.hypot(mean([math.sin(x) for x in angles]),mean([math.cos(x) for x in angles]))
    accel=[abs(b-a) for a,b in zip(speeds,speeds[1:])]
    gaps=[int(b.get("frame_index",0))-int(a.get("frame_index",0)) for a,b in zip(obs,obs[1:])]
    d2=[abs(zs[i+1]-2*zs[i]+zs[i-1]) for i in range(1,len(zs)-1)]
    closing=median([-f(dict(o.get("motion",{})).get("rel_vz",0)) for o in obs])
    ttc=abs(zs[-1])/closing if zs and closing>1e-6 else 1e6
    penalty={"stationary":mean(speeds)+disp_x+abs(disp_z),
      "same_direction":ds+max(0,-disp_z),"opposite_direction":ds+max(0,disp_z),
      "approaching":max(0,disp_z)+(0 if closing>0 else mean(speeds)),
      "receding":max(0,-disp_z)+(0 if disp_z>0 else mean(speeds)),
      "crossing":abs(disp_z)/max(1e-6,disp_x+abs(disp_z)),
      "turning":max(0,.5-ds),"lane_entry":abs(xs[-1]) if xs else 0,
      "overtaking":max(0,abs(disp_z)-disp_x),"unknown":0}[pid]
    return {"position":linear_rmse(xs)+linear_rmse(zs)+penalty,"direction":ds+penalty,
      "speed":statistics.pstdev(speeds) if len(speeds)>1 else 0,
      "acceleration":mean(accel),"path_intersection":min(map(abs,xs)) if xs else 0,
      "ttc":0 if pid=="approaching" and ttc<1e6 else min(100,ttc)/100,
      "continuity":mean([max(0,x-1) for x in gaps]),"depth_consistency":mean(d2),
      "ego_motion_consistency":mean(egores)}

def interpretation_prompt(track,candidates,table):
    public={key:track.get(key) for key in ("video_id","track_id","object_class","position","bbox_size",
      "relative_motion","direction","persistence","confidence","provenance","source_validation","source_decision")}
    compact_candidates=[{"pattern_id":row.get("pattern_id"),"residual_vector":row.get("residual_vector",{})}
                        for row in candidates]
    return ("Interpret every pattern residual using only facts, provenance, confidence and statistics. "
      "Return plausibility [0,1], ignorable_errors, structural_conflicts, recommended_repairs from "+
      ",".join(REPAIRS)+". No corrected values, thresholds or final decisions. "+
      '{"assessments":[{"pattern_id":"stationary","plausibility":0.5,'
      '"ignorable_errors":[],"structural_conflicts":[],"recommended_repairs":["kalman_smoothing"],'
      '"explanation":"..."}]}. track='+json.dumps(public,separators=(",",":"))+
      " candidates="+json.dumps(compact_candidates,separators=(",",":"))+
      " statistics_summary="+json.dumps(statistics_summary(table,track.get("object_class")),separators=(",",":")))

def assessments(raw,candidates):
    src={str(x.get("pattern_id","")):x for x in raw.get("assessments",[]) if isinstance(x,dict)}
    out=[]
    for c in candidates:
        x=src.get(c["pattern_id"],{})
        out.append({"pattern_id":c["pattern_id"],"plausibility":max(0,min(1,f(x.get("plausibility",0)))),
          "ignorable_errors":[str(v)[:200] for v in x.get("ignorable_errors",[])],
          "structural_conflicts":[str(v)[:200] for v in x.get("structural_conflicts",[])],
          "recommended_repairs":[str(v) for v in x.get("recommended_repairs",[]) if str(v) in REPAIRS],
          "explanation":str(x.get("explanation",""))[:600]})
    return out

def ego_frames(video):
    return {int(x.get("frame_index",i)):(f(x.get("refined_ego_vx",x.get("ego_vx_smoothed",x.get("ego_vx",0)))),
      f(x.get("refined_ego_vz",x.get("ego_vz_smoothed",x.get("ego_vz",0))))) for i,x in enumerate(video.get("frames",[]))}

def _residual_score(vector):
    return sum(f(value) for value in vector.values())

def select_pattern_hypotheses(candidates,interp,top_k=5):
    """Retain LLM, deterministic residual, diversity, and unknown hypotheses."""
    by_pattern={row["pattern_id"]:row for row in candidates}
    llm_by_pattern={row["pattern_id"]:row for row in interp}
    llm_ranked=sorted(interp,key=lambda row:(-f(row["plausibility"]),row["pattern_id"]))
    residual_ranked=sorted(candidates,key=lambda row:(_residual_score(row["residual_vector"]),row["pattern_id"]))
    selected=[]
    def add(pattern_id,source):
        if pattern_id not in by_pattern:return
        existing=next((row for row in selected if row["pattern_id"]==pattern_id),None)
        if existing:
            if source not in existing["selection_sources"]:existing["selection_sources"].append(source)
            return
        prior=llm_by_pattern.get(pattern_id,{})
        selected.append({"pattern_id":pattern_id,"selection_sources":[source],
          "llm_prior":f(prior.get("plausibility",0)),
          "recommended_repairs":list(prior.get("recommended_repairs",[]))})
    for row in llm_ranked[:2]:add(row["pattern_id"],"llm_top_k")
    llm_top_ids={row["pattern_id"] for row in llm_ranked[:2]}
    baseline=next((row for row in residual_ranked if row["pattern_id"]!="unknown" and row["pattern_id"] not in llm_top_ids),None)
    if baseline:add(baseline["pattern_id"],"minimum_residual_baseline")
    add("unknown","mandatory_unknown_baseline")
    for row in residual_ranked:
        if len(selected)>=top_k:break
        add(row["pattern_id"],"residual_diversity")
    return selected

def _physical_validity(observations):
    if len(observations)<2:return False
    for row in observations:
        position=list(row.get("position_3d",[]))
        if len(position)<3:return False
        try: values=[float(value) for value in position]
        except (TypeError,ValueError):return False
        if not all(math.isfinite(value) for value in values) or values[2]<0:return False
    return True

def repair_candidates(track,candidates,interp,ego,current_table):
    original=copy.deepcopy(track["observations"]); nf=max([int(x.get("frame_index",0)) for x in original] or [0])+1
    before_eval=_evaluate(original,nf); bypid={x["pattern_id"]:x["residual_vector"] for x in candidates}
    hypotheses=select_pattern_hypotheses(candidates,interp); out=[]
    for hypothesis in hypotheses:
        pid=hypothesis["pattern_id"]
        ops=list(dict.fromkeys(list(hypothesis["recommended_repairs"])+list(DEFAULT_REPAIRS[pid])))[:5]
        for op in ops:
            repaired=copy.deepcopy(original)
            if op=="ego_motion_refinement":
                repaired=_recompute_motion(repaired,ego); executor="ego_motion_refinement";params={"source":"current_ego"}
            else:
                executor,params=EXECUTORS[op]; repaired=_apply_strategy(repaired,executor,params)
                repaired=_recompute_motion(repaired,ego,velocity_window=int(params.get("window",1)))
            aftertrack={**track,"observations":repaired}
            allafter={p["pattern_id"]:residual(p["pattern_id"],aftertrack) for p in candidates}
            bv=bypid[pid];av=allafter[pid]
            improvement=(_residual_score(bv)-_residual_score(av))/max(1e-6,_residual_score(bv))
            ev=_evaluate(repaired,nf)
            retention=len({int(x["frame_index"]) for x in repaired}&{int(x["frame_index"]) for x in original})/max(1,len(original))
            new=sorted(set(ev["validation"].get("rejection_reasons",[]))-set(before_eval["validation"].get("rejection_reasons",[])))
            labels={str(x.get("frame_label","")).strip().lower() for x in repaired if str(x.get("frame_label","")).strip()}
            classes=bool(labels) and labels=={str(track.get("object_class","unknown")).strip().lower()}
            status=str(ev["validation"].get("validation_status","invalid"))
            hard={"physical_validity":_physical_validity(repaired),"observation_retention":retention>=.95,
              "class_consistency":classes,"no_critical_new_anomalies":not new,
              "acceptable_validation_severity":status in {"valid","repaired","uncertain"}}
            passed=all(hard.values())
            matching=[row for row in current_table.get("rows",[]) if row.get("object_class")==track["object_class"] and row.get("trajectory_pattern")==pid]
            stat_prior=mean([f(row.get("repair_success_rate",0)) for row in matching]) if matching else 0.5
            sample_count=sum(int(row.get("sample_count",0)) for row in matching)
            calibration_confidence=min(1.0,math.log1p(sample_count)/math.log(21.0))
            llm_prior=f(hypothesis["llm_prior"])
            score=(.40*max(-1,min(1,improvement))+.25*stat_prior+
                   .20*calibration_confidence+.15*llm_prior) if passed else None
            out.append({"candidate_id":pid+":"+op,"pattern_id":pid,
              "pattern_hypothesis":{"pattern_id":pid,"selection_sources":hypothesis["selection_sources"]},
              "repair_hypothesis":{"operation":op,"executor":executor,"parameters":params},
              "validated_pattern":pid if passed else None,"LLM_prior":llm_prior,
              "llm_plausibility":llm_prior,"repair_operation":op,"executor":executor,
              "parameters":params,"decision":"accept" if passed else "reject","symbolic_verdict":"pass" if passed else "reject",
              "hard_constraint_results":hard,"final_score":score,"combined_score":score,
              "pre_pattern_scores":copy.deepcopy(bypid),"post_repair_pattern_scores":allafter,
              "residual_vector_before":bv,"residual_vector_after":av,"all_pattern_residuals_after":allafter,
              "residual_improvement":improvement,"issue_cost_before":_issue_cost(before_eval),
              "issue_cost_after":_issue_cost(ev),"observation_retention":retention,
              "class_consistent":classes,"new_anomalies":new,"modified_frame_ids":_modified_frames(original,repaired),
              "statistical_success_prior":stat_prior,"calibration_confidence":calibration_confidence,
              "signals_before":_snapshot(original),"signals_after":_snapshot(repaired),
              "symbolic_predicates_after":{"direction":("rightward" if repaired[-1].get("position_3d",[0])[0]>repaired[0].get("position_3d",[0])[0] else "leftward") if repaired else "unknown","persistence":len(repaired)/max(1,nf),"confidence":ev["uncertainty"].get("confidence_score",0),"validation_status":status},
              "validation":ev,"_observations":repaired})
    return out

def stat_rows(dataset,records,version):
    buckets=defaultdict(lambda:{"v":[],"a":0,"r":0,"llm":[]})
    for rec in records:
        cls=rec["symbolic_track"]["object_class"]; decisions=defaultdict(list)
        for c in rec["candidate_repairs"]:decisions[c["pattern_id"]].append(c["decision"])
        llm={x["pattern_id"]:x for x in rec["llm_residual_interpretation"]}
        for p in rec["pattern_candidates"]:
            for rt,val in p["residual_vector"].items():
                b=buckets[(rec["video_id"],cls,p["pattern_id"],rt)];b["v"].append(f(val))
                b["a"]+=sum(x=="accept" for x in decisions[p["pattern_id"]]);b["r"]+=sum(x=="reject" for x in decisions[p["pattern_id"]])
                if p["pattern_id"] in llm:b["llm"].append(llm[p["pattern_id"]])
    out=[]
    for (vid,cls,pid,rt),b in sorted(buckets.items()):
        vals=b["v"]; total=b["a"]+b["r"]
        out.append({"dataset":dataset,"video":vid,"object_class":cls,"trajectory_pattern":pid,
          "residual_type":rt,"sample_count":len(vals),"mean":mean(vals),
          "std":statistics.pstdev(vals) if len(vals)>1 else 0,"median":median(vals),
          "quantiles":{"q05":quantile(vals,.05),"q25":quantile(vals,.25),"q50":quantile(vals,.5),
                       "q75":quantile(vals,.75),"q95":quantile(vals,.95)},
          "accepted_count":b["a"],"rejected_count":b["r"],"repair_success_rate":b["a"]/max(1,total),
          "false_match_rate":sum(rec["final_pattern"]==pid and not rec["repair_applied"] for rec in records)/max(1,len(records)),
          "LLM_assessment":b["llm"][-1] if b["llm"] else {},"version":version})
    return out

def partition_records(records):
    videos=sorted({str(row["video_id"]) for row in records},key=lambda value:hashlib.sha256(value.encode()).hexdigest())
    if len(videos)<2:return list(records),[],videos,[]
    validation_count=max(1,int(round(len(videos)*0.2)))
    validation_videos=set(videos[-validation_count:]);update_videos=set(videos)-validation_videos
    return ([row for row in records if row["video_id"] in update_videos],
            [row for row in records if row["video_id"] in validation_videos],
            sorted(update_videos),sorted(validation_videos))

def evaluate_candidate_table(records, candidate_rows):
    """Re-rank only validation-video candidates using statistics learned elsewhere."""
    evaluated=[]
    for source in records:
        record=copy.deepcopy(source);object_class=record["symbolic_track"]["object_class"]
        passing=[]
        for candidate in record.get("candidate_repairs",[]):
            if candidate.get("symbolic_verdict")!="pass":continue
            matching=[row for row in candidate_rows if row.get("object_class")==object_class and row.get("trajectory_pattern")==candidate.get("pattern_id")]
            stat_prior=mean([f(row.get("repair_success_rate",0)) for row in matching]) if matching else 0.5
            sample_count=sum(int(row.get("sample_count",0)) for row in matching)
            calibration=min(1.0,math.log1p(sample_count)/math.log(21.0))
            score=(.40*max(-1,min(1,f(candidate.get("residual_improvement",0))))+
                   .25*stat_prior+.20*calibration+.15*f(candidate.get("LLM_prior",0)))
            candidate["candidate_table_statistical_success_prior"]=stat_prior
            candidate["candidate_table_calibration_confidence"]=calibration
            candidate["candidate_table_final_score"]=score
            passing.append(candidate)
        selected=max(passing,key=lambda row:f(row.get("candidate_table_final_score"),-1e9),default=None)
        if selected:
            record["selected_candidate"]=selected;record["repair_applied"]=True
            record["validated_pattern"]=selected["pattern_id"];record["final_pattern"]=selected["pattern_id"]
            record["final_validation_status"]=selected.get("validation",{}).get("validation",{}).get("validation_status","invalid")
            record["record_status"]="completed_validated"
        else:
            record["selected_candidate"]={};record["repair_applied"]=False
            record["validated_pattern"]="unknown";record["final_pattern"]="unknown"
            record["record_status"]="completed_unresolved"
        evaluated.append(record)
    return evaluated
def validation_metrics(records):
    before=sum(row["symbolic_track"]["source_validation"].get("validation_status")=="invalid" for row in records)
    after=sum(row["final_validation_status"]=="invalid" for row in records)
    retention=mean([f(row.get("selected_candidate",{}).get("observation_retention",1),1) for row in records]) if records else 0.0
    improve=mean([f(row.get("selected_candidate",{}).get("residual_improvement",0)) for row in records]) if records else 0.0
    accepted=sum(row["repair_applied"] for row in records)
    return {"validation_video_count":len({row["video_id"] for row in records}),"track_count":len(records),
      "accepted_repairs":accepted,"invalid_before":before,"invalid_after":after,
      "mean_observation_retention":retention,"mean_residual_improvement":improve,
      "overall_score":(before-after)*2+accepted+improve+retention}

def review_prompt(current,candidate,metrics):
    return ("Review residual interpretation priorities and pattern assumptions. Return candidate_update "
      "with rationale, residual_priority, pattern_hypotheses, critical_regressions. Do not change "
      "thresholds, overwrite current configuration, or approve promotion. current_summary="+
      json.dumps(statistics_summary(current),separators=(",",":"))+
      " candidate_summary="+json.dumps(statistics_summary(candidate),separators=(",",":"))+
      " metrics="+json.dumps(metrics,separators=(",",":")))

def promote(root,candidate,review,metrics):
    root.mkdir(parents=True,exist_ok=True);current_path=root/"current_table.json"
    current=json.loads(current_path.read_text()) if current_path.exists() else {}
    current_score=f(current.get("validation_metrics",{}).get("overall_score",-1e9),-1e9)
    independent=bool(candidate.get("update_video_ids")) and bool(candidate.get("validation_video_ids")) and not set(candidate["update_video_ids"])&set(candidate["validation_video_ids"])
    no_reg=(metrics["track_count"]>0 and metrics["invalid_after"]<=metrics["invalid_before"] and
            metrics["mean_observation_retention"]>=.95 and
            not review.get("candidate_update",{}).get("critical_regressions",[]))
    improved=(not current and no_reg) or (bool(current) and metrics["overall_score"]>current_score and no_reg)
    ok=independent and improved
    if not independent:reason="independent_validation_split_unavailable"
    elif not no_reg:reason="validation_regression"
    elif not improved:reason="metrics_did_not_improve"
    else:reason="metrics_improved_without_regression"
    decision={"promoted":ok,"decision":"accept" if ok else "rollback","candidate_score":metrics["overall_score"],
      "current_score":current_score,"no_critical_regression":no_reg,"independent_split":independent,
      "update_video_ids":candidate.get("update_video_ids",[]),"validation_video_ids":candidate.get("validation_video_ids",[]),
      "reason":reason}
    (root/f"candidate_table_v{candidate['version']:04d}.json").write_text(json.dumps(candidate,indent=2))
    if ok:current_path.write_text(json.dumps({**candidate,"status":"current","validation_metrics":metrics,"LLM_review":review,"promotion":decision},indent=2))
    (root/f"promotion_decision_v{candidate['version']:04d}.json").write_text(json.dumps(decision,indent=2))
    return decision

def run_trajectory_pattern_closed_loop(state,output_root,*,dataset="driving_mini",llm_generate=None):
    root=Path(output_root);root.mkdir(parents=True,exist_ok=True); audit=root/"llm_audit";statsroot=root/"statistics"
    current=json.loads((statsroot/"current_table.json").read_text()) if (statsroot/"current_table.json").exists() else {}
    tracks=symbolic_tracks(state.get("trajectory_motion_evidence",[]))
    patterns=validate_patterns({})
    ego={str(x.get("video_id","")):ego_frames(x) for x in state.get("ego_motion",[])}
    from src.exp_july.perception.trajectory_pattern_llm_batch import compact_track
    from src.exp_july.perception.trajectory_pattern_epoch import (
        begin_epoch, compile_patch, deterministic_interpretation, evaluate_and_stage,
        fixed_video_split, review_package, review_prompt as epoch_review_prompt,
    )
    from src.exp_july.perception.trajectory_pattern_runtime_monitor import (
        Step8CRuntimeMonitor,
    )
    track_items=[]
    for track in tracks:
        candidates=[{"pattern_id":p["pattern_id"],"pattern_definition":p,
                     "residual_vector":residual(p["pattern_id"],track)} for p in patterns]
        track_items.append({"track":track,"candidates":candidates})
    policy_root=root/"policies"
    epoch_id,frozen_policy,epoch_snapshot=begin_epoch(policy_root)
    review_interval=max(1,int(os.environ.get("CAUVID_STEP8C_REVIEW_INTERVAL_TRACKS","500")))
    runtime_monitor=Step8CRuntimeMonitor(
        root/"runtime_monitor",len(track_items),
        rolling_window=max(5,int(os.environ.get("CAUVID_STEP8C_MONITOR_WINDOW","50"))),
    )
    interpretations={};llm_telemetry={};compact_inputs={}
    for item in track_items:
        uid=compact_track(item["track"],item["candidates"])["track_uid"]
        compact_inputs[uid]=compact_track(item["track"],item["candidates"])
        interpretations[uid]=deterministic_interpretation(item["candidates"],frozen_policy)
        llm_telemetry[uid]={"track_uid":uid,"needs_llm_review":False,"gate_reasons":[],
          "llm_called":False,"llm_skipped":True,"cache_hit":False,"heuristic_fallback":False,
          "timeout_occurred":False,"llm_backend":"frozen_epoch_policy","llm_model":f"policy_v{frozen_policy['version']}",
          "batch_size":0,"retry_count":0,"escalated_to_single":False,"latency":0.0,
          "estimated_token_cost":0.0,"validation_outcome":"deterministic_epoch_policy"}
        runtime_monitor.interpretation_complete(llm_telemetry[uid],compact_inputs[uid])
    records=[];material=defaultdict(list)
    for track_index,item in enumerate(track_items,1):
        track=item["track"];candidates=item["candidates"]
        runtime_monitor.track_start(track["video_id"],track["track_id"],track_index,len(track_items))
        uid=compact_track(track,candidates)["track_uid"]
        interp=interpretations[uid]
        repairs=repair_candidates(track,candidates,interp,ego.get(track["video_id"],{}),current)
        accepted=[row for row in repairs if row["symbolic_verdict"]=="pass"]
        selected=max(accepted,key=lambda row:f(row["final_score"],-1e9)) if accepted else None
        reason="highest_ranked_after_hard_constraints" if selected else "no_candidate_passed_hard_constraints_original_preserved"
        for row in repairs:
            if selected and row["candidate_id"]==selected["candidate_id"]:
                row["final_selection_reason"]="selected_highest_final_score_after_hard_constraints"
            elif row["symbolic_verdict"]=="pass":
                row["final_selection_reason"]="eligible_but_lower_final_score"
            else:
                failed=[name for name,value in row["hard_constraint_results"].items() if not value]
                row["final_selection_reason"]="hard_constraints_failed:"+",".join(failed)
        applied=selected is not None; finalobs=selected["_observations"] if applied else track["observations"]
        finaltrack={**track,"observations":finalobs}; finaleval=_evaluate(finalobs,max([int(x.get("frame_index",0)) for x in finalobs]or[0])+1)
        llm_preferred=max(interp,key=lambda row:f(row["plausibility"])) ["pattern_id"]
        finalpattern=selected["validated_pattern"] if selected else "unknown"
        publicrep=[{k:v for k,v in x.items() if k!="_observations"} for x in repairs]
        publicsel={k:v for k,v in selected.items() if k!="_observations"} if selected else {}
        rec={"version":VERSION,"video_id":track["video_id"],"track_id":track["track_id"],
          "symbolic_track":{k:v for k,v in track.items() if k!="observations"},"pattern_candidates":candidates,
          "llm_residual_interpretation":interp,"llm_compact_input":compact_inputs[uid],
          "llm_processing":llm_telemetry[uid],"candidate_repairs":publicrep,"selected_candidate":publicsel,
          "repair_applied":applied,"pattern_hypothesis":selected.get("pattern_hypothesis",{}) if selected else {},
          "repair_hypothesis":selected.get("repair_hypothesis",{}) if selected else {},
          "validated_pattern":finalpattern,"final_pattern":finalpattern,"LLM_preferred_pattern":llm_preferred,
          "resolution_status":"validated" if selected else "unresolved_uncertain",
          "final_pattern_candidates":[{**c,"residual_vector":residual(c["pattern_id"],finaltrack)} for c in candidates],
          "final_validation_status":finaleval["validation"].get("validation_status",""),
          "final_symbolic_predicates":{"direction":finaltrack["direction"],"confidence":finaleval["uncertainty"].get("confidence_score",0),
                                      "persistence":len(finalobs)/max(1,len(track["observations"]))},
          "record_status":"completed_validated" if selected else "completed_unresolved",
          "final_selection_reason":reason,"provenance":{"source_step":state.get("trajectory_motion_evidence_phase","repaired"),
          "llm_role":"interval_policy_review_only","numeric_repair_role":"deterministic_executor",
          "epoch_id":epoch_id,"frozen_policy_version":frozen_policy["version"],"original_observations_preserved":True}}
        records.append(rec);material[track["video_id"]].append({"track_id":track["track_id"],
          "_repair_selected":applied,"_final_observations":finalobs,
          "modified_frame_ids":selected.get("modified_frame_ids",[]) if selected else []})
        td=root/"tracks"/track["video_id"];td.mkdir(parents=True,exist_ok=True)
        (td/f"track_{track['track_id']:04d}.json").write_text(json.dumps(rec,indent=2))
        runtime_monitor.track_complete(rec)
    original=list(state.get("relative_object_motion",[]))
    refined=[_materialize_repaired_relative_video(v,material[str(v.get("video_id",""))]) for v in original]
    version=int(current.get("version",0))+1
    update_video_ids,validation_video_ids=fixed_video_split([row["video_id"] for row in records])
    update_set=set(update_video_ids);validation_set=set(validation_video_ids)
    update_records=[row for row in records if row["video_id"] in update_set]
    validation_records=[row for row in records if row["video_id"] in validation_set]
    completed_update=[row for row in update_records if row.get("record_status")=="completed_validated"]
    candidate_rows=stat_rows(dataset,completed_update,version)
    validation_evaluated=evaluate_candidate_table(validation_records,candidate_rows)
    metrics=validation_metrics(validation_evaluated)
    candidate={"dataset":dataset,"version":version,"status":"candidate","schema_version":VERSION,
               "rows":candidate_rows,"validation_metrics":metrics,
               "update_video_ids":update_video_ids,"validation_video_ids":validation_video_ids,
               "num_completed_validated_update_records":len(completed_update),
               "num_independent_validation_records":len(validation_evaluated)}
    statsroot.mkdir(parents=True,exist_ok=True)
    (statsroot/f"candidate_table_v{version:04d}.json").write_text(json.dumps(candidate,indent=2))
    review_root=root/"epoch_reviews";review_root.mkdir(parents=True,exist_ok=True)
    reviews=[];promotion_decisions=[]
    for offset in range(0,len(update_records),review_interval):
        interval_index=offset//review_interval+1;interval_records=update_records[offset:offset+review_interval]
        package=review_package(interval_records,epoch_id,interval_index,frozen_policy,validation_video_ids)
        package_path=review_root/f"epoch_{epoch_id:04d}_interval_{interval_index:04d}_package.json"
        package_path.write_text(json.dumps(package,indent=2))
        response=llm_call("policy_interval_review",epoch_review_prompt(package),audit,llm_generate)
        review_row={"interval_index":interval_index,"package":package,"response":response}
        try:
            candidate_policy=compile_patch(frozen_policy,response)
            candidate_policy["version"]=int(frozen_policy["version"])+interval_index
            decision=evaluate_and_stage(policy_root,frozen_policy,candidate_policy,validation_records,response)
        except (TypeError,ValueError,KeyError) as exc:
            decision={"promoted":False,"decision":"reject","reason":"patch_compilation_failed","error":str(exc),
                      "active_policy_version":frozen_policy["version"],"activation_epoch":None}
        review_row["promotion"]=decision;reviews.append(review_row);promotion_decisions.append(decision)
        (review_root/f"epoch_{epoch_id:04d}_interval_{interval_index:04d}_review.json").write_text(json.dumps(review_row,indent=2))
    promotion={"promoted":any(row.get("promoted") for row in promotion_decisions),
      "decision":"stage_for_next_epoch" if any(row.get("promoted") for row in promotion_decisions) else "rollback",
      "reason":next((row.get("reason") for row in reversed(promotion_decisions) if row.get("promoted")),
                    promotion_decisions[-1].get("reason") if promotion_decisions else "no_update_records"),
      "epoch_id":epoch_id,"active_policy_version":frozen_policy["version"],
      "activation_epoch":"next_epoch" if any(row.get("promoted") for row in promotion_decisions) else None,
      "interval_decisions":promotion_decisions,"update_video_ids":update_video_ids,"validation_video_ids":validation_video_ids,
      "independent_split":bool(update_video_ids and validation_video_ids and not set(update_video_ids)&set(validation_video_ids))}
    epoch_snapshot.update({"status":"completed","review_interval_tracks":review_interval,
      "review_count":len(reviews),"promotion":promotion})
    (policy_root/f"epoch_{epoch_id:04d}.json").write_text(json.dumps(epoch_snapshot,indent=2))
    runtime_state=runtime_monitor.finalize()
    manifest={"version":VERSION,"method":"deterministic_epoch_trajectory_pattern_repair",
      "execution_flow":["epoch_boundary_activation","freeze_versioned_policy","symbolic_abstraction",
      "deterministic_policy_interpretation","multi_hypothesis_repair","symbolic_validation",
      "interval_statistical_aggregation","single_llm_policy_patch_review","compile_candidate_policy",
      "fixed_split_validation","stage_for_next_epoch_or_reject"],
      "num_videos":len({x["video_id"] for x in records}),"num_tracks":len(records),
      "num_patterns":len(patterns),"num_candidates":sum(len(x["candidate_repairs"]) for x in records),
      "num_repairs_applied":sum(x["repair_applied"] for x in records),
      "statistics_version":version,"statistics_update_video_ids":update_video_ids,
      "statistics_validation_video_ids":validation_video_ids,"promotion":promotion,"patterns":patterns,
      "epoch_id":epoch_id,"frozen_policy_version":frozen_policy["version"],"policy_frozen":True,
      "review_interval_tracks":review_interval,"interval_review_count":len(reviews),
      "llm_batch_size":0,"llm_called":len(reviews),
      "llm_skipped":sum(row["llm_skipped"] for row in llm_telemetry.values()),
      "llm_cache_hits":sum(row["cache_hit"] for row in llm_telemetry.values()),
      "llm_escalated_to_single":0,"batch_evaluation":{},"runtime_monitor":runtime_state}
    (root/"trajectory_pattern_manifest.json").write_text(json.dumps(manifest,indent=2))
    (root/"symbolic_tracks.json").write_text(json.dumps([{k:v for k,v in x.items() if k!="observations"} for x in tracks],indent=2))
    result={**state,"pre_pattern_relative_object_motion":original,"relative_object_motion":refined,
      "filtered_relative_object_motion":refined,"trajectory_pattern_records":records,
      "trajectory_pattern_definitions":patterns,"trajectory_pattern_statistics_candidate":candidate,
      "trajectory_pattern_statistics_review":reviews,"trajectory_pattern_statistics_promotion":promotion,
      "trajectory_pattern_epoch_policy":frozen_policy,"trajectory_pattern_epoch_reviews":reviews,
      "trajectory_pattern_manifest":manifest,"trajectory_pattern_output_root":root,
      "trajectory_pattern_runtime_monitor":runtime_state,
      "trajectory_pattern_runtime_dashboard_path":runtime_state["runtime_dashboard_path"],
      "trajectory_pattern_runtime_output_root":runtime_state["output_root"]}
    from src.exp_july.perception.trajectory_pattern_visualization import (
        render_trajectory_pattern_visualizations,
    )
    from src.exp_july.perception.trajectory_pattern_dashboard import (
        build_trajectory_pattern_dashboard,
    )
    visualized=render_trajectory_pattern_visualizations(result,root/"visualizations")
    return build_trajectory_pattern_dashboard(visualized,root/"dashboard",audit)
