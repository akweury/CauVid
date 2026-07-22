"""Gated, cached, two-stage batch LLM interpretation for Step 8C."""
from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

PATTERNS=("stationary","same_direction","opposite_direction","approaching","receding","crossing","turning","lane_entry","overtaking","unknown")
RESIDUALS=("position","direction","speed","acceleration","path_intersection","ttc","continuity","depth_consistency","ego_motion_consistency")
REPAIRS=("interpolation","outlier_removal","kalman_smoothing","segment_split","fragment_merge","depth_refinement","ego_motion_refinement","motion_recomputation")
_STAGE1_FIELDS={"track_uid","assessments","requires_repair_planning","batch_confidence","batch_conflicts"}
_ASSESSMENT_FIELDS={"pattern_id","plausibility","ignorable_errors","structural_conflicts","explanation"}
_STAGE2_FIELDS={"track_uid","repair_recommendations"}


def _f(value,default=0.0):
    try:
        result=float(value);return result if math.isfinite(result) else default
    except (TypeError,ValueError):return default


def _total(vector):return sum(_f(value) for value in vector.values())


def _residual_summary(candidates):
    totals={row["pattern_id"]:_total(row.get("residual_vector",{})) for row in candidates}
    values=list(totals.values());lo=min(values or [0]);hi=max(values or [1]);span=max(1e-9,hi-lo)
    ranked=sorted(totals,key=lambda key:(totals[key],key))
    buckets={key:("low" if (totals[key]-lo)/span<.34 else "medium" if (totals[key]-lo)/span<.67 else "high") for key in totals}
    normalized={key:round((totals[key]-lo)/span,4) for key in totals}
    return {"ranked_patterns":ranked,"normalized":normalized,"buckets":buckets}


def compact_track(track,candidates):
    summary=_residual_summary(candidates);validation=dict(track.get("source_validation",{}))
    return {"track_uid":f"{track.get('video_id','')}::track_{int(track.get('track_id',-1))}",
      "track_id":int(track.get("track_id",-1)),"video_id":str(track.get("video_id","")),
      "object_class":str(track.get("object_class","unknown")),
      "symbolic_facts":{"direction":track.get("direction"),"persistence":round(_f(track.get("persistence")),4),
        "confidence":round(_f(track.get("confidence")),4),"bbox_size":track.get("bbox_size"),
        "position":track.get("position"),"relative_motion":track.get("relative_motion")},
      "validation_state":str(validation.get("validation_status",track.get("source_decision","unknown"))),
      "validation_issues":list(validation.get("rejection_reasons",[])) or list(validation.get("issues",[])),
      "provenance":track.get("provenance",{}),"top_candidate_patterns":summary["ranked_patterns"][:4],
      "normalized_residual_summaries":summary["normalized"],"residual_buckets":summary["buckets"],
      "uncertainty_indicators":{"track_confidence":round(_f(track.get("confidence")),4),
        "top_margin":round((summary["normalized"].get(summary["ranked_patterns"][1],1)-summary["normalized"].get(summary["ranked_patterns"][0],0)) if len(summary["ranked_patterns"])>1 else 1,4)}}


def needs_llm_review(compact):
    top=list(compact["top_candidate_patterns"]);unc=compact["uncertainty_indicators"]
    issues=compact["validation_issues"];state=compact["validation_state"].lower()
    reasons=[]
    if _f(unc["track_confidence"])<.8:reasons.append("low_confidence")
    if _f(unc["top_margin"])<.25:reasons.append("ambiguous_pattern_margin")
    if issues:reasons.append("validation_issues")
    if state not in {"valid","keep"}:reasons.append("validation_conflict")
    if top and top[0]=="unknown":reasons.append("unknown_top_pattern")
    return bool(reasons),reasons


def _value_bucket(value):
    value=_f(value)
    return "low" if value<.34 else "medium" if value<.67 else "high"

def _signature(compact):
    facts=compact["symbolic_facts"]
    normalized_facts={"direction":facts.get("direction"),
      "persistence_bucket":_value_bucket(facts.get("persistence")),
      "confidence_bucket":_value_bucket(facts.get("confidence")),
      "bbox_size_type":type(facts.get("bbox_size")).__name__,
      "top_candidate_patterns":compact.get("top_candidate_patterns",[])[:3]}
    normalized={"object_class":compact["object_class"],"symbolic_facts":normalized_facts,
      "validation_state":compact["validation_state"],"validation_issues":compact["validation_issues"],
      "residual_buckets":compact["residual_buckets"]}
    return hashlib.sha256(json.dumps(normalized,sort_keys=True,separators=(",",":"),default=str).encode()).hexdigest()


def deterministic_assessments(candidates):
    summary=_residual_summary(candidates);out=[]
    for pattern in PATTERNS:
        plausibility=max(0.01,1-summary["normalized"].get(pattern,1))
        out.append({"pattern_id":pattern,"plausibility":plausibility,"ignorable_errors":[],
          "structural_conflicts":[],"recommended_repairs":[],"explanation":"deterministic clear-track bypass"})
    return out


def _stage1_prompt(rows):
    return ("Independently interpret each track. Never compare tracks. Return exactly one result for every track_uid. "
      "Do not generate corrected values, thresholds, or final decisions. Allowed top-level result fields: track_uid, "
      "assessments, requires_repair_planning, batch_confidence, batch_conflicts. Each assessment must use pattern_id, "
      "plausibility, ignorable_errors, structural_conflicts, explanation. Include all patterns: "+",".join(PATTERNS)+
      '. JSON only: {"results":[...]}. inputs='+json.dumps(rows,separators=(",",":"),default=str))


def _stage2_prompt(rows):
    return ("Plan repairs independently for each track_uid. Never compare tracks. Use only repair names "+",".join(REPAIRS)+
      ". Do not generate corrected numerical values or final decisions. Return exactly one result per input. "
      'JSON only: {"results":[{"track_uid":"...","repair_recommendations":{"pattern_id":["repair"]}}]}. inputs='+
      json.dumps(rows,separators=(",",":"),default=str))


def _validate_stage1(payload,expected):
    errors={uid:[] for uid in expected};valid={};rows=payload.get("results") if isinstance(payload,dict) else None
    if not isinstance(rows,list):return {},{uid:["malformed_json_schema"] for uid in expected}
    seen=set()
    for row in rows:
        if not isinstance(row,dict):continue
        uid=str(row.get("track_uid",""))
        if uid not in expected:continue
        if uid in seen:errors[uid].append("duplicate_id");continue
        seen.add(uid)
        illegal=set(row)-_STAGE1_FIELDS
        if illegal:errors[uid].append("illegal_fields:"+",".join(sorted(illegal)))
        assessments=row.get("assessments")
        if not isinstance(assessments,list):errors[uid].append("missing_assessments");continue
        by_pattern={}
        for assessment in assessments:
            if not isinstance(assessment,dict):errors[uid].append("malformed_assessment");continue
            if set(assessment)-_ASSESSMENT_FIELDS:errors[uid].append("illegal_assessment_fields")
            pattern=str(assessment.get("pattern_id",""))
            if pattern in by_pattern:errors[uid].append("duplicate_pattern:"+pattern)
            if pattern in PATTERNS:by_pattern[pattern]=assessment
        if set(by_pattern)!=set(PATTERNS):errors[uid].append("incomplete_patterns")
        if not errors[uid]:valid[uid]=row
    for uid in expected:
        if uid not in seen:errors[uid].append("missing_id")
    return valid,{uid:value for uid,value in errors.items() if value}


def _validate_stage2(payload,expected):
    errors={uid:[] for uid in expected};valid={};rows=payload.get("results") if isinstance(payload,dict) else None
    if not isinstance(rows,list):return {},{uid:["malformed_json_schema"] for uid in expected}
    seen=set()
    for row in rows:
        if not isinstance(row,dict):continue
        uid=str(row.get("track_uid",""))
        if uid not in expected:continue
        if uid in seen:errors[uid].append("duplicate_id");continue
        seen.add(uid)
        if set(row)-_STAGE2_FIELDS:errors[uid].append("illegal_fields")
        recommendations=row.get("repair_recommendations")
        if not isinstance(recommendations,dict):errors[uid].append("missing_repair_recommendations")
        else:
            for pattern,ops in recommendations.items():
                if pattern not in PATTERNS or not isinstance(ops,list) or any(op not in REPAIRS for op in ops):errors[uid].append("illegal_repair_recommendation")
        if not errors[uid]:valid[uid]=row
    for uid in expected:
        if uid not in seen:errors[uid].append("missing_id")
    return valid,{uid:value for uid,value in errors.items() if value}


def _invoke_batch(kind,rows,invoke,validator,batch_size,telemetry,event_callback=None):
    output={};pending=list(rows);attempt=0;size=max(1,batch_size)
    while pending and attempt<2:
        next_pending=[]
        for offset in range(0,len(pending),size):
            chunk=pending[offset:offset+size];uids={row["track_uid"] for row in chunk};start=time.perf_counter()
            prompt=_stage1_prompt(chunk) if kind=="stage1" else _stage2_prompt(chunk)
            try:payload=invoke(f"batch_{kind}",prompt);response_text=json.dumps(payload,separators=(",",":"),default=str)
            except (ValueError,json.JSONDecodeError):payload={};response_text="{}"
            elapsed=time.perf_counter()-start;valid,errors=validator(payload,uids);output.update(valid)
            if event_callback:
                event_callback({"stage":kind,"batch_size":len(chunk),"retry_level":attempt,
                  "latency":elapsed,"input_count":len(chunk),"valid_count":len(valid),
                  "failed_count":len(errors),"malformed_count":len(errors),
                  "token_cost":(len(prompt)+len(response_text))/4,"track_uids":sorted(uids)})
            for uid in uids:
                telemetry[uid]["llm_called"]=True;telemetry[uid]["batch_size"]=len(chunk);telemetry[uid]["latency"]+=elapsed/len(chunk)
                telemetry[uid]["estimated_token_cost"]+=(len(prompt)+len(response_text))/4/len(chunk)
                if uid in errors:
                    telemetry[uid]["retry_count"]+=1;telemetry[uid]["validation_outcome"]="failed:"+"|".join(errors[uid])
                    next_pending.append(next(row for row in chunk if row["track_uid"]==uid))
                else:telemetry[uid]["validation_outcome"]="valid"
        pending=next_pending;attempt+=1;size=max(1,size//2)
    for row in pending:
        uid=row["track_uid"];telemetry[uid]["escalated_to_single"]=True
        prompt=_stage1_prompt([row]) if kind=="stage1" else _stage2_prompt([row]);start=time.perf_counter()
        try:payload=invoke(f"{kind}_individual",prompt);response_text=json.dumps(payload,separators=(",",":"),default=str)
        except (ValueError,json.JSONDecodeError):payload={};response_text="{}"
        elapsed=time.perf_counter()-start;valid,errors=validator(payload,{uid})
        telemetry[uid]["latency"]+=elapsed;telemetry[uid]["estimated_token_cost"]+=(len(prompt)+len(response_text))/4
        if event_callback:
            event_callback({"stage":kind+"_individual","batch_size":1,"retry_level":"individual",
              "latency":elapsed,"input_count":1,"valid_count":len(valid),"failed_count":len(errors),
              "malformed_count":len(errors),"token_cost":(len(prompt)+len(response_text))/4,"track_uids":[uid]})
        if uid in valid:output[uid]=valid[uid];telemetry[uid]["validation_outcome"]="valid_after_single"
        else:telemetry[uid]["validation_outcome"]="failed_after_single:"+"|".join(errors.get(uid,["unknown"]))
    return output

def _merge(stage1,stage2,candidates):
    repairs=stage2.get("repair_recommendations",{}) if stage2 else {}
    by_pattern={str(row.get("pattern_id","")):row for row in stage1.get("assessments",[])}
    out=[]
    for candidate in candidates:
        pattern=candidate["pattern_id"];row=by_pattern.get(pattern,{})
        out.append({"pattern_id":pattern,"plausibility":max(0,min(1,_f(row.get("plausibility",0)))),
          "ignorable_errors":[str(value)[:200] for value in row.get("ignorable_errors",[])],
          "structural_conflicts":[str(value)[:200] for value in row.get("structural_conflicts",[])],
          "recommended_repairs":[op for op in repairs.get(pattern,[]) if op in REPAIRS],
          "explanation":str(row.get("explanation",""))[:600]})
    return out


def _unstable(stage1):
    assessments=list(stage1.get("assessments",[]));ranked=sorted(assessments,key=lambda row:-_f(row.get("plausibility",0)))
    confidence=_f(stage1.get("batch_confidence",0));conflicts=len(stage1.get("batch_conflicts",[]))
    return confidence<.55 or conflicts>=2 or not ranked or str(ranked[0].get("pattern_id"))=="unknown" or (len(ranked)>1 and _f(ranked[0].get("plausibility"))-_f(ranked[1].get("plausibility"))<.08)


def process_tracks(items,cache_root,invoke,batch_size=10,use_cache=True,event_callback=None):
    cache_root=Path(cache_root);cache_root.mkdir(parents=True,exist_ok=True);telemetry={};results={};review=[];compacts={}
    for item in items:
        compact=compact_track(item["track"],item["candidates"]);uid=compact["track_uid"];compacts[uid]=compact
        gate,reasons=needs_llm_review(compact);signature=_signature(compact);path=cache_root/f"{signature}.json"
        telemetry[uid]={"track_uid":uid,"needs_llm_review":gate,"gate_reasons":reasons,"llm_called":False,"llm_skipped":not gate,
          "cache_hit":False,"batch_size":0,"retry_count":0,"escalated_to_single":False,"latency":0.0,"estimated_token_cost":0.0,"validation_outcome":"deterministic_bypass"}
        if not gate:results[uid]=deterministic_assessments(item["candidates"]);continue
        if use_cache and path.exists():
            try:cached=json.loads(path.read_text());results[uid]=cached["interpretation"];telemetry[uid].update({"cache_hit":True,"llm_skipped":True,"validation_outcome":"signature_cache_hit"});continue
            except (OSError,json.JSONDecodeError,KeyError):pass
        review.append(compact)
    stage1=_invoke_batch("stage1",review,invoke,_validate_stage1,batch_size,telemetry,event_callback) if review else {}
    escalation=[compacts[uid] for uid,row in stage1.items() if _unstable(row)]
    if escalation:
        escalated=_invoke_batch("stage1",escalation,invoke,_validate_stage1,1,telemetry,event_callback)
        for row in escalation:telemetry[row["track_uid"]]["escalated_to_single"]=True
        stage1.update(escalated)
    planning=[]
    for uid,row in stage1.items():
        if bool(row.get("requires_repair_planning")) or compacts[uid]["validation_issues"] or compacts[uid]["validation_state"].lower() not in {"valid","keep"}:
            planning.append({**compacts[uid],"stage1_assessments":row.get("assessments",[])})
    stage2=_invoke_batch("stage2",planning,invoke,_validate_stage2,batch_size,telemetry,event_callback) if planning else {}
    item_by_uid={compact_track(item["track"],item["candidates"])["track_uid"]:item for item in items}
    for uid,row in stage1.items():
        interpretation=_merge(row,stage2.get(uid,{}),item_by_uid[uid]["candidates"]);results[uid]=interpretation
        signature=_signature(compacts[uid]);(cache_root/f"{signature}.json").write_text(json.dumps({"signature":signature,"interpretation":interpretation},indent=2))
    for uid,item in item_by_uid.items():
        if uid not in results:
            results[uid]=deterministic_assessments(item["candidates"]);telemetry[uid]["validation_outcome"]="llm_failed_deterministic_fallback"
    return results,telemetry,compacts


def evaluate_batch_sizes(items,invoke,output_path):
    sizes=(1,5,10,20);runs={};sample=list(items)[:20]
    for size in sizes:
        start=time.perf_counter();results,telemetry,_=process_tracks(sample,Path(output_path).parent/f"eval_cache_{size}",invoke,size,use_cache=False)
        rankings={uid:max(rows,key=lambda row:_f(row.get("plausibility",0)))["pattern_id"] for uid,rows in results.items()}
        advice={uid:sorted({op for row in rows for op in row.get("recommended_repairs",[])}) for uid,rows in results.items()}
        runs[size]={"rankings":rankings,"advice":advice,"latency":time.perf_counter()-start,
          "malformed_output_rate":sum(not str(row["validation_outcome"]).startswith("valid") for row in telemetry.values())/max(1,len(telemetry)),
          "symbolic_validation_success_rate":sum(str(row["validation_outcome"]).startswith("valid") for row in telemetry.values())/max(1,len(telemetry)),
          "token_cost":sum(_f(row["estimated_token_cost"]) for row in telemetry.values())}
    base=runs[1];summary=[];chosen=1
    for size in sizes:
        run=runs[size];uids=set(base["rankings"])&set(run["rankings"])
        pattern=sum(base["rankings"][uid]==run["rankings"][uid] for uid in uids)/max(1,len(uids))
        repair=sum(base["advice"][uid]==run["advice"][uid] for uid in uids)/max(1,len(uids))
        row={"batch_size":size,"pattern_ranking_agreement":pattern,"repair_advice_agreement":repair,
          **{key:value for key,value in run.items() if key not in {"rankings","advice"}}};summary.append(row)
        if pattern>=.95 and repair>=.9 and row["symbolic_validation_success_rate"]>=.98 and row["malformed_output_rate"]<=.02:chosen=size
    payload={"representative_track_count":len(sample),"single_track_reference_batch_size":1,"results":summary,"chosen_batch_size":chosen}
    Path(output_path).write_text(json.dumps(payload,indent=2));return payload