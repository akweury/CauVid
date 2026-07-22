"""Live runtime monitoring for Step 8C without affecting decisions."""
from __future__ import annotations

import csv
import json
import math
import statistics
import time
from collections import Counter, deque
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm


_HTML="""<!doctype html><html><head><meta charset='utf-8'><meta http-equiv='refresh' content='3'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Step 8C Runtime</title><style>body{margin:0;background:#10151b;color:#edf2f7;font:14px system-ui}header,main{padding:16px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}.card,section{background:#1b232d;border:1px solid #354254;border-radius:8px;padding:12px}.value{font-size:25px;font-weight:700}.good{color:#57dc91}.warn{color:#ffc857}.bad{color:#ff6b76}table{width:100%;border-collapse:collapse}td,th{padding:6px;border-bottom:1px solid #354254;text-align:left}pre{white-space:pre-wrap;max-height:300px;overflow:auto}.bar{height:12px;background:#283342;border-radius:8px}.fill{height:100%;background:#62b5ff;border-radius:8px}</style></head><body><header><h1>Step 8C Live Runtime Monitor</h1><div>Auto-refresh: 3 seconds · read-only · updated __UPDATED__</div><div class='bar'><div class='fill' style='width:__PERCENT__%'></div></div></header><main><div class='grid'>__CARDS__</div><section><h2>Rolling window</h2><pre>__ROLLING__</pre></section><section><h2>Anomaly alerts</h2>__ALERTS__</section><section><h2>Recent qualitative samples</h2>__SAMPLES__</section><section><h2>Recent batch events</h2>__BATCHES__</section></main></body></html>"""


def _now():return datetime.now(timezone.utc).isoformat()


def _safe(value,default=0.0):
    try:return float(value)
    except (TypeError,ValueError):return default


class Step8CRuntimeMonitor:
    def __init__(self,output_root,total_tracks,rolling_window=50,refresh_seconds=3):
        self.root=Path(output_root);self.root.mkdir(parents=True,exist_ok=True)
        self.total_tracks=max(0,int(total_tracks));self.total_units=max(1,2*self.total_tracks)
        self.window=max(5,int(rolling_window));self.refresh_seconds=max(1,int(refresh_seconds))
        self.started=time.time();self.last_write=0.0;self.units=0
        self.counts=Counter();self.latencies=deque(maxlen=self.window);self.outcomes=deque(maxlen=self.window)
        self.batches=deque(maxlen=30);self.samples=deque(maxlen=20);self.alerts=deque(maxlen=50)
        self.csv_path=self.root/"batch_metrics.csv";self.jsonl_path=self.root/"batch_metrics.jsonl"
        self.alert_path=self.root/"anomaly_alerts.jsonl";self.sample_path=self.root/"recent_qualitative_samples.json"
        self.summary_path=self.root/"runtime_summary.json";self.dashboard_path=self.root/"runtime_dashboard.html"
        fields=("timestamp","stage","batch_size","retry_level","latency","input_count","valid_count","failed_count","malformed_count","token_cost","error")
        with self.csv_path.open("w",newline="",encoding="utf-8") as file:csv.DictWriter(file,fieldnames=fields).writeheader()
        self.jsonl_path.touch();self.alert_path.touch()
        self.progress=tqdm(total=self.total_units,desc="[step 8c] live",unit="unit",dynamic_ncols=True)
        self._write(force=True)

    def _rolling(self):
        lat=list(self.latencies);elapsed=max(1e-6,time.time()-self.started);throughput=self.units/elapsed
        remaining=max(0,self.total_units-self.units);eta=remaining/max(1e-9,throughput)
        return {"window_size":len(self.outcomes),"mean_latency":statistics.mean(lat) if lat else 0.0,
          "p95_latency":sorted(lat)[min(len(lat)-1,int(.95*(len(lat)-1)))] if lat else 0.0,
          "throughput_units_per_second":throughput,"eta_seconds":eta,
          "failure_rate":sum(row=="failure" for row in self.outcomes)/max(1,len(self.outcomes)),
          "uncertainty_rate":sum(row=="uncertain" for row in self.outcomes)/max(1,len(self.outcomes))}

    def _postfix(self):
        r=self._rolling();return {"thr":f"{r['throughput_units_per_second']:.2f}/s","eta":f"{r['eta_seconds']:.0f}s",
          "call":self.counts["llm_called"],"skip":self.counts["llm_skipped"],"cache":self.counts["cache_hit"],
          "accept":self.counts["repair_accepted"],"unc":self.counts["uncertain"],"fail":self.counts["failure"],
          "lat":f"{r['mean_latency']:.2f}s"}

    def _alert(self,kind,message,severity="warning",context=None):
        row={"timestamp":_now(),"kind":kind,"severity":severity,"message":message,"context":context or {}}
        self.alerts.appendleft(row)
        with self.alert_path.open("a",encoding="utf-8") as file:file.write(json.dumps(row,default=str)+"\n")

    def handle_batch_event(self,event):
        event={"timestamp":_now(),**dict(event)};latency=_safe(event.get("latency"));self.latencies.append(latency)
        self.batches.appendleft(event);self.counts["batch_events"]+=1
        self.counts["batch_failures"]+=int(event.get("failed_count",0));self.counts["llm_requests"]+=1
        if event.get("error"):self.counts["failure"]+=1;self._alert("llm_request_error",str(event["error"]),"critical",event)
        rolling=list(self.latencies)
        if len(rolling)>=5:
            baseline=statistics.mean(rolling[:-1]);spread=statistics.pstdev(rolling[:-1])
            if latency>max(10.0,baseline+3*max(.01,spread)):self._alert("latency_spike",f"batch latency {latency:.2f}s",context=event)
        if int(event.get("failed_count",0))>0:self._alert("batch_validation_failure",f"{event['failed_count']} failed outputs",context=event)
        exists=self.csv_path.exists()
        fields=("timestamp","stage","batch_size","retry_level","latency","input_count","valid_count","failed_count","malformed_count","token_cost","error")
        with self.csv_path.open("a",newline="",encoding="utf-8") as file:
            writer=csv.DictWriter(file,fieldnames=fields,extrasaction="ignore")
            if not exists:writer.writeheader()
            writer.writerow(event)
        with self.jsonl_path.open("a",encoding="utf-8") as file:file.write(json.dumps(event,default=str)+"\n")
        self.progress.set_postfix(self._postfix(),refresh=True);self._write()

    def interpretation_complete(self,telemetry,compact):
        self.units+=1;self.progress.update(1)
        for key in ("llm_called","llm_skipped","cache_hit","escalated_to_single"):
            self.counts[key]+=bool(telemetry.get(key))
        outcome=str(telemetry.get("validation_outcome",""));failed=outcome.startswith("failed")
        self.outcomes.append("failure" if failed else "ok");self.counts["failure"]+=failed
        latency=_safe(telemetry.get("latency"));self.latencies.append(latency)
        if failed:self._alert("interpretation_failure",outcome,"critical",{"track_uid":telemetry.get("track_uid")})
        self.samples.appendleft({"timestamp":_now(),"phase":"interpretation","track_uid":telemetry.get("track_uid"),
          "object_class":compact.get("object_class"),"top_patterns":compact.get("top_candidate_patterns",[]),
          "gate_reasons":telemetry.get("gate_reasons",[]),"outcome":outcome})
        self.progress.set_postfix(self._postfix(),refresh=True);self._write()

    def track_complete(self,record):
        self.units+=1;self.progress.update(1);accepted=bool(record.get("repair_applied"));uncertain=str(record.get("resolution_status"))!="validated"
        failed=str(record.get("final_validation_status"))=="invalid"
        self.counts["repair_accepted"]+=accepted;self.counts["uncertain"]+=uncertain;self.counts["failure"]+=failed
        self.outcomes.append("failure" if failed else "uncertain" if uncertain else "ok")
        rolling=self._rolling()
        if len(self.outcomes)>=5 and rolling["failure_rate"]>=.2:self._alert("high_failure_rate",f"rolling failure rate {rolling['failure_rate']:.1%}","critical")
        if len(self.outcomes)>=5 and rolling["uncertainty_rate"]>=.4:self._alert("high_uncertainty_rate",f"rolling uncertainty rate {rolling['uncertainty_rate']:.1%}")
        if failed:self._alert("final_validation_failure","final trajectory remains invalid","critical",{"video_id":record.get("video_id"),"track_id":record.get("track_id")})
        if uncertain:self._alert("unresolved_track","no repair candidate passed all constraints",context={"video_id":record.get("video_id"),"track_id":record.get("track_id")})
        self.samples.appendleft({"timestamp":_now(),"phase":"repair_validation","video_id":record.get("video_id"),"track_id":record.get("track_id"),
          "llm_preferred":record.get("LLM_preferred_pattern"),"validated_pattern":record.get("validated_pattern"),
          "repair_applied":accepted,"resolution_status":record.get("resolution_status"),"reason":record.get("final_selection_reason")})
        self.progress.set_postfix(self._postfix(),refresh=True);self._write()

    def _snapshot(self,status="running"):
        return {"version":1,"status":status,"updated_at":_now(),"total_tracks":self.total_tracks,"total_units":self.total_units,
          "completed_units":self.units,"progress":min(1.0,self.units/self.total_units),"elapsed_seconds":time.time()-self.started,
          "counts":dict(self.counts),"rolling":self._rolling(),"recent_batches":list(self.batches),
          "recent_samples":list(self.samples),"alerts":list(self.alerts)}

    def _write(self,force=False,status="running"):
        now=time.time()
        if not force and now-self.last_write<.5:return
        self.last_write=now;snapshot=self._snapshot(status)
        self.summary_path.write_text(json.dumps(snapshot,indent=2,default=str),encoding="utf-8")
        self.sample_path.write_text(json.dumps(list(self.samples),indent=2,default=str),encoding="utf-8")
        cards=[]
        values={"Progress":f"{100*snapshot['progress']:.1f}%","Throughput":f"{snapshot['rolling']['throughput_units_per_second']:.2f}/s",
          "ETA":f"{snapshot['rolling']['eta_seconds']:.0f}s","LLM calls":self.counts["llm_called"],"LLM skips":self.counts["llm_skipped"],
          "Cache hits":self.counts["cache_hit"],"Repair accepted":self.counts["repair_accepted"],"Uncertain":self.counts["uncertain"],
          "Failures":self.counts["failure"],"Rolling latency":f"{snapshot['rolling']['mean_latency']:.2f}s"}
        for label,value in values.items():cards.append(f"<div class='card'><div>{label}</div><div class='value'>{value}</div></div>")
        alerts="".join(f"<div class='card {'bad' if row['severity']=='critical' else 'warn'}'><b>{row['kind']}</b> {row['message']}</div>" for row in snapshot["alerts"][:15]) or "None"
        samples="<pre>"+json.dumps(snapshot["recent_samples"],indent=2,default=str)+"</pre>"
        batches="<pre>"+json.dumps(snapshot["recent_batches"],indent=2,default=str)+"</pre>"
        html=_HTML.replace("__UPDATED__",snapshot["updated_at"]).replace("__PERCENT__",f"{100*snapshot['progress']:.1f}").replace("__CARDS__","".join(cards)).replace("__ROLLING__",json.dumps(snapshot["rolling"],indent=2)).replace("__ALERTS__",alerts).replace("__SAMPLES__",samples).replace("__BATCHES__",batches)
        self.dashboard_path.write_text(html,encoding="utf-8")

    def abort(self,error):
        self.counts["failure"]+=1
        self._alert("step8c_runtime_abort",str(error),"critical")
        self.progress.close();self._write(force=True,status="failed")

    def finalize(self):
        self.progress.n=min(self.total_units,self.units);self.progress.refresh();self.progress.close();self._write(force=True,status="completed")
        return {"output_root":str(self.root),"runtime_dashboard_path":str(self.dashboard_path),"summary_path":str(self.summary_path),
          "batch_csv_path":str(self.csv_path),"batch_jsonl_path":str(self.jsonl_path),"alerts_path":str(self.alert_path),
          "recent_samples_path":str(self.sample_path),**self._snapshot("completed")}