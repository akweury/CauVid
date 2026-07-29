"""Step 7A MP4 audit renderer (display-only hypothesis selection)."""
from pathlib import Path
import math
COLORS={"forward":(70,205,80),"backward":(65,80,235),"static":(145,145,145),"left":(235,190,55),"right":(205,75,205),"straight":(40,210,250),"unavailable":(60,60,60)}
PLOT_COLORS={"forward":"#46cd50","backward":"#eb5041","static":"#919191","left":"#37beeb","right":"#cd4bcd","straight":"#41a5eb","unavailable":"#3c3c3c"}
def _signal(f,a):
 for k in (f"refined_ego_{a}",f"ego_{a}_smoothed",f"ego_{a}"):
  try:v=float(f.get(k))
  except (TypeError,ValueError):continue
  if math.isfinite(v):return v
 return None
def _segments(frames,a,n,labels):
 out=[]; active=None; prev=None
 for i,f in enumerate(frames):
  fi=int(f.get("frame_index",i));v=_signal(f,a)
  if v is None:continue
  state=labels[0] if v < -n else labels[2] if v > n else labels[1]
  if not active or active["state"]!=state or prev is None or fi!=prev+1:
   if active:out.append(active)
   active={"state":state,"start_frame":fi,"end_frame":fi,"duration_frames":1,"signal_sum":float(v)}
  else:active["end_frame"]=fi;active["duration_frames"]+=1;active["signal_sum"]+=float(v)
  prev=fi
 if active:out.append(active)
 for segment in out:segment["mean_signal"]=float(segment.pop("signal_sum")/max(1,int(segment["duration_frames"])))
 return out
def _bridge_kwargs(data):
 config=data.get("noise_filter",{})
 return {
  "bridge_total_max_frames":int(config.get("bridge_total_max_frames",15)),
  "anchor_min_frames":int(config.get("anchor_min_frames",8)),
  "bridge_max_segments":int(config.get("bridge_max_segments",5)),
  "bridge_max_anchor_ratio":float(config.get("bridge_max_anchor_ratio",.75)),
 }

def _all_candidates(result,a,audit):
 key=f"{a}_segmentation";plateaus={int(p["plateau_id"]):p for p in result.get(key,{}).get("qualifying_plateaus",[])}
 points={int(p["plateau_id"]):p for p in audit.get("points",[]) if p.get("axis")==a and str(p.get("video_id",""))==str(result.get("video_id","")) and int(p.get("plateau_id",-1)) in plateaus}
 out=[]
 for plateau_id,plateau in plateaus.items():
  point=points.get(plateau_id,{});enabled=bool(point.get("enabled",False));reasons=list(point.get("disabled_reasons",[]))
  if not enabled and not reasons:reasons=[str(point.get("disabled_reason") or "missing_or_disabled_scatter_audit")]
  out.append({"threshold_n":float(plateau["midpoint_n"]),"confidence":None if point.get("confidence") is None else float(point["confidence"]),"plateau_id":plateau_id,"selection":"qualifying_plateau","display_only":True,"enabled":enabled,"status":"enabled" if enabled else "disabled","disabled_reasons":reasons,"activation_status":"ENABLED" if enabled else "DISABLED","segments":plateau.get("segments",[])})
 return sorted(out,key=lambda row:(row["threshold_n"],row["plateau_id"]))

def _enabled_candidates(result,a,audit):
 return [candidate for candidate in _all_candidates(result,a,audit) if candidate["enabled"]]
def _candidate(result,a,audit):
 key=f"{a}_segmentation"; ps=result.get(key,{}).get("qualifying_plateaus",[])
 enabled=_enabled_candidates(result,a,audit);scored=[row for row in enabled if row.get("confidence") is not None]
 if scored:return max(scored,key=lambda row:(float(row["confidence"]),-float(row["threshold_n"])))
 elif ps:p=max(ps,key=lambda q:(int(q["num_n_values"]),-float(q["midpoint_n"])));conf=None;sel="widest_qualifying_plateau_fallback"
 else:
  ls=result.get(key,{}).get("labels",{}); labels=(ls.get("negative","negative"),ls.get("center","static"),ls.get("positive","positive"))
  from src.exp_july.perception.ego_axis_threshold_segmentation import filter_short_state_interruptions, merge_remaining_short_segments
  tolerance=int(result.get(key,{}).get("noise_filter",{}).get("tolerance_frames",5))
  bridged=filter_short_state_interruptions(_segments(result.get("frames",[]),a,0.,labels),tolerance,**_bridge_kwargs(result.get(key,{})))
  segments=merge_remaining_short_segments(bridged,tolerance)
  return {"threshold_n":0.,"confidence":None,"selection":"zero_threshold_fallback","display_only":True,"segments":segments}
 return {"threshold_n":float(p["midpoint_n"]),"confidence":conf,"plateau_id":int(p["plateau_id"]),"selection":sel,"display_only":True,"segments":p.get("segments",[])}
def _state(c,fi):
 for s in c["segments"]:
  if int(s["start_frame"])<=fi<=int(s["end_frame"]):return str(s["state"])
 return "unavailable"
def _chart(im,box,frames,a,now):
 import cv2,numpy as np
 x,y,w,h=box;cv2.rectangle(im,(x,y),(x+w,y+h),(38,42,49),-1); vals=[_signal(f,a) for f in frames]; finite=[v for v in vals if v is not None];lim=max([abs(v) for v in finite]+[1.])*1.12;l,r,t,b=x+42,x+w-12,y+38,y+h-22;z=(t+b)//2
 for q in range(l,r,20):cv2.line(im,(q,z),(min(q+12,r),z),(0,245,255),2,cv2.LINE_AA)
 pts=[]
 for i,v in enumerate(vals):
  if v is not None:pts.append((l if len(vals)<2 else int(l+i*(r-l)/(len(vals)-1)),int(t+(lim-v)*(b-t)/(2*lim))))
 if len(pts)>1:cv2.polylines(im,[np.array(pts,np.int32)],False,(95,205,255),3,cv2.LINE_AA)
 q=l if len(vals)<2 else int(l+now*(r-l)/(len(vals)-1));cv2.line(im,(q,t),(q,b),(255,255,255),2,cv2.LINE_AA);cv2.putText(im,f"EGO {a.upper()}",(x+12,y+27),cv2.FONT_HERSHEY_DUPLEX,.7,(245,245,245),2,cv2.LINE_AA)
def _bar(im,box,title,c,indices,now):
 import cv2
 x,y,w,h=box;conf="n/a" if c.get("confidence") is None else f"{c['confidence']:.2f}";cv2.putText(im,f"{title} | N={c['threshold_n']:.3g} | conf={conf}",(x,y-18),cv2.FONT_HERSHEY_DUPLEX,.66,(245,245,245),2,cv2.LINE_AA);total=len(indices)
 for i,fi in enumerate(indices):x0=x+int(i*w/total);x1=x+int((i+1)*w/total);cv2.rectangle(im,(x0,y),(max(x0+1,x1),y+h),COLORS[_state(c,fi)],-1)
 q=x+int((now+.5)*w/total);cv2.line(im,(q,y-7),(q,y+h+7),(255,255,255),3,cv2.LINE_AA);state=_state(c,indices[now]);cv2.putText(im,state.upper(),(x,y+h+34),cv2.FONT_HERSHEY_DUPLEX,.7,COLORS[state],2,cv2.LINE_AA)
def _state_legend(im,x,y,w,states):
 import cv2
 cell=max(1,w//len(states));swatch=14
 for index,state in enumerate(states):
  cell_x=x+index*cell
  cv2.rectangle(im,(cell_x,y),(cell_x+swatch,y+swatch),COLORS[state],-1)
  cv2.rectangle(im,(cell_x,y),(cell_x+swatch,y+swatch),(225,225,225),1)
  cv2.putText(im,state.upper(),(cell_x+swatch+5,y+12),cv2.FONT_HERSHEY_DUPLEX,.34,(225,228,235),1,cv2.LINE_AA)

def _final_prediction(result, axis):
 final=result.get(f"{axis}_segmentation",{}).get("final_segmentation",{})
 frames=list(final.get("frames",[]))
 return {
  "selection":"final_confidence_weighted_dp",
  "status":str(final.get("status","unavailable_no_enabled_candidates")),
  "final_prediction":True,
  "segments":list(final.get("segments",[])),
  "frames":frames,
  "frames_by_index":{int(row["frame_index"]):row for row in frames},
 }

def _candidate_stack(im,box,result,audit,indices,now,show_final=True,highlights=None):
 import cv2
 groups=[
  ("VX SEGMENTS","vx",("right","straight","left"),_enabled_candidates(result,"vx",audit),_final_prediction(result,"vx")),
  ("VZ SEGMENTS","vz",("backward","static","forward"),_enabled_candidates(result,"vz",audit),_final_prediction(result,"vz")),
 ]
 x,y,w,h=box
 total=sum(max(1,len(rows))+(1 if show_final else 0) for _,_,_,rows,_ in groups)
 header_h=54
 pitch=min(92,max(18,int((h-header_h*len(groups))/max(1,total))))
 cursor=y
 total_frames=max(1,len(indices))
 for title,axis,states,rows,final_prediction in groups:
  cv2.putText(im,f"{title} | candidates={len(rows)}"+(" + FINAL" if show_final else ""),(x,cursor+18),cv2.FONT_HERSHEY_DUPLEX,.53,(255,255,255),1,cv2.LINE_AA)
  _state_legend(im,x,cursor+27,w,states)
  cursor+=header_h
  if not rows:
   cv2.putText(im,"No enabled threshold candidates",(x+8,cursor+min(16,pitch-5)),cv2.FONT_HERSHEY_SIMPLEX,.43,(150,155,165),1,cv2.LINE_AA)
   cursor+=pitch
  for candidate in rows:
   axis_highlights=(highlights or {}).get(axis,{})
   optimal=axis_highlights.get("optimal_final_similarity")
   best=axis_highlights.get("best_heatmap_confidence")
   candidate_id=int(candidate.get("plateau_id",-1))
   is_optimal=optimal is not None and candidate_id==int(optimal.get("plateau_id",-2))
   is_best=best is not None and candidate_id==int(best.get("plateau_id",-2))
   confidence="n/a" if candidate.get("confidence") is None else f"{candidate['confidence']:.3f}"
   text_scale=.48 if pitch>=36 else .38
   badges=(" | FINAL BEST" if is_optimal else "")+(" | HEATMAP BEST" if is_best else "")
   highlight_scale=.54 if pitch>=36 else .44
   text_color=(70,255,90) if is_optimal else (0,145,255) if is_best else (235,235,235)
   cv2.putText(im,f"N={candidate['threshold_n']:.5g} | conf={confidence}{badges}",(x,cursor+min(15,pitch-7)),cv2.FONT_HERSHEY_SIMPLEX,highlight_scale if (is_optimal or is_best) else text_scale,text_color,2 if (is_optimal or is_best) else 1,cv2.LINE_AA)
   bar_y=cursor+min(21,pitch-5)
   bar_h=max(3,min(34,pitch-(bar_y-cursor)-2))
   for i,frame_index in enumerate(indices):
    x0=x+int(i*w/total_frames);x1=x+int((i+1)*w/total_frames)
    cv2.rectangle(im,(x0,bar_y),(max(x0+1,x1),bar_y+bar_h),COLORS[_state(candidate,frame_index)],-1)
   if is_optimal:cv2.rectangle(im,(x-4,bar_y-4),(x+w+4,bar_y+bar_h+4),(70,255,90),4)
   if is_best:cv2.rectangle(im,(x-1,bar_y-1),(x+w+1,bar_y+bar_h+1),(0,145,255),3)
   marker_x=x+int((now+.5)*w/total_frames)
   cv2.line(im,(marker_x,bar_y-2),(marker_x,bar_y+bar_h+2),(255,255,255),2,cv2.LINE_AA)
   cursor+=pitch
  if not show_final:
   continue
  current_frame_index=indices[now]
  metrics=final_prediction["frames_by_index"].get(current_frame_index,{})
  confidence=float(metrics.get("confidence",0.0))
  consensus=float(metrics.get("consensus",0.0))
  margin=float(metrics.get("margin",0.0))
  disagreement=float(metrics.get("candidate_disagreement",0.0))
  text_scale=.46 if pitch>=36 else .34
  final_text=(
   f"FINAL PREDICTION | C={confidence:.2f} S={consensus:.2f} M={margin:+.2f} D={disagreement:.2f}"
   if final_prediction.get("status") == "completed"
   else "FINAL UNAVAILABLE | NO ENABLED N"
  )
  final_color=(0,245,255) if final_prediction.get("status") == "completed" else (90,90,245)
  cv2.putText(im,final_text,(x,cursor+min(15,pitch-7)),cv2.FONT_HERSHEY_DUPLEX,text_scale,final_color,2,cv2.LINE_AA)
  bar_y=cursor+min(21,pitch-5)
  bar_h=max(3,min(34,pitch-(bar_y-cursor)-2))
  for i,frame_index in enumerate(indices):
   x0=x+int(i*w/total_frames);x1=x+int((i+1)*w/total_frames)
   cv2.rectangle(im,(x0,bar_y),(max(x0+1,x1),bar_y+bar_h),COLORS[_state(final_prediction,frame_index)],-1)
  cv2.rectangle(im,(x,bar_y),(x+w,bar_y+bar_h),final_color,2)
  marker_x=x+int((now+.5)*w/total_frames)
  cv2.line(im,(marker_x,bar_y-3),(marker_x,bar_y+bar_h+3),(255,255,255),3,cv2.LINE_AA)
  cursor+=pitch

def _step7b_scatter_highlights(result,audit):
 video_id=str(result.get("video_id",""));highlights={}
 for axis in ("vx","vz"):
  current=[row for row in audit.get("points",[]) if str(row.get("video_id",""))==video_id and str(row.get("axis",""))==axis and row.get("enabled")]
  scored=[row for row in current if row.get("confidence") is not None]
  best_confidence=max(scored,key=lambda row:(float(row["confidence"]),-float(row["midpoint_n"]),-int(row.get("plateau_id",-1)))) if scored else None
  selection=result.get(f"{axis}_segmentation",{}).get("optimal_n_selection",{})
  selected_id=selection.get("selected_candidate_id")
  optimal=next((row for row in current if int(row.get("plateau_id",-1))==int(selected_id)),None) if selected_id is not None else None
  highlights[axis]={
   "optimal_final_similarity":None if optimal is None else dict(optimal),
   "optimal_final_similarity_score":selection.get("selected_similarity"),
   "best_heatmap_confidence":None if best_confidence is None else dict(best_confidence),
  }
 return highlights

def _threshold_summary(im,x,y,w,result,audit,show_final=False,highlights=None):
 import cv2
 cv2.rectangle(im,(x,y),(x+w,y+112),(36,40,47),-1)
 cv2.rectangle(im,(x,y),(x+w,y+112),(105,112,124),1)
 cv2.putText(im,"EVAL VIDEO THRESHOLD HIGHLIGHTS" if show_final else "EVAL VIDEO CANDIDATE THRESHOLDS",(x+10,y+23),cv2.FONT_HERSHEY_DUPLEX,.47,(255,255,255),1,cv2.LINE_AA)
 for row,(axis,color) in enumerate((("vx",(95,205,255)),("vz",(90,220,130)))):
  if show_final:
   axis_highlights=(highlights or {}).get(axis,{})
   optimal=axis_highlights.get("optimal_final_similarity");best=axis_highlights.get("best_heatmap_confidence")
   optimal_text="n/a" if optimal is None else f"{float(optimal['midpoint_n']):.4g}"
   best_text="n/a" if best is None else f"{float(best['midpoint_n']):.4g}"
   text=f"{axis.upper()} final-match N={optimal_text} | best-heatmap N={best_text}"
  else:
   values=[candidate["threshold_n"] for candidate in _enabled_candidates(result,axis,audit)]
   shown=values[:5];text=f"{axis.upper()} enabled N: "+(", ".join(f"{value:.4g}" for value in shown) if shown else "none")
   if len(values)>len(shown):text+=f"  (+{len(values)-len(shown)} more)"
  cv2.putText(im,text,(x+10,y+52+row*25),cv2.FONT_HERSHEY_SIMPLEX,.43,color,1,cv2.LINE_AA)
 cv2.putText(im,"Star = final-similarity optimum | X = best heat-map confidence" if show_final else "Plateau-middle candidates; no final N selected",(x+10,y+103),cv2.FONT_HERSHEY_SIMPLEX,.37,(175,183,195),1,cv2.LINE_AA)

def _scatter_panel(audit,video_id,size,result=None,show_heatmap=False):
 """Render all-video vx/vz plateau points and emphasize the current eval video."""
 import cv2,numpy as np
 import matplotlib
 matplotlib.use("Agg")
 from matplotlib.backends.backend_agg import FigureCanvasAgg
 from matplotlib.figure import Figure
 from src.exp_july.perception.ego_axis_threshold_segmentation import _confidence_surface
 points=list(audit.get("points",[])); fig=Figure(figsize=(6.2 if show_heatmap else 4.8,10.4),dpi=100,facecolor="#181b20"); canvas=FigureCanvasAgg(fig); axes=fig.subplots(2,1)
 highlights=_step7b_scatter_highlights(result or {},audit) if show_heatmap else {}
 for ax,axis in zip(axes,("vx","vz")):
  rows=[p for p in points if p.get("axis")==axis]; current=[p for p in rows if str(p.get("video_id",""))==str(video_id)]; others=[p for p in rows if str(p.get("video_id",""))!=str(video_id)]
  ax.set_facecolor("#262a31")
  limits=audit.get("plot_limits_by_axis",{}).get(axis,{})
  if show_heatmap:
   train=[p for p in rows if p.get("split")=="train" and p.get("enabled")]
   bounds=(float(limits.get("x_min",0.)),float(limits.get("x_max",1.)),float(limits.get("y_min",0.)),float(limits.get("y_max",1.)))
   model=_confidence_surface(train,bounds=bounds) if train else None
   if model is not None:ax.contourf(model["x"],model["y"],model["confidence"],levels=np.linspace(0.,1.,13),cmap="viridis",alpha=.72,zorder=0)
   enabled_current=[p for p in current if p.get("enabled")]
   disabled_current=[p for p in current if not p.get("enabled")]
   if enabled_current:ax.scatter([p["midpoint_n"] for p in enabled_current],[p["segment_count"] for p in enabled_current],s=52,c="#e35daf",marker="D",edgecolors="white",linewidths=.8,zorder=6,label="eval enabled N")
   if disabled_current:ax.scatter([p["midpoint_n"] for p in disabled_current],[p["segment_count"] for p in disabled_current],s=38,c="#8a8a8a",marker="x",linewidths=1.1,zorder=5,label="eval disabled N")
   axis_highlights=highlights.get(axis,{})
   optimal=axis_highlights.get("optimal_final_similarity");best=axis_highlights.get("best_heatmap_confidence")
   if optimal is not None:ax.scatter([optimal["midpoint_n"]],[optimal["segment_count"]],s=260,c="#f6e85c",marker="*",edgecolors="#111111",linewidths=1.1,zorder=9,label="best match to 7B final")
   if best is not None:ax.scatter([best["midpoint_n"]],[best["segment_count"]],s=190,c="#00f5ff",marker="X",edgecolors="#111111",linewidths=1.0,zorder=10,label="best heat-map confidence")
  else:
   for split,color,marker in (("train","#4b8fca","o"),("eval","#a36bc1","D")):
    subset=[p for p in others if p.get("split")==split and p.get("enabled")]
    if subset:ax.scatter([p["midpoint_n"] for p in subset],[p["segment_count"] for p in subset],s=24,c=color,marker=marker,alpha=.7,label=f"other {split}")
   disabled=[p for p in others if not p.get("enabled")]
   if disabled:ax.scatter([p["midpoint_n"] for p in disabled],[p["segment_count"] for p in disabled],s=22,c="#777777",marker="x",alpha=.55,label="disabled")
   if current:ax.scatter([p["midpoint_n"] for p in current],[p["segment_count"] for p in current],s=150,c="#00f5ff",marker="*",edgecolors="white",linewidths=1.1,zorder=8,label="CURRENT VIDEO")
  if limits:
   ax.set_xlim(float(limits.get("x_min",0.)),float(limits["x_max"]));ax.set_ylim(float(limits.get("y_min",0.)),float(limits["y_max"]))
  ax.set_title(f"{axis.upper()} eval thresholds + train heat map" if show_heatmap else f"{axis.upper()} plateaus",color="white",fontsize=15 if show_heatmap else 13,fontweight="bold");ax.set_xlabel("middle N",color="#e5e5e5",fontsize=12 if show_heatmap else 10);ax.set_ylabel("segments",color="#e5e5e5",fontsize=12 if show_heatmap else 10);ax.tick_params(colors="#dddddd",labelsize=10 if show_heatmap else 8);ax.grid(True,alpha=.2,color="white")
  for spine in ax.spines.values():spine.set_color("#888888")
  ax.legend(fontsize=10 if show_heatmap else 7,loc="best",facecolor="#30343b",labelcolor="white",framealpha=.94,borderpad=.7,labelspacing=.55,handlelength=1.8)
 fig.suptitle(("TRAIN HEAT MAP + EVAL VIDEO\n" if show_heatmap else "ALL-VIDEO PLATEAUS\n")+f"current={video_id}",color="white",fontsize=14,fontweight="bold");fig.tight_layout(rect=(0,.01,1,.94));canvas.draw();rgba=np.asarray(canvas.buffer_rgba());bgr=cv2.cvtColor(rgba,cv2.COLOR_RGBA2BGR)
 return cv2.resize(bgr,size,interpolation=cv2.INTER_AREA)

def render_eval_signal_segmentation_chart(result,audit,output_path):
 """Render every qualifying vx/vz candidate with enabled/disabled status."""
 import matplotlib
 matplotlib.use("Agg")
 import matplotlib.pyplot as plt
 import numpy as np
 path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True);frames=list(result.get("frames",[]));candidates={"vx":_all_candidates(result,"vx",audit),"vz":_all_candidates(result,"vz",audit)};k=max(1,len(candidates["vx"]),len(candidates["vz"]))
 fig,axes=plt.subplots(k,2,figsize=(18,max(6.5,4.2*k)),squeeze=False,constrained_layout=True)
 for column,(axis,labels) in enumerate((("vx",("right","straight","left")),("vz",("backward","static","forward")))):
  for row in range(k):
   ax=axes[row,column]
   if row>=len(candidates[axis]):
    ax.set_facecolor("#f1f3f5");ax.text(.5,.5,f"No {axis.upper()} qualifying candidate for row {row+1}",transform=ax.transAxes,ha="center",va="center",color="#777777",fontsize=11);ax.set_xticks([]);ax.set_yticks([])
    continue
   candidate=candidates[axis][row]
   enabled=bool(candidate.get("enabled"));ax.set_facecolor("#ffffff" if enabled else "#fff1f1")
   indices=[int(frame.get("frame_index",i)) for i,frame in enumerate(frames)]
   values=[_signal(frame,axis) for frame in frames]
   for segment_index,segment in enumerate(candidate.get("segments",[])):
    state=str(segment.get("state","unavailable"));start=float(segment["start_frame"]);end=float(segment["end_frame"])
    ax.axvspan(start-.5,end+.5,color=PLOT_COLORS.get(state,PLOT_COLORS["unavailable"]),alpha=.2 if enabled else .1,label=state if not any(str(previous.get("state"))==state for previous in candidate["segments"][:segment_index]) else None,zorder=0)
   ax.plot(indices,[np.nan if value is None else value for value in values],color="#17202a",linewidth=1.8,label=f"ego {axis}",zorder=3)
   threshold=float(candidate["threshold_n"])
   ax.axhline(-threshold,color=PLOT_COLORS[labels[0]],linestyle="--",linewidth=1.8,label=f"{labels[0]} / {labels[1]}: −N",zorder=2)
   ax.axhline(threshold,color=PLOT_COLORS[labels[2]],linestyle="--",linewidth=1.8,label=f"{labels[1]} / {labels[2]}: +N",zorder=2)
   confidence="n/a" if candidate.get("confidence") is None else f"{candidate['confidence']:.3f}"
   status="ENABLED" if enabled else "DISABLED"
   ax.set_title(f"[{status}] {axis.upper()} {row+1}/{len(candidates[axis])} | N={threshold:.5g} | conf={confidence}",fontweight="bold",fontsize=10.5,color="#16803a" if enabled else "#b42318",pad=7)
   if not enabled:
    reasons=candidate.get("disabled_reasons",[]) or ["unspecified"]
    reason_text="Disabled because:\n"+"\n".join(f"• {reason}" for reason in reasons)
    ax.text(.02,.96,reason_text,transform=ax.transAxes,ha="left",va="top",fontsize=8.5,color="#8f1d16",bbox={"boxstyle":"round,pad=0.3","fc":"#fff7f6","ec":"#d65a50","alpha":.92},zorder=8)
    ax.text(.98,.88,"DISABLED",transform=ax.transAxes,ha="right",va="top",fontsize=16,fontweight="bold",color="#d65a50",alpha=.30)
   ax.set_xlabel("Frame index");ax.set_ylabel(f"Ego {axis}");ax.grid(True,alpha=.22);ax.legend(loc="best",fontsize=8,ncol=2)
   if indices:ax.set_xlim(min(indices)-.5,max(indices)+.5)
 fig.suptitle(f"Step 7A qualifying threshold segmentations (enabled + disabled) | video={result.get('video_id','')} | rows={k}",fontsize=16,fontweight="bold")
 fig.savefig(path,dpi=170);plt.close(fig)
 return {"status":"rendered","path":str(path),"layout":"k_by_2_all_qualifying_threshold_segmentations","num_rows":k,"vx_candidates":candidates["vx"],"vz_candidates":candidates["vz"],"vx_enabled_candidates":[row for row in candidates["vx"] if row["enabled"]],"vz_enabled_candidates":[row for row in candidates["vz"] if row["enabled"]],"vx_disabled_candidates":[row for row in candidates["vx"] if not row["enabled"]],"vz_disabled_candidates":[row for row in candidates["vz"] if not row["enabled"]]}

def _segment_length_rows(segments,tolerance):
 rows=[]
 for segment in segments:
  row=dict(segment);duration=int(row.get("duration_frames",int(row["end_frame"])-int(row["start_frame"])+1));row["duration_frames"]=duration;row["length_class"]="short" if duration<=tolerance else "long";rows.append(row)
 return rows

def render_eval_candidate_filter_comparisons(result, output_root, max_candidates=20):
    """Render 4x1 raw/filtered segmentation and confidence charts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch
    from src.exp_july.perception.ego_axis_threshold_segmentation import (
        filter_short_state_interruptions, frame_label_confidences,
        merge_remaining_short_segments,
    )
    output_root = Path(output_root)
    frames = list(result.get("frames", []))
    outputs = []
    limit = max(0, int(max_candidates))
    for axis in ("vx", "vz"):
        data = result.get(f"{axis}_segmentation", {})
        label_map = data.get("labels", {})
        labels = (
            str(label_map.get("negative", "negative")),
            str(label_map.get("center", "static")),
            str(label_map.get("positive", "positive")),
        )
        tolerance = int(data.get("noise_filter", {}).get("tolerance_frames", 5))
        candidates = sorted(
            data.get("threshold_candidates", []),
            key=lambda row: (float(row["threshold"]), int(row.get("candidate_index", 0))),
        )[:limit]
        axis_root = output_root / axis
        axis_root.mkdir(parents=True, exist_ok=True)
        indices = [int(frame.get("frame_index", index)) for index, frame in enumerate(frames)]
        values = [_signal(frame, axis) for frame in frames]
        for rank, candidate in enumerate(candidates, 1):
            threshold = float(candidate["threshold"])
            raw = _segment_length_rows(_segments(frames, axis, threshold, labels), tolerance)
            bridged = filter_short_state_interruptions(raw, tolerance, **_bridge_kwargs(data))
            filtered = _segment_length_rows(
                merge_remaining_short_segments(bridged, tolerance), tolerance,
            )
            minimum_long_length = tolerance + 1
            raw_frame_labels = frame_label_confidences(
                frames, raw, raw, minimum_long_length,
            )
            filtered_frame_labels = frame_label_confidences(
                frames, raw, filtered, minimum_long_length,
            )
            path = axis_root / (
                f"candidate_{rank:02d}_index_"
                f"{int(candidate.get('candidate_index', rank - 1)):03d}.png"
            )
            figure, axes = plt.subplots(
                4, 1, figsize=(16, 11), sharex=True,
                gridspec_kw={"height_ratios": [3.0, 1.0, 3.0, 1.0]},
                constrained_layout=True,
            )

            def draw_segmentation(ax, title, segments):
                observed = set()
                for segment_index, segment in enumerate(segments):
                    state = str(segment.get("state", "unavailable"))
                    start_frame = float(segment["start_frame"])
                    end_frame = float(segment["end_frame"])
                    duration = int(segment["duration_frames"])
                    length_class = str(segment["length_class"])
                    ax.axvspan(
                        start_frame - 0.5, end_frame + 0.5,
                        color=PLOT_COLORS.get(state, PLOT_COLORS["unavailable"]),
                        alpha=0.28, zorder=0,
                    )
                    observed.add(state)
                    midpoint = 0.5 * (start_frame + end_frame)
                    ax.text(
                        midpoint, 0.88 - 0.17 * (segment_index % 2),
                        f"{state.upper()}\n{length_class.upper()} {duration}f",
                        transform=ax.get_xaxis_transform(), ha="center", va="top",
                        rotation=90 if length_class == "short" else 0,
                        fontsize=7, fontweight="bold",
                        color="#8f1d16" if length_class == "short" else "#176b35",
                        bbox={
                            "boxstyle": "round,pad=0.2",
                            "fc": "#fff0ee" if length_class == "short" else "#edf9f0",
                            "ec": "#d65a50" if length_class == "short" else "#4aa564",
                            "alpha": 0.88,
                        },
                        zorder=7, clip_on=True,
                    )
                ax.plot(
                    indices, [np.nan if value is None else value for value in values],
                    color="#17202a", linewidth=2.0, label=f"ego {axis}", zorder=3,
                )
                ax.axhline(
                    -threshold, color=PLOT_COLORS[labels[0]], linestyle="--",
                    linewidth=1.7, label="−N",
                )
                ax.axhline(
                    threshold, color=PLOT_COLORS[labels[2]], linestyle="--",
                    linewidth=1.7, label="+N",
                )
                ax.axhline(
                    0.0, color="#f4c542", linestyle=":", linewidth=1.2,
                    label="zero",
                )
                handles = [
                    Patch(facecolor=PLOT_COLORS[state], alpha=0.35, label=state)
                    for state in labels if state in observed
                ]
                handles.extend([
                    Patch(
                        facecolor="#fff0ee", edgecolor="#d65a50",
                        label=f"SHORT ≤ {tolerance}f",
                    ),
                    Patch(
                        facecolor="#edf9f0", edgecolor="#4aa564",
                        label=f"LONG > {tolerance}f",
                    ),
                ])
                line_handles, line_labels = ax.get_legend_handles_labels()
                ax.legend(
                    handles + line_handles,
                    [handle.get_label() for handle in handles] + line_labels,
                    loc="best", fontsize=8, ncol=4,
                )
                ax.set_title(
                    f"{title} | segments={len(segments)}",
                    fontsize=12, fontweight="bold",
                )
                ax.set_ylabel(f"Ego {axis}")
                ax.grid(True, alpha=0.2)
                if indices:
                    ax.set_xlim(min(indices) - 0.5, max(indices) + 0.5)

            def draw_confidence(ax, title, frame_labels):
                confidence_by_frame = {
                    int(row["frame_index"]): float(row["confidence"])
                    for row in frame_labels
                }
                confidence_values = np.asarray([
                    confidence_by_frame.get(frame_index, np.nan)
                    for frame_index in indices
                ], dtype=float)
                confidence_values = np.clip(confidence_values, 0.0, 1.0)
                if indices:
                    ax.imshow(
                        confidence_values.reshape(1, -1),
                        extent=(min(indices) - 0.5, max(indices) + 0.5, 0.0, 1.0),
                        origin="lower", aspect="auto", interpolation="bilinear",
                        cmap="viridis", vmin=0.0, vmax=1.0, zorder=0,
                    )
                    ax.plot(
                        indices, confidence_values, color="#17202a",
                        linewidth=1.8, marker=".", markersize=3.5, zorder=3,
                    )
                    ax.set_xlim(min(indices) - 0.5, max(indices) + 0.5)
                ax.axhline(0.5, color="white", linestyle="--", linewidth=1.0, alpha=0.9)
                ax.set_ylim(0.0, 1.0)
                ax.set_yticks([0.0, 0.5, 1.0])
                ax.set_ylabel("Confidence")
                ax.set_title(
                    f"{title} | Viridis: purple=0 · green=0.5 · yellow=1",
                    fontsize=10.5, fontweight="bold",
                )
                ax.grid(True, axis="x", alpha=0.18)

            draw_segmentation(axes[0], "BEFORE short-segment merge", raw)
            draw_confidence(axes[1], "BEFORE frame-label confidence", raw_frame_labels)
            draw_segmentation(axes[2], "AFTER short-segment merge", filtered)
            draw_confidence(axes[3], "AFTER frame-label confidence", filtered_frame_labels)
            axes[3].set_xlabel("Frame index")
            figure.suptitle(
                f"Step 7A {axis.upper()} candidate {rank}/{len(candidates)} | "
                f"N={threshold:.6g} | tolerance={tolerance} frames | "
                f"video={result.get('video_id', '')}",
                fontsize=15, fontweight="bold",
            )
            figure.savefig(path, dpi=150)
            plt.close(figure)
            outputs.append({
                "status": "rendered",
                "axis": axis,
                "candidate_rank": rank,
                "candidate_index": int(candidate.get("candidate_index", rank - 1)),
                "threshold_n": threshold,
                "raw_segment_count": len(raw),
                "filtered_segment_count": len(filtered),
                "raw_segments": raw,
                "filtered_segments": filtered,
                "raw_frame_confidence_min": min(
                    (row["confidence"] for row in raw_frame_labels), default=None,
                ),
                "raw_frame_confidence_max": max(
                    (row["confidence"] for row in raw_frame_labels), default=None,
                ),
                "filtered_frame_confidence_min": min(
                    (row["confidence"] for row in filtered_frame_labels), default=None,
                ),
                "filtered_frame_confidence_max": max(
                    (row["confidence"] for row in filtered_frame_labels), default=None,
                ),
                "noise_tolerance_frames": tolerance,
                "bridge_config": _bridge_kwargs(data),
                "short_segment_definition": f"duration_frames <= {tolerance}",
                "long_segment_definition": f"duration_frames > {tolerance}",
                "path": str(path),
            })
    return {
        "status": "rendered",
        "layout": "4x1_before_after_segmentation_and_confidence",
        "max_candidates_per_axis": limit,
        "num_charts": len(outputs),
        "charts": outputs,
    }

def render_axis_segmentation_mp4(result,ego_video,audit,output_path,fps=10.,show_final=True,step_label="7B"):
 import cv2,numpy as np
 frames=list(ego_video.get("frames",[]))
 if not frames:return {"status":"skipped","reason":"no_frames","path":None}
 path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True);vx=_candidate(result,"vx",audit);vz=_candidate(result,"vz",audit);W,H=1920,1080;C1,C2=(900,1370) if show_final else (1000,1440);right_panel_highlights=_step7b_scatter_highlights(result,audit) if show_final else {};scatter_panel=_scatter_panel(audit,result.get("video_id",""),(W-C2-40,H-164),result=result,show_heatmap=show_final);wr=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),max(1.,float(fps)),(W,H))
 if not wr.isOpened():raise RuntimeError(f"Could not open Step 7A MP4 writer: {path}")
 ids=[int(f.get("frame_index",i)) for i,f in enumerate(frames)]
 for i,f in enumerate(frames):
  im=np.full((H,W,3),(24,27,32),np.uint8);p=Path(str(f.get("image_path","")));src=cv2.imread(str(p)) if p.is_file() else None
  if src is not None:s=min((C1-44)/src.shape[1],650/src.shape[0]);src=cv2.resize(src,(int(src.shape[1]*s),int(src.shape[0]*s)));x=22+(C1-44-src.shape[1])//2;y=48+(650-src.shape[0])//2;im[y:y+src.shape[0],x:x+src.shape[1]]=src
  else:cv2.putText(im,"SOURCE FRAME UNAVAILABLE",(190,370),cv2.FONT_HERSHEY_DUPLEX,1.2,(80,100,235),2,cv2.LINE_AA)
  cv2.putText(im,f"STEP {step_label} EGO SEGMENTATION | FRAME {ids[i]}",(22,32),cv2.FONT_HERSHEY_DUPLEX,.78,(245,245,245),2,cv2.LINE_AA);cw=(C1-66)//2;_chart(im,(22,730,cw,320),frames,"vx",i);_chart(im,(44+cw,730,cw,320),frames,"vz",i)
  cv2.line(im,(C1,0),(C1,H),(95,100,110),2);cv2.line(im,(C2,0),(C2,H),(95,100,110),2);x=C1+24;w=C2-C1-48;cv2.putText(im,"CANDIDATES + FINAL PREDICTIONS" if show_final else "ENABLED SEGMENTATION CANDIDATES",(x,44),cv2.FONT_HERSHEY_DUPLEX,.72,(255,255,255),2,cv2.LINE_AA);_candidate_stack(im,(x,62,w,H-88),result,audit,ids,i,show_final=show_final,highlights=right_panel_highlights)
  _threshold_summary(im,C2+20,18,W-C2-40,result,audit,show_final=show_final,highlights=right_panel_highlights)
  scatter_y=142;im[scatter_y:scatter_y+scatter_panel.shape[0],C2+20:C2+20+scatter_panel.shape[1]]=scatter_panel
  wr.write(im)
 wr.release();return {"status":"rendered","path":str(path),"fps":float(fps),"num_frames":len(frames),"layout":"source_signals_left_"+("candidates_plus_final_predictions_middle_train_heatmap_eval_highlights_right" if show_final else "enabled_candidates_only_middle_scatters_right"),"step_label":str(step_label),"show_final":bool(show_final),"right_panel_highlights":right_panel_highlights,"vx_candidate":vx,"vz_candidate":vz,"vx_enabled_candidates":_enabled_candidates(result,"vx",audit),"vz_enabled_candidates":_enabled_candidates(result,"vz",audit)}
