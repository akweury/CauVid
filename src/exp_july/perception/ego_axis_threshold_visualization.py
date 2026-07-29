"""Step 7A MP4 audit renderer (display-only hypothesis selection)."""
from pathlib import Path
import math
COLORS={"forward":(70,205,80),"backward":(65,80,235),"static":(145,145,145),"left":(235,190,55),"right":(205,75,205),"straight":(235,165,65),"unavailable":(60,60,60)}
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
   active={"state":state,"start_frame":fi,"end_frame":fi}
  else:active["end_frame"]=fi
  prev=fi
 if active:out.append(active)
 return out
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
  from src.exp_july.perception.ego_axis_threshold_segmentation import filter_short_state_interruptions
  tolerance=int(result.get(key,{}).get("noise_filter",{}).get("tolerance_frames",5))
  segments=filter_short_state_interruptions(_segments(result.get("frames",[]),a,0.,labels),tolerance)
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

def _candidate_stack(im,box,result,audit,indices,now):
 import cv2
 x,y,w,h=box;groups=[("VX SEGMENTS","vx",("right","straight","left"),_enabled_candidates(result,"vx",audit)),("VZ SEGMENTS","vz",("backward","static","forward"),_enabled_candidates(result,"vz",audit))];total=sum(max(1,len(rows)) for _,_,_,rows in groups);header_h=54;pitch=min(92,max(18,int((h-header_h*len(groups))/max(1,total))))
 cursor=y
 for title,axis,states,rows in groups:
  cv2.putText(im,f"{title} | enabled={len(rows)}",(x,cursor+18),cv2.FONT_HERSHEY_DUPLEX,.53,(255,255,255),1,cv2.LINE_AA)
  _state_legend(im,x,cursor+27,w,states)
  cursor+=header_h
  if not rows:
   cv2.putText(im,"No enabled threshold candidates",(x+8,cursor+18),cv2.FONT_HERSHEY_SIMPLEX,.48,(150,155,165),1,cv2.LINE_AA);cursor+=pitch
   continue
  for candidate in rows:
   confidence="n/a" if candidate.get("confidence") is None else f"{candidate['confidence']:.3f}";text_scale=.48 if pitch>=36 else .38;cv2.putText(im,f"N={candidate['threshold_n']:.5g} | confidence={confidence}",(x,cursor+min(15,pitch-7)),cv2.FONT_HERSHEY_SIMPLEX,text_scale,(235,235,235),1,cv2.LINE_AA);bar_y=cursor+min(21,pitch-5);bar_h=max(3,min(34,pitch-(bar_y-cursor)-2));total_frames=max(1,len(indices))
   for i,frame_index in enumerate(indices):
    x0=x+int(i*w/total_frames);x1=x+int((i+1)*w/total_frames);cv2.rectangle(im,(x0,bar_y),(max(x0+1,x1),bar_y+bar_h),COLORS[_state(candidate,frame_index)],-1)
   marker_x=x+int((now+.5)*w/total_frames);cv2.line(im,(marker_x,bar_y-2),(marker_x,bar_y+bar_h+2),(255,255,255),2,cv2.LINE_AA);cursor+=pitch
def _threshold_summary(im,x,y,w,result,audit):
 import cv2
 cv2.rectangle(im,(x,y),(x+w,y+112),(36,40,47),-1)
 cv2.rectangle(im,(x,y),(x+w,y+112),(105,112,124),1)
 cv2.putText(im,"EVAL VIDEO CANDIDATE THRESHOLDS",(x+10,y+23),cv2.FONT_HERSHEY_DUPLEX,.47,(255,255,255),1,cv2.LINE_AA)
 for row,(axis,color) in enumerate((("vx",(95,205,255)),("vz",(90,220,130)))):
  values=[candidate["threshold_n"] for candidate in _enabled_candidates(result,axis,audit)]
  shown=values[:5]
  text=", ".join(f"{value:.4g}" for value in shown) if shown else "none"
  if len(values)>len(shown):text+=f"  (+{len(values)-len(shown)} more)"
  cv2.putText(im,f"{axis.upper()} enabled N: {text}",(x+10,y+52+row*25),cv2.FONT_HERSHEY_SIMPLEX,.43,color,1,cv2.LINE_AA)
 cv2.putText(im,"Plateau-middle candidates; no final N selected",(x+10,y+103),cv2.FONT_HERSHEY_SIMPLEX,.37,(175,183,195),1,cv2.LINE_AA)

def _scatter_panel(audit,video_id,size):
 """Render all-video vx/vz plateau points and emphasize the current eval video."""
 import cv2,numpy as np
 import matplotlib
 matplotlib.use("Agg")
 from matplotlib.backends.backend_agg import FigureCanvasAgg
 from matplotlib.figure import Figure
 points=list(audit.get("points",[])); fig=Figure(figsize=(4.8,10.4),dpi=100,facecolor="#181b20"); canvas=FigureCanvasAgg(fig); axes=fig.subplots(2,1)
 for ax,axis in zip(axes,("vx","vz")):
  rows=[p for p in points if p.get("axis")==axis]; current=[p for p in rows if str(p.get("video_id",""))==str(video_id)]; others=[p for p in rows if str(p.get("video_id",""))!=str(video_id)]
  ax.set_facecolor("#262a31")
  for split,color,marker in (("train","#4b8fca","o"),("eval","#a36bc1","D")):
   subset=[p for p in others if p.get("split")==split and p.get("enabled")]
   if subset:ax.scatter([p["midpoint_n"] for p in subset],[p["segment_count"] for p in subset],s=24,c=color,marker=marker,alpha=.7,label=f"other {split}")
  disabled=[p for p in others if not p.get("enabled")]
  if disabled:ax.scatter([p["midpoint_n"] for p in disabled],[p["segment_count"] for p in disabled],s=22,c="#777777",marker="x",alpha=.55,label="disabled")
  if current:ax.scatter([p["midpoint_n"] for p in current],[p["segment_count"] for p in current],s=150,c="#00f5ff",marker="*",edgecolors="white",linewidths=1.1,zorder=8,label="CURRENT VIDEO")
  limits=audit.get("plot_limits_by_axis",{}).get(axis,{})
  if limits:
   ax.set_xlim(float(limits.get("x_min",0.)),float(limits["x_max"]));ax.set_ylim(float(limits.get("y_min",0.)),float(limits["y_max"]))
  ax.set_title(f"{axis.upper()} plateaus",color="white",fontsize=13,fontweight="bold");ax.set_xlabel("middle N",color="#e5e5e5",fontsize=10);ax.set_ylabel("segments",color="#e5e5e5",fontsize=10);ax.tick_params(colors="#dddddd",labelsize=8);ax.grid(True,alpha=.2,color="white")
  for spine in ax.spines.values():spine.set_color("#888888")
  ax.legend(fontsize=7,loc="best",facecolor="#30343b",labelcolor="white",framealpha=.9)
 fig.suptitle(f"ALL-VIDEO PLATEAUS\ncurrent={video_id}",color="white",fontsize=14,fontweight="bold");fig.tight_layout(rect=(0,.01,1,.94));canvas.draw();rgba=np.asarray(canvas.buffer_rgba());bgr=cv2.cvtColor(rgba,cv2.COLOR_RGBA2BGR)
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
   reason="" if enabled else " | reason="+",".join(candidate.get("disabled_reasons",[]))
   status="ENABLED" if enabled else "DISABLED"
   ax.set_title(f"[{status}] {axis.upper()} candidate {row+1}/{len(candidates[axis])} | threshold N={threshold:.5g} | confidence={confidence}{reason}",fontweight="bold",color="#16803a" if enabled else "#b42318")
   if not enabled:ax.text(.98,.88,"DISABLED",transform=ax.transAxes,ha="right",va="top",fontsize=16,fontweight="bold",color="#d65a50",alpha=.38)
   ax.set_xlabel("Frame index");ax.set_ylabel(f"Ego {axis}");ax.grid(True,alpha=.22);ax.legend(loc="best",fontsize=8,ncol=2)
   if indices:ax.set_xlim(min(indices)-.5,max(indices)+.5)
 fig.suptitle(f"Step 7A qualifying threshold segmentations (enabled + disabled) | video={result.get('video_id','')} | rows={k}",fontsize=16,fontweight="bold")
 fig.savefig(path,dpi=170);plt.close(fig)
 return {"status":"rendered","path":str(path),"layout":"k_by_2_all_qualifying_threshold_segmentations","num_rows":k,"vx_candidates":candidates["vx"],"vz_candidates":candidates["vz"],"vx_enabled_candidates":[row for row in candidates["vx"] if row["enabled"]],"vz_enabled_candidates":[row for row in candidates["vz"] if row["enabled"]],"vx_disabled_candidates":[row for row in candidates["vx"] if not row["enabled"]],"vz_disabled_candidates":[row for row in candidates["vz"] if not row["enabled"]]}

def render_axis_segmentation_mp4(result,ego_video,audit,output_path,fps=10.):
 import cv2,numpy as np
 frames=list(ego_video.get("frames",[]))
 if not frames:return {"status":"skipped","reason":"no_frames","path":None}
 path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True);vx=_candidate(result,"vx",audit);vz=_candidate(result,"vz",audit);W,H,C1,C2=1920,1080,1000,1440;scatter_panel=_scatter_panel(audit,result.get("video_id",""),(W-C2-40,H-164));wr=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),max(1.,float(fps)),(W,H))
 if not wr.isOpened():raise RuntimeError(f"Could not open Step 7A MP4 writer: {path}")
 ids=[int(f.get("frame_index",i)) for i,f in enumerate(frames)]
 for i,f in enumerate(frames):
  im=np.full((H,W,3),(24,27,32),np.uint8);p=Path(str(f.get("image_path","")));src=cv2.imread(str(p)) if p.is_file() else None
  if src is not None:s=min((C1-44)/src.shape[1],650/src.shape[0]);src=cv2.resize(src,(int(src.shape[1]*s),int(src.shape[0]*s)));x=22+(C1-44-src.shape[1])//2;y=48+(650-src.shape[0])//2;im[y:y+src.shape[0],x:x+src.shape[1]]=src
  else:cv2.putText(im,"SOURCE FRAME UNAVAILABLE",(190,370),cv2.FONT_HERSHEY_DUPLEX,1.2,(80,100,235),2,cv2.LINE_AA)
  cv2.putText(im,f"STEP 7A EGO SEGMENTATION | FRAME {ids[i]}",(22,32),cv2.FONT_HERSHEY_DUPLEX,.78,(245,245,245),2,cv2.LINE_AA);cw=(C1-66)//2;_chart(im,(22,730,cw,320),frames,"vx",i);_chart(im,(44+cw,730,cw,320),frames,"vz",i)
  cv2.line(im,(C1,0),(C1,H),(95,100,110),2);cv2.line(im,(C2,0),(C2,H),(95,100,110),2);x=C1+24;w=C2-C1-48;cv2.putText(im,"ALL ENABLED SEGMENTATIONS",(x,44),cv2.FONT_HERSHEY_DUPLEX,.72,(255,255,255),2,cv2.LINE_AA);_candidate_stack(im,(x,62,w,H-88),result,audit,ids,i)
  _threshold_summary(im,C2+20,18,W-C2-40,result,audit)
  scatter_y=142;im[scatter_y:scatter_y+scatter_panel.shape[0],C2+20:C2+20+scatter_panel.shape[1]]=scatter_panel
  wr.write(im)
 wr.release();return {"status":"rendered","path":str(path),"fps":float(fps),"num_frames":len(frames),"layout":"source_signals_left_all_enabled_segmentations_middle_scatters_right","vx_candidate":vx,"vz_candidate":vz,"vx_enabled_candidates":_enabled_candidates(result,"vx",audit),"vz_enabled_candidates":_enabled_candidates(result,"vz",audit)}
