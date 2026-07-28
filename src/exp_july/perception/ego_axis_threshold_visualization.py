"""Step 7A MP4 audit renderer (display-only hypothesis selection)."""
from pathlib import Path
import math
COLORS={"forward":(70,205,80),"backward":(65,80,235),"static":(145,145,145),"left":(235,190,55),"right":(205,75,205),"straight":(235,165,65),"unavailable":(60,60,60)}
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
def _candidate(result,a,audit):
 key=f"{a}_segmentation"; ps=result.get(key,{}).get("qualifying_plateaus",[]); by={int(p["plateau_id"]):p for p in ps}
 pts=[p for p in audit.get("points",[]) if p.get("axis")==a and str(p.get("video_id",""))==str(result.get("video_id","")) and p.get("enabled") and p.get("confidence") is not None and int(p.get("plateau_id",-1)) in by]
 if pts:q=max(pts,key=lambda p:(float(p["confidence"]),-float(p["midpoint_n"])));p=by[int(q["plateau_id"])];conf=float(q["confidence"]);sel="highest_confidence_enabled_plateau"
 elif ps:p=max(ps,key=lambda q:(int(q["num_n_values"]),-float(q["midpoint_n"])));conf=None;sel="widest_qualifying_plateau_fallback"
 else:
  ls=result.get(key,{}).get("labels",{}); labels=(ls.get("negative","negative"),ls.get("center","static"),ls.get("positive","positive"))
  return {"threshold_n":0.,"confidence":None,"selection":"zero_threshold_fallback","display_only":True,"segments":_segments(result.get("frames",[]),a,0.,labels)}
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
def _scatter_panel(audit,video_id,size):
 """Render all-video vx/vz plateau points and emphasize the current eval video."""
 import cv2,numpy as np
 import matplotlib
 matplotlib.use("Agg")
 from matplotlib.backends.backend_agg import FigureCanvasAgg
 from matplotlib.figure import Figure
 points=list(audit.get("points",[])); fig=Figure(figsize=(8.0,4.6),dpi=100,facecolor="#181b20"); canvas=FigureCanvasAgg(fig); axes=fig.subplots(1,2)
 for ax,axis in zip(axes,("vx","vz")):
  rows=[p for p in points if p.get("axis")==axis]; current=[p for p in rows if str(p.get("video_id",""))==str(video_id)]; others=[p for p in rows if str(p.get("video_id",""))!=str(video_id)]
  ax.set_facecolor("#262a31")
  for split,color,marker in (("train","#4b8fca","o"),("eval","#a36bc1","D")):
   subset=[p for p in others if p.get("split")==split and p.get("enabled")]
   if subset:ax.scatter([p["midpoint_n"] for p in subset],[p["segment_count"] for p in subset],s=24,c=color,marker=marker,alpha=.7,label=f"other {split}")
  disabled=[p for p in others if not p.get("enabled")]
  if disabled:ax.scatter([p["midpoint_n"] for p in disabled],[p["segment_count"] for p in disabled],s=22,c="#777777",marker="x",alpha=.55,label="disabled")
  if current:ax.scatter([p["midpoint_n"] for p in current],[p["segment_count"] for p in current],s=150,c="#00f5ff",marker="*",edgecolors="white",linewidths=1.1,zorder=8,label="CURRENT VIDEO")
  ax.set_title(f"{axis.upper()} plateaus",color="white",fontsize=13,fontweight="bold");ax.set_xlabel("middle N",color="#e5e5e5",fontsize=10);ax.set_ylabel("segments",color="#e5e5e5",fontsize=10);ax.tick_params(colors="#dddddd",labelsize=8);ax.grid(True,alpha=.2,color="white")
  for spine in ax.spines.values():spine.set_color("#888888")
  ax.legend(fontsize=7,loc="best",facecolor="#30343b",labelcolor="white",framealpha=.9)
 fig.suptitle(f"ALL-VIDEO PLATEAUS | current={video_id}",color="white",fontsize=14,fontweight="bold");fig.tight_layout(rect=(0,.01,1,.92));canvas.draw();rgba=np.asarray(canvas.buffer_rgba());bgr=cv2.cvtColor(rgba,cv2.COLOR_RGBA2BGR)
 return cv2.resize(bgr,size,interpolation=cv2.INTER_AREA)

def render_axis_segmentation_mp4(result,ego_video,audit,output_path,fps=10.):
 import cv2,numpy as np
 frames=list(ego_video.get("frames",[]))
 if not frames:return {"status":"skipped","reason":"no_frames","path":None}
 path=Path(output_path);path.parent.mkdir(parents=True,exist_ok=True);vx=_candidate(result,"vx",audit);vz=_candidate(result,"vz",audit);W,H,L=1920,1080,1180;scatter_panel=_scatter_panel(audit,result.get("video_id",""),(W-L-40,480));wr=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*"mp4v"),max(1.,float(fps)),(W,H))
 if not wr.isOpened():raise RuntimeError(f"Could not open Step 7A MP4 writer: {path}")
 ids=[int(f.get("frame_index",i)) for i,f in enumerate(frames)]
 for i,f in enumerate(frames):
  im=np.full((H,W,3),(24,27,32),np.uint8);p=Path(str(f.get("image_path","")));src=cv2.imread(str(p)) if p.is_file() else None
  if src is not None:s=min(1136/src.shape[1],650/src.shape[0]);src=cv2.resize(src,(int(src.shape[1]*s),int(src.shape[0]*s)));x=22+(1136-src.shape[1])//2;y=48+(650-src.shape[0])//2;im[y:y+src.shape[0],x:x+src.shape[1]]=src
  else:cv2.putText(im,"SOURCE FRAME UNAVAILABLE",(190,370),cv2.FONT_HERSHEY_DUPLEX,1.2,(80,100,235),2,cv2.LINE_AA)
  cv2.putText(im,f"STEP 7A EGO SEGMENTATION | FRAME {ids[i]}",(22,32),cv2.FONT_HERSHEY_DUPLEX,.78,(245,245,245),2,cv2.LINE_AA);cw=(L-66)//2;_chart(im,(22,730,cw,320),frames,"vx",i);_chart(im,(44+cw,730,cw,320),frames,"vz",i);cv2.line(im,(L,0),(L,H),(95,100,110),2);x=L+40;w=W-L-80;cv2.putText(im,"SEGMENT LABELS",(x,72),cv2.FONT_HERSHEY_DUPLEX,1.,(255,255,255),2,cv2.LINE_AA);_bar(im,(x,155,w,75),"VX RIGHT | STRAIGHT | LEFT",vx,ids,i);_bar(im,(x,390,w,75),"VZ BACKWARD | STATIC | FORWARD",vz,ids,i)
  sy=570;im[sy:sy+scatter_panel.shape[0],L+20:L+20+scatter_panel.shape[1]]=scatter_panel
  cv2.putText(im,"White marker = current frame",(x,535),cv2.FONT_HERSHEY_DUPLEX,.6,(240,240,240),2,cv2.LINE_AA);wr.write(im)
 wr.release();return {"status":"rendered","path":str(path),"fps":float(fps),"num_frames":len(frames),"vx_candidate":vx,"vz_candidate":vz}
