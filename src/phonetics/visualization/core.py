"""Strict, versioned project visualization primitives."""
from __future__ import annotations
from dataclasses import dataclass
from hashlib import sha256
import json, math, os, re, shutil, tomllib
from pathlib import Path
from typing import Any
import numpy as np
ROOT=Path(__file__).resolve().parent; STYLE_PATH=ROOT/"style.toml"; REGISTRY_PATH=ROOT/"registry.toml"; MM_PER_INCH=25.4
class VisualizationError(ValueError): """An actionable protocol violation."""
STYLE_KEYS={"schema":{"version"},"font":{"family","axis_label_pt","tick_pt","legend_pt","panel_heading_pt","panel_letter_pt","annotation_pt","normal_weight","panel_letter_weight"},"color":{"background","foreground","secondary","tertiary","quaternary"},"line":{"data_pt","axes_pt","reference_pt","interval_pt","caps_pt"},"marker":{"size_pt","edge_pt","face","markevery_target"},"layout":{"single_width_mm","multi_width_mm","panel_aspect","left_mm","right_mm","bottom_mm","top_mm","hgap_mm","vgap_mm","legend_row_mm","legend_gap_mm","default_columns"},"axes":{"grid","minor_ticks","tick_direction","major_length_pt","major_width_pt","major_pad_pt","major_target","panel_label_x_mm","panel_label_y_mm"},"legend":{"max_per_row","handle_length_em","column_spacing_em","location"},"histogram":{"density","fill","bootstrap_bins"},"export":{"formats","dpi","transparent","overwrite"},"backend":{"rcparams"}}
FIGURE_KEYS={"status","kind","name","xlabel","ylabel","series","order","scale","bin_rule","bins","density","columns","xlim","ylim","edges_start","edges_stop","edges_count"};SEMANTIC_KEYS={"label","color","linestyle","marker"};ALLOWED_RCPARAMS={"pdf.fonttype","ps.fonttype","svg.fonttype"}
def _read(p):raw=Path(p).read_bytes();return tomllib.loads(raw.decode()),raw
def _unknown(d,a,w):
 e=set(d)-a
 if e:raise VisualizationError(f"Unknown key(s) in {w}: {', '.join(sorted(e))}")
def _positive(v,n):
 if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) or v<=0:raise VisualizationError(f"{n} must be finite and positive")
@dataclass(frozen=True)
class Protocol:
 style:dict[str,Any];registry:dict[str,Any];style_hash:str;registry_hash:str;style_path:Path;registry_path:Path
 def figure(self,fid):
  try:return self.registry["figures"][fid]
  except KeyError as e:raise VisualizationError(f"Unknown figure ID: {fid}") from e
def load_protocol(style_path=STYLE_PATH,registry_path=REGISTRY_PATH):
 sp=Path(style_path).resolve();rp=Path(registry_path).resolve();s,sr=_read(sp);r,rr=_read(rp);_unknown(s,set(STYLE_KEYS),"style root")
 for sec,keys in STYLE_KEYS.items():
  if sec not in s:raise VisualizationError(f"Missing style section: {sec}")
  _unknown(s[sec],keys,f"style.{sec}");missing=keys-set(s[sec])
  if missing:raise VisualizationError(f"Missing key(s) in style.{sec}: {', '.join(sorted(missing))}")
 if s["schema"]["version"]!=1:raise VisualizationError("Unsupported style schema version")
 if s["font"]["family"]!="Arial":raise VisualizationError("The protocol requires Arial")
 for k,v in s["font"].items():
  if k.endswith("_pt"):_positive(v,f"font.{k}")
 if (s["font"]["normal_weight"],s["font"]["panel_letter_weight"])!=("normal","bold"):raise VisualizationError("Font weights must be normal and bold")
 for k,v in s["layout"].items():
  if k!="default_columns":_positive(v,f"layout.{k}")
 if s["layout"]["default_columns"]!=[1,2,2,2,3,3,3]:raise VisualizationError("Invalid default columns")
 a=s["axes"]
 if a["grid"] is not False or a["minor_ticks"] is not False or a["tick_direction"]!="in":raise VisualizationError("Axes require grid/minor false and inward ticks")
 for k in ("major_length_pt","major_width_pt","major_pad_pt","major_target"):_positive(a[k],f"axes.{k}")
 if s["export"]["formats"]!=["png","eps"] or s["export"]["dpi"]!=300 or s["export"]["transparent"] is not False:raise VisualizationError("Export requires opaque PNG300 then EPS")
 _unknown(s["backend"]["rcparams"],ALLOWED_RCPARAMS,"style.backend.rcparams");_unknown(r,{"schema","figures","semantics"},"registry root")
 if r.get("schema",{}).get("version")!=1 or not isinstance(r.get("figures"),dict) or not isinstance(r.get("semantics"),dict):raise VisualizationError("Invalid registry schema")
 for sid,sem in r["semantics"].items():
  _unknown(sem,SEMANTIC_KEYS,f"semantics.{sid}")
  if set(sem)!=SEMANTIC_KEYS or not all(isinstance(x,str) and x.strip() for x in sem.values()):raise VisualizationError(f"Invalid semantics.{sid}")
 for fid,f in r["figures"].items():
  _unknown(f,FIGURE_KEYS,f"figures.{fid}");req={"status","kind","name","xlabel","ylabel","series","order","scale"}
  if req-set(f):raise VisualizationError(f"Missing required registry keys for {fid}")
  if f["status"] not in {"draft","approved"} or f["kind"] not in {"ecdf","histogram","bootstrap_histogram"} or f["scale"] not in {"linear","log"}:raise VisualizationError(f"Invalid status/kind/scale for {fid}")
  if not all(isinstance(f[k],str) and f[k].strip() for k in ("name","xlabel","ylabel")):raise VisualizationError(f"Invalid labels for {fid}")
  if f["series"]!=f["order"] or not f["series"] or len(set(f["series"]))!=len(f["series"]) or set(f["series"])-set(r["semantics"]):raise VisualizationError(f"Invalid series/order for {fid}")
  if "bin_rule" in f and f["bin_rule"]!="pooled_fd_sturges":raise VisualizationError(f"Invalid bin_rule for {fid}")
  if "bins" in f and (isinstance(f["bins"],bool) or not isinstance(f["bins"],int) or f["bins"]<1):raise VisualizationError(f"Invalid bins for {fid}")
  if "xlim" in f and (not isinstance(f["xlim"],list) or len(f["xlim"])!=2 or not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in f["xlim"]) or f["xlim"][0]>=f["xlim"][1]):raise VisualizationError(f"Invalid xlim for {fid}")
  if f["scale"]=="log" and "xlim" in f and f["xlim"][0]<=0:raise VisualizationError(f"Log xlim must be positive for {fid}")
  if f["kind"]=="ecdf":
   if "ylim" not in f or not isinstance(f["ylim"],list) or len(f["ylim"])!=2 or not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in f["ylim"]) or f["ylim"][0]>=f["ylim"][1]:raise VisualizationError(f"ECDF ylim must be two ascending finite values for {fid}")
  else:
   if not isinstance(f.get("density"),bool):raise VisualizationError(f"Histogram density must be boolean for {fid}")
   y=f.get("ylim")
   if not isinstance(y,list) or len(y) not in {1,2} or not all(isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in y) or (len(y)==2 and y[0]>=y[1]):raise VisualizationError(f"Histogram ylim must be a finite bottom or ascending pair for {fid}")
   if f["kind"]=="histogram" and not ({"edges_start","edges_stop","edges_count"}<=set(f) or f.get("bin_rule")=="pooled_fd_sturges"):raise VisualizationError(f"Histogram requires fixed edges or pooled bin rule for {fid}")
   if f["kind"]=="bootstrap_histogram" and "bins" not in f:raise VisualizationError(f"Bootstrap histogram requires bins for {fid}")
  edgekeys={"edges_start","edges_stop","edges_count"}&set(f)
  if edgekeys and (edgekeys!={"edges_start","edges_stop","edges_count"} or not all(isinstance(f[k],(int,float)) and not isinstance(f[k],bool) and math.isfinite(f[k]) for k in ("edges_start","edges_stop")) or f["edges_stop"]<=f["edges_start"] or isinstance(f["edges_count"],bool) or not isinstance(f["edges_count"],int) or f["edges_count"]<2):raise VisualizationError(f"Invalid edge specification for {fid}")
 return Protocol(s,r,sha256(sr).hexdigest(),sha256(rr).hexdigest(),sp,rp)
def _integrity(p):
 if sha256(p.style_path.read_bytes()).hexdigest()!=p.style_hash or sha256(p.registry_path.read_bytes()).hexdigest()!=p.registry_hash:raise VisualizationError("Protocol files changed; reload")
 if tomllib.loads(p.style_path.read_text())!=p.style or tomllib.loads(p.registry_path.read_text())!=p.registry:raise VisualizationError("Protocol mappings were mutated; reload")
def require_arial():
 from matplotlib import font_manager
 if not any(f.name=="Arial" for f in font_manager.fontManager.ttflist):raise VisualizationError("Arial is not installed; font fallback is prohibited")
def apply_style(p):
 import matplotlib as mpl
 require_arial();s=p.style;f=s["font"];a=s["axes"];c=s["color"]
 rc={"font.family":f["family"],"font.weight":f["normal_weight"],"font.size":f["tick_pt"],"axes.labelsize":f["axis_label_pt"],"axes.titlesize":f["panel_heading_pt"],"axes.labelweight":f["normal_weight"],"axes.titleweight":f["normal_weight"],"xtick.labelsize":f["tick_pt"],"ytick.labelsize":f["tick_pt"],"legend.fontsize":f["legend_pt"],"axes.linewidth":s["line"]["axes_pt"],"lines.linewidth":s["line"]["data_pt"],"lines.markersize":s["marker"]["size_pt"],"lines.markeredgewidth":s["marker"]["edge_pt"],"lines.markerfacecolor":s["marker"]["face"],"axes.edgecolor":c["foreground"],"axes.labelcolor":c["foreground"],"text.color":c["foreground"],"xtick.color":c["foreground"],"ytick.color":c["foreground"],"figure.facecolor":c["background"],"axes.facecolor":c["background"],"axes.grid":a["grid"],"xtick.direction":a["tick_direction"],"ytick.direction":a["tick_direction"],"xtick.top":False,"ytick.right":False,"xtick.minor.visible":False,"ytick.minor.visible":False,"xtick.major.size":a["major_length_pt"],"ytick.major.size":a["major_length_pt"],"xtick.major.width":a["major_width_pt"],"ytick.major.width":a["major_width_pt"],"xtick.major.pad":a["major_pad_pt"],"ytick.major.pad":a["major_pad_pt"],"axes.spines.left":True,"axes.spines.right":True,"axes.spines.top":True,"axes.spines.bottom":True};rc.update(s["backend"]["rcparams"]);mpl.rcParams.update(rc)
def layout(n,p,*,columns=None,legend_rows=0):
 import matplotlib.pyplot as plt
 from matplotlib.ticker import MaxNLocator
 if not isinstance(n,int) or not 1<=n<=26:raise VisualizationError("Panel count must be 1..26")
 l=p.style["layout"];cols=columns if columns is not None else (l["default_columns"][n-1] if n<=7 else 3)
 if not isinstance(cols,int) or not 1<=cols<=min(3,n):raise VisualizationError("columns must be 1..min(3,n)")
 rows=(n+cols-1)//cols;width=l["single_width_mm"] if cols==1 else l["multi_width_mm"];pw=(width-l["left_mm"]-l["right_mm"]-(cols-1)*l["hgap_mm"])/cols;ph=pw/l["panel_aspect"];leg=legend_rows*l["legend_row_mm"]+(l["legend_gap_mm"] if legend_rows else 0);height=l["bottom_mm"]+l["top_mm"]+leg+rows*ph+(rows-1)*l["vgap_mm"];fig=plt.figure(figsize=(width/MM_PER_INCH,height/MM_PER_INCH));axes=[];a=p.style["axes"]
 for i in range(n):
  row=i//cols;nr=min(cols,n-row*cols);col=i%cols;x=l["left_mm"]+(cols-nr)*(pw+l["hgap_mm"])/2+col*(pw+l["hgap_mm"]);y=l["bottom_mm"]+(rows-1-row)*(ph+l["vgap_mm"]);ax=fig.add_axes([x/width,y/height,pw/width,ph/height]);axes.append(ax);ax.tick_params(top=False,right=False,bottom=True,left=True,direction="in",length=a["major_length_pt"],width=a["major_width_pt"],pad=a["major_pad_pt"]);ax.minorticks_off();ax.grid(False);ax.xaxis.set_major_locator(MaxNLocator(a["major_target"]));ax.yaxis.set_major_locator(MaxNLocator(a["major_target"]));ax.text(a["panel_label_x_mm"]/pw,1+a["panel_label_y_mm"]/ph,chr(97+i),transform=ax.transAxes,fontsize=p.style["font"]["panel_letter_pt"],fontweight=p.style["font"]["panel_letter_weight"],va="bottom")
 return fig,axes
def finite(values,*,policy="raise"):
 if policy not in {"raise","drop"}:raise VisualizationError("finite_policy must be raise or drop")
 a=np.asarray(values,float);m=np.isfinite(a)
 if not m.all() and policy=="raise":raise VisualizationError("Non-finite values present; select drop explicitly")
 a=a[m]
 if not len(a):raise VisualizationError("No finite observations")
 return a
def ecdf(values,*,finite_policy="raise"):x=np.sort(finite(values,policy=finite_policy),kind="mergesort");return x,np.arange(1,len(x)+1)/len(x)
def pooled_edges(series,*,finite_policy="raise"):
 x=np.concatenate([finite(v,policy=finite_policy) for v in series])
 if x.min()==x.max():return np.array([x[0]-.5,x[0]+.5])
 q=np.percentile(x,[25,75]);iqr=q[1]-q[0];bins=int(np.ceil((x.max()-x.min())/(2*iqr/np.cbrt(len(x))))) if iqr>0 else int(np.ceil(np.log2(len(x)))+1);return np.linspace(x.min(),x.max(),max(1,bins)+1)
def _record(fig,d):fig._phonetics_resolution=d
def plot_ecdf(ax,series,figure,p,*,finite_policy="raise"):
 counts={}
 for sid in figure["order"]:
  x,y=ecdf(series[sid],finite_policy=finite_policy);counts[sid]=len(x);sem=p.registry["semantics"][sid];ax.step(x,y,where="post",label=sem["label"],color=sem["color"],linestyle=sem["linestyle"],marker=None,linewidth=p.style["line"]["data_pt"])
 ax.set(xlabel=figure["xlabel"],ylabel=figure["ylabel"],ylim=tuple(figure["ylim"]),xscale=figure["scale"]);ax.set_xlim(tuple(figure["xlim"])) if "xlim" in figure else None;_record(ax.figure,{"kind":"ecdf","finite_policy":finite_policy,"counts":counts,"limits":{"x":list(ax.get_xlim()),"y":list(ax.get_ylim())}})
def plot_histogram(ax,series,figure,p,*,edges=None,finite_policy="raise"):
 arr=[finite(series[s],policy=finite_policy) for s in figure["order"]];edges=np.linspace(figure["edges_start"],figure["edges_stop"],figure["edges_count"]) if edges is None and "edges_count" in figure else edges;edges=np.histogram_bin_edges(np.concatenate(arr),bins=figure["bins"]) if edges is None and "bins" in figure else (pooled_edges(arr) if edges is None else np.asarray(edges,float));edges=np.asarray(edges,float)
 if len(edges)<2 or not np.all(np.isfinite(edges)) or not np.all(np.diff(edges)>0):raise VisualizationError("Histogram edges must be finite and strictly increasing")
 for sid,x in zip(figure["order"],arr):
  sem=p.registry["semantics"][sid];ax.hist(x,bins=edges,density=figure["density"],histtype="step",label=sem["label"],color=sem["color"],linestyle=sem["linestyle"],linewidth=p.style["line"]["data_pt"])
 ax.set(xlabel=figure["xlabel"],ylabel=figure["ylabel"],xscale=figure["scale"]);ax.set_ylim(tuple(figure["ylim"])) if len(figure["ylim"])==2 else ax.set_ylim(bottom=figure["ylim"][0]);ax.set_xlim(tuple(figure["xlim"])) if "xlim" in figure else None;_record(ax.figure,{"kind":figure["kind"],"histogram_edges":edges.tolist(),"finite_policy":finite_policy,"counts":{s:len(x) for s,x in zip(figure["order"],arr)},"limits":{"x":list(ax.get_xlim()),"y":list(ax.get_ylim())}});return edges
def shared_legend(fig,axes,p,*,rows=1):
 h,l=axes[0].get_legend_handles_labels();c=p.style["legend"];return fig.legend(h,l,loc=c["location"],ncol=min(c["max_per_row"],max(1,int(np.ceil(len(l)/rows)))),frameon=False,handlelength=c["handle_length_em"],columnspacing=c["column_spacing_em"])
def export_figure(fig,output_dir,stem,figure_id,p,*,scientific_provenance,overwrite=None):
 _integrity(p);spec=p.figure(figure_id)
 if spec["status"]!="approved":raise VisualizationError(f"Figure {figure_id!r} is draft; approval required")
 if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*",stem) or stem in {".",".."}:raise VisualizationError("Unsafe export stem")
 prov=dict(scientific_provenance)
 if not re.fullmatch(r"[0-9a-fA-F]{64}",str(prov.get("source_digest",""))):raise VisualizationError("source_digest must be SHA-256 hex")
 require_arial();out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
 if not out.is_dir():raise VisualizationError("Export target is not a directory")
 paths=[out/f"{stem}.{x}" for x in p.style["export"]["formats"]];mp=out/f"{stem}.metadata.json";allow=p.style["export"]["overwrite"] if overwrite is None else overwrite
 if not allow and any(x.exists() for x in paths+[mp]):raise FileExistsError("Refusing to overwrite export or metadata")
 from matplotlib.colors import to_rgba_array
 for a in fig.findobj():
  if hasattr(a,"get_visible") and not a.get_visible():continue
  alpha=a.get_alpha() if hasattr(a,"get_alpha") else None
  if alpha is not None and 0<alpha<1:raise VisualizationError("EPS forbids partial transparency")
  for getter in ("get_color","get_markerfacecolor","get_markeredgecolor","get_facecolor","get_edgecolor"):
   if not hasattr(a,getter):continue
   try: rgba=to_rgba_array(getattr(a,getter)())
   except (TypeError,ValueError):continue
   if rgba.size and np.any((rgba[:,3]>0)&(rgba[:,3]<1)):raise VisualizationError("EPS forbids RGBA partial transparency")
 fig.canvas.draw();renderer=fig.canvas.get_renderer();box=fig.get_window_extent(renderer)
 from matplotlib.legend import Legend
 from matplotlib.text import Text
 ticks=set()
 for ax in fig.axes:
  for axis in (ax.xaxis,ax.yaxis):
   lo,hi=sorted(axis.get_view_interval());tol=max(1,abs(hi-lo))*1e-12
   for tick in axis._update_ticks():
    loc=tick.get_loc()
    if math.isfinite(loc) and lo-tol<=loc<=hi+tol:
     ticks.update(x for x in (tick.label1,tick.label2) if x.get_visible())
 for a in fig.findobj(lambda x:isinstance(x,(Text,Legend))):
  if isinstance(a,Text) and a not in ticks and any(a is z for ax in fig.axes for z in ax.get_xticklabels()+ax.get_yticklabels()):continue
  if not a.get_visible():continue
  b=a.get_window_extent(renderer)
  if b.width and b.height and (b.x0<box.x0-1 or b.y0<box.y0-1 or b.x1>box.x1+1 or b.y1>box.y1+1):raise VisualizationError("Visible text or legend is clipped")
 meta={"schema_version":1,"config_schema_version":p.style["schema"]["version"],"figure_id":figure_id,"style_hash":p.style_hash,"registry_hash":p.registry_hash,"resolved_style":p.style,"resolved_figure":spec,"resolved_data":getattr(fig,"_phonetics_resolution",{}),"scientific_provenance":prov};temps=[]
 try:
  for dest in paths:
   t=dest.with_name(dest.stem+".tmp"+dest.suffix);temps.append(t)
   import matplotlib as mpl
   with mpl.rc_context({"savefig.bbox":None,"savefig.pad_inches":0,"savefig.dpi":p.style["export"]["dpi"],"savefig.facecolor":p.style["color"]["background"],"savefig.transparent":False}):fig.savefig(t,format=dest.suffix[1:],dpi=p.style["export"]["dpi"],bbox_inches=None,pad_inches=0,transparent=False,facecolor=p.style["color"]["background"],metadata={"Creator":"phonetics.visualization"})
  mt=mp.with_name(mp.stem+".tmp"+mp.suffix);temps.append(mt);mt.write_text(json.dumps(meta,indent=2,sort_keys=True))
  for t,d in zip(temps,paths+[mp]):os.replace(t,d)
 finally:
  for t in temps:
   if t.exists():t.unlink()
 return paths
def create_snapshot(destination,p):
 _integrity(p);d=Path(destination)
 if d.exists():raise FileExistsError("Snapshot destination must be new")
 d.mkdir(parents=True);srcs={"__init__.py":ROOT/"__init__.py","core.py":ROOT/"core.py","README.md":ROOT/"README.md","style.toml":p.style_path,"registry.toml":p.registry_path};hashes={}
 for n,src in srcs.items():shutil.copy2(src,d/n);hashes[n]=sha256((d/n).read_bytes()).hexdigest()
 (d/"snapshot.json").write_text(json.dumps({"schema_version":1,"files":hashes,"style_hash":p.style_hash,"registry_hash":p.registry_hash},indent=2,sort_keys=True));return d
def verify_snapshot(destination):
 d=Path(destination);m=json.loads((d/"snapshot.json").read_text());expected={"__init__.py","core.py","README.md","style.toml","registry.toml"}
 if m.get("schema_version")!=1 or set(m.get("files",{}))!=expected or {x.name for x in d.iterdir()}!=expected|{"snapshot.json"}:raise VisualizationError("Invalid snapshot manifest or file set")
 for n,w in m["files"].items():
  if sha256((d/n).read_bytes()).hexdigest()!=w:raise VisualizationError(f"Snapshot hash mismatch: {n}")
 if m["files"]["style.toml"]!=m.get("style_hash") or m["files"]["registry.toml"]!=m.get("registry_hash"):raise VisualizationError("Snapshot config hashes do not match payload")
