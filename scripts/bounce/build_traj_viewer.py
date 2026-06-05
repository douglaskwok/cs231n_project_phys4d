#!/usr/bin/env python
"""Build an interactive 3D trajectory viewer (GT vs extracted vs predicted) in the metric frame.

Consumes the Step 4a trajectory (3DGS frame), GT object_poses.csv, and the
refit_metric.json produced by refit_metric_gt.py (similarity + fitted params),
and emits a self-contained Plotly HTML with an animated ball.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from phys4d.bounce.physics import simulate_trajectory  # noqa: E402


def _load_traj(path: Path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    fr = np.array([int(float(r["frame"])) for r in rows], dtype=np.int64)
    t = np.array([float(r["t_sec"]) for r in rows], dtype=np.float64)
    p = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=np.float64)
    return fr, t, p


def _load_gt(path: Path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    return {int(float(r["frame"])): np.array([float(r["x_m"]), float(r["y_m"]), float(r["z_m"])]) for r in rows}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traj", type=Path, required=True)
    ap.add_argument("--gt-poses", type=Path, required=True)
    ap.add_argument("--refit", type=Path, required=True, help="refit_metric.json")
    ap.add_argument("--out", type=Path, required=True, help="output index.html")
    ap.add_argument("--radius", type=float, default=0.20)
    args = ap.parse_args()

    fr, t, traj3d = _load_traj(args.traj)
    gt = _load_gt(args.gt_poses)
    refit = json.loads(args.refit.read_text(encoding="utf-8"))

    s = float(refit["similarity"]["scale"])
    R = np.array(refit["similarity"]["R"], dtype=np.float64)
    tv = np.array(refit["similarity"]["t"], dtype=np.float64)
    pr = refit["params"]
    p0 = np.array(pr["p0_m"]); v0 = np.array(pr["v0_m_s"])
    g = float(pr["gravity_z_m_s2"]); e = float(pr["restitution"]); grd = float(pr["ground_z_m"])
    train_end = int(refit["split"]["train"][1])
    test_start = int(refit["split"]["test"][0])
    test_end = int(refit["split"]["test"][1])

    common = [int(f) for f in fr.tolist() if int(f) in gt]
    idx = {int(f): i for i, f in enumerate(fr.tolist())}
    extracted_m = (s * (R @ traj3d.T)).T + tv[None, :]
    extracted_m = np.stack([extracted_m[idx[f]] for f in common], axis=0)
    gt_full = np.stack([gt[f] for f in common], axis=0)
    times = np.array([t[idx[f]] for f in common]); times = times - times[0]
    pred_full = simulate_trajectory(times, p0=p0, v0=v0, gravity_z=g, restitution=e, ground_z=grd)

    def split_of(f):
        if f <= train_end:
            return "train"
        if test_start <= f <= test_end:
            return "test"
        return "future"

    splits = [split_of(f) for f in common]

    data = {
        "frames": common,
        "splits": splits,
        "gt": gt_full.tolist(),
        "extracted": extracted_m.tolist(),
        "pred": pred_full.tolist(),
        "ground": grd,
        "radius": float(args.radius),
        "train_end": train_end, "test_start": test_start, "test_end": test_end,
        "e": e, "g": g, "scale": s,
        "test_rmse_cm": float(refit["heldout_test_rmse_m"]) * 100.0,
        "bounces": refit["bounces"],
    }

    html = _HTML_TEMPLATE.replace("__DATA__", json.dumps(data))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote viewer -> {args.out}")
    return 0


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>Ball bounce — GT vs predicted (metric frame)</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
 body{margin:0;font-family:system-ui,Arial,sans-serif;background:#0e1117;color:#e6e6e6}
 #hdr{padding:10px 16px;border-bottom:1px solid #222}
 #hdr h2{margin:0 0 4px 0;font-size:16px}
 #hdr .m{font-size:13px;color:#9fb3c8}
 #plot{width:100vw;height:calc(100vh - 64px)}
 .pill{display:inline-block;padding:1px 8px;border-radius:10px;margin-right:6px;font-size:12px}
 .train{background:#1f6feb33;color:#79c0ff} .test{background:#23863633;color:#7ee787} .pred{background:#da363322;color:#ff7b72}
</style></head>
<body>
<div id="hdr">
 <h2>Ball bounce — Ground truth vs physics-predicted <span style="color:#666">(metric frame)</span></h2>
 <div class="m" id="meta"></div>
</div>
<div id="plot"></div>
<script>
const D = __DATA__;
const gt=D.gt, pr=D.pred, ex=D.extracted, N=D.frames.length;
const colx=a=>a.map(p=>p[0]), coly=a=>a.map(p=>p[1]), colz=a=>a.map(p=>p[2]);
document.getElementById('meta').innerHTML =
 `<span class="pill train">train ≤ f${D.train_end}</span>`+
 `<span class="pill test">test f${D.test_start}–${D.test_end}</span>`+
 ` e=${D.e.toFixed(3)} (GT 0.93) · g=${D.g.toFixed(2)} · scale=${D.scale.toFixed(3)} · `+
 `held-out RMSE=<b>${D.test_rmse_cm.toFixed(2)} cm</b> · bounces GT=${D.bounces.gt_full}/pred=${D.bounces.predicted_full}`;

// ground plane
const gx=[],gy=[]; const xr=[-0.6,0.6], yr=[-0.4,0.8];
const plane={type:'mesh3d',x:[xr[0],xr[1],xr[1],xr[0]],y:[yr[0],yr[0],yr[1],yr[1]],
  z:[D.ground,D.ground,D.ground,D.ground],i:[0,0],j:[1,2],k:[2,3],opacity:0.18,color:'#888',hoverinfo:'skip',showscale:false,name:'contact plane'};

const tGT={type:'scatter3d',mode:'lines',x:colx(gt),y:coly(gt),z:colz(gt),line:{color:'#2ecc71',width:5},name:'GT trajectory'};
const tEX={type:'scatter3d',mode:'markers',x:colx(ex),y:coly(ex),z:colz(ex),marker:{size:2,color:'#888'},name:'extracted (4DGS→metric)',opacity:0.5};
const tPR={type:'scatter3d',mode:'lines',x:colx(pr),y:coly(pr),z:colz(pr),line:{color:'#e74c3c',width:4,dash:'solid'},name:'physics predict'};
const ballGT={type:'scatter3d',mode:'markers',x:[gt[0][0]],y:[gt[0][1]],z:[gt[0][2]],marker:{size:10,color:'#2ecc71'},name:'GT ball'};
const ballPR={type:'scatter3d',mode:'markers',x:[pr[0][0]],y:[pr[0][1]],z:[pr[0][2]],marker:{size:10,color:'#e74c3c',symbol:'circle-open'},name:'pred ball'};

const frames=[];
for(let k=0;k<N;k++){
  frames.push({name:''+k,data:[{x:[gt[k][0]],y:[gt[k][1]],z:[gt[k][2]]},{x:[pr[k][0]],y:[pr[k][1]],z:[pr[k][2]]}],traces:[3,4]});
}
const layout={paper_bgcolor:'#0e1117',scene:{xaxis:{title:'x (m)',color:'#aaa',range:xr},
  yaxis:{title:'y (m)',color:'#aaa',range:yr},zaxis:{title:'z (m)',color:'#aaa',range:[D.ground-0.05,1.3]},
  aspectmode:'manual',aspectratio:{x:1.2,y:1.0,z:1.3},bgcolor:'#0e1117'},
  margin:{l:0,r:0,t:0,b:0},legend:{font:{color:'#ccc'},x:0,y:1},
  updatemenus:[{type:'buttons',showactive:false,x:0.05,y:0,buttons:[
    {label:'▶ Play',method:'animate',args:[null,{frame:{duration:40,redraw:true},transition:{duration:0},fromcurrent:true}]},
    {label:'❚❚ Pause',method:'animate',args:[[null],{mode:'immediate',frame:{duration:0}}]}]}],
  sliders:[{active:0,steps:frames.map((f,k)=>({label:''+D.frames[k],method:'animate',args:[[''+k],{mode:'immediate',frame:{duration:0,redraw:true}}]})),x:0.12,len:0.85,pad:{t:10},currentvalue:{prefix:'frame ',font:{color:'#9fb3c8'}}}]};

Plotly.newPlot('plot',[tGT,tEX,tPR,ballGT,ballPR,plane],layout,{responsive:true}).then(()=>{
  Plotly.addFrames('plot',frames);
});
</script></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
