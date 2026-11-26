"""Self-contained interactive 3D viewer (3Dmol.js inlined, PDB data inlined)."""
import json
V = "report/viz"
info = json.load(open(f"{V}/viz_info.json"))
lib = open(f"{V}/3Dmol-min.js").read()
data = {}
for n in info:
    f = info[n]["frame"]
    data[n] = {k: open(f"{V}/{n}_{k}.pdb").read() for k in [f"true_f{f}", f"recon_f{f}", "recon_traj", "true_traj"]}
    data[n]["frame"] = f
html = """<!doctype html><html><head><meta charset="utf-8"><title>md_compress: true vs reconstructed</title>
<style>body{font-family:sans-serif;margin:0;display:flex;height:100vh}#ctl{width:300px;padding:12px;box-sizing:border-box;overflow:auto;font-size:13px}
#v{flex:1;position:relative}label{display:block;margin:4px 0}h3{margin:6px 0}.k{display:inline-block;width:12px;height:12px;margin-right:4px;vertical-align:middle}</style>
<script>__LIB__</script></head><body>
<div id="ctl"><h3>True vs reconstructed</h3>
<label>System <select id="sys"><option value="adk">AdK (214 res, test frame __ADKF__)</option><option value="2ygs_A">2ygs_A APAF1 CARD (92 res, test frame __YF__)</option></select></label>
<label><input type="checkbox" id="showTrue" checked> <span class="k" style="background:#666"></span>True MD frame (grey)</label>
<label><input type="checkbox" id="showRecon" checked> <span class="k" style="background:orange"></span>Reconstruction (orange)</label>
<label><input type="checkbox" id="errColor"> Color reconstruction by per-residue side-chain error (blue 0 Å, white 1.25 Å, red at least 2.5 Å)</label>
<label><input type="checkbox" id="sticks" checked> Side-chain sticks</label>
<hr><h3>Trajectory</h3>
<label><input type="checkbox" id="anim"> Animate reconstructed frames (every 4th frame), true overlay optional</label>
<p>Reconstructions are decoded from 4 fp16 numbers per residue per frame, superposed on the true frame by backbone atoms. Drag to rotate, scroll to zoom. Hover a residue to see its error.</p>
<p id="hover"></p></div><div id="v"></div>
<script>
const DATA = __DATA__;
let viewer = $3Dmol.createViewer(document.getElementById('v'), {backgroundColor:'white'});
function errColor(b){const t=Math.min(b/2.5,1); if(t<0.5){const s=t*2; return `rgb(${Math.round(60+195*s)},${Math.round(60+195*s)},255)`;} const s=(t-0.5)*2; return `rgb(255,${Math.round(255-195*s)},${Math.round(255-195*s)})`;}
function draw(){
  viewer.clear(); const s=document.getElementById('sys').value, d=DATA[s], f=d.frame;
  const st=document.getElementById('sticks').checked, anim=document.getElementById('anim').checked;
  const sc={and:[{not:{atom:['N','C','O']}},{not:{resn:'GLY'}}]};
  if(anim){
    const m=viewer.addModelsAsFrames(d['recon_traj'],'pdb');
    m.setStyle({}, {cartoon:{color:'orange',opacity:0.9}}); if(st) m.setStyle(sc,{cartoon:{color:'orange'},stick:{colorscheme:'orangeCarbon',radius:0.18}});
    if(document.getElementById('showTrue').checked){const t=viewer.addModelsAsFrames(d['true_traj'],'pdb'); t.setStyle({}, {cartoon:{color:'#999',opacity:0.5}}); if(st) t.setStyle(sc,{cartoon:{color:'#999',opacity:0.5},stick:{colorscheme:'grayCarbon',radius:0.15}});}
    viewer.zoomTo(); viewer.animate({loop:'forward',reps:0,interval:250}); viewer.render(); return;
  }
  viewer.stopAnimate();
  if(document.getElementById('showTrue').checked){const t=viewer.addModel(d['true_f'+f],'pdb'); t.setStyle({}, {cartoon:{color:'#bbb',opacity:0.6}}); if(st) t.setStyle(sc,{cartoon:{color:'#bbb',opacity:0.6},stick:{colorscheme:'grayCarbon',radius:0.17}});}
  if(document.getElementById('showRecon').checked){const r=viewer.addModel(d['recon_f'+f],'pdb');
    if(document.getElementById('errColor').checked){r.setStyle({}, {cartoon:{colorfunc:(a)=>errColor(a.b)}}); if(st) r.setStyle(sc,{cartoon:{colorfunc:(a)=>errColor(a.b)},stick:{colorfunc:(a)=>errColor(a.b),radius:0.2}});}
    else {r.setStyle({}, {cartoon:{color:'orange',opacity:0.85}}); if(st) r.setStyle(sc,{cartoon:{color:'orange',opacity:0.85},stick:{colorscheme:'orangeCarbon',radius:0.2}});}
    r.setHoverable({},true,(a)=>{document.getElementById('hover').textContent=`${a.resn}${a.resi}: side-chain error ${a.b.toFixed(2)} Å`;},()=>{});}
  viewer.zoomTo(); viewer.render();
}
['sys','showTrue','showRecon','errColor','sticks','anim'].forEach(id=>document.getElementById(id).addEventListener('change',draw));
draw();
</script></body></html>"""
html = (html.replace("__LIB__", lib).replace("__DATA__", json.dumps(data))
        .replace("__ADKF__", str(info["adk"]["frame"])).replace("__YF__", str(info["2ygs_A"]["frame"])))
open(f"{V}/viewer.html", "w").write(html)
print("wrote", f"{V}/viewer.html", len(html) // 1024, "KB")
