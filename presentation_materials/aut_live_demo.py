#!/usr/bin/env python3
"""
Live Alternate Uses Task (AUT) clustering demo.

  * Audience opens the printed URL (or scans the QR code) on their phones and
    submits creative uses for an object (default: bubble wrap).
  * You open the presenter view on your laptop, watch ideas arrive, hit
    "Run analysis", and get an interactive cluster plot.

Pipeline (mirrors the analysis notebook): embeddings -> L2 normalize ->
          HDBSCAN (euclidean, EOM; noise = -1, light gray) -> t-SNE ->
          interactive Plotly scatter, colorblind palette, largest cluster starred.

Setup:
    pip install flask scikit-learn plotly sentence-transformers
    pip install hdbscan qrcode         # optional: falls back to sklearn's HDBSCAN / no QR

Run:
    python aut_live_demo.py                    # live
    python aut_live_demo.py --seed             # preload sample ideas (rehearsal / fallback)
    python aut_live_demo.py --object "paperclip" --min-cluster-size 4 --min-samples 2

Everything runs locally. The sentence-transformers model is downloaded once
(~90 MB) on first use, so run it once with internet BEFORE the talk. If the
model isn't available, the script falls back to TF-IDF so the demo never dies.
All submissions are also appended to ideas.csv as a backup.
"""
import argparse
import csv
import html
import os
import socket
import threading
import time

import numpy as np
from flask import Flask, Response, abort, jsonify, request

# --------------------------------------------------------------------------
# State
# --------------------------------------------------------------------------
IDEAS = []            # list of {"id": int, "text": str}
NEXT_ID = [0]
LOCK = threading.Lock()
LAST_PLOT = {"html": "<p style='font-family:sans-serif;padding:2em'>No analysis yet.</p>"}
CFG = {"object": "bubble wrap", "min_cluster_size": 6, "min_samples": 1,
       "epsilon": 0.25, "csv": "ideas.csv"}
_MODEL = {"m": None, "failed": False}

SEED_IDEAS = [
    # protection / packaging
    "wrap fragile dishes when moving", "cushion for shipping a laptop",
    "padding inside a helmet", "protect plants from frost by wrapping pots",
    "insulate a window in winter", "line a pet carrier so it's softer",
    # play / stress relief
    "pop it for stress relief", "a popping contest with friends",
    "fidget toy for anxious hands", "make a hopscotch course you pop as you go",
    "sensory toy for toddlers", "pop it as a metronome for rhythm practice",
    # art / craft
    "print texture stamp: dip in paint and press on paper",
    "make bubble-wrap paintings", "jellyfish sculpture for a school project",
    "use as texture in mixed-media art", "costume material for an alien suit",
    "decorate a bulletin board with fake water droplets",
    # clothing / body
    "cushion insoles for uncomfortable shoes", "wrist pad for the keyboard",
    "knee pads while gardening", "makeshift ice pack wrap",
    "body armor for a Halloween costume",
    # home / DIY
    "keep a cold drink cold inside a lunch bag", "line drawers to stop things sliding",
    "sound dampening for a drum practice room", "anti-slip mat under a rug",
    "greenhouse insulation for seedlings", "cushion the feet of furniture on a wood floor",
    # weird / playful
    "sleeping bag mattress on a camping trip", "float for a fishing line",
    "bubble wrap suit for a sumo-style game", "pretend it's rain in a school play",
    "a prank: cover a friend's whole desk", "use as a boat bumper on a dock",
    "make a pretend snowfield diorama", "shipwreck raft padding for a toy",
    "wrap a birthday gift in something you can pop", "train a cat to walk on odd textures",
    "measure hand-eye coordination by timing popping",
]


# --------------------------------------------------------------------------
# Analysis pipeline
# --------------------------------------------------------------------------
def _load_model():
    if _MODEL["m"] is not None or _MODEL["failed"]:
        return _MODEL["m"]
    try:
        from sentence_transformers import SentenceTransformer
        _MODEL["m"] = SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] sentence-transformers unavailable ({e}); using TF-IDF fallback")
        _MODEL["failed"] = True
    return _MODEL["m"]


def embed(texts):
    model = _load_model()
    if model is not None:
        return np.asarray(model.encode(texts, normalize_embeddings=True)), "MiniLM sentence embeddings"
    from sklearn.feature_extraction.text import TfidfVectorizer
    X = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit_transform(texts).toarray()
    return X, "TF-IDF (fallback)"


def get_clusterer():
    """Same settings as the analysis notebook (HDBSCAN, euclidean on normalized vectors, EOM)."""
    try:
        import hdbscan
        HDBSCAN = hdbscan.HDBSCAN
    except ImportError:  # sklearn >= 1.3 ships an equivalent implementation
        from sklearn.cluster import HDBSCAN
    return HDBSCAN(
        min_cluster_size=CFG["min_cluster_size"],
        min_samples=CFG["min_samples"],
        cluster_selection_epsilon=float(CFG["epsilon"]),
        metric="euclidean",
        cluster_selection_method="eom",
    )


def cluster_ideas(X):
    from sklearn.preprocessing import normalize
    vectors = normalize(X)
    if len(vectors) < CFG["min_cluster_size"]:
        return np.full(len(vectors), -1)
    return get_clusterer().fit_predict(vectors)


def project_2d(X):
    # t-SNE, as in the notebook, but perplexity must be < n_samples for small live samples
    from sklearn.manifold import TSNE
    n = len(X)
    perp = max(2, min(30, (n - 1) // 3))
    return TSNE(n_components=2, perplexity=perp, init="pca", random_state=0).fit_transform(X)


def jitter_duplicates(XY, texts, frac=0.012, seed=0):
    """Nudge repeated ideas apart so they show as separate dots (clustering is unaffected:
    this is applied to the 2D plot coordinates only). The first copy stays put; the others
    get a small random offset, ~1.2% of the plot span."""
    XY = XY.copy()
    rng = np.random.RandomState(seed)
    span = float(np.ptp(XY, axis=0).max()) or 1.0
    groups = {}
    for i, t in enumerate(texts):
        groups.setdefault(t.strip().lower(), []).append(i)
    for idx in groups.values():
        for i in idx[1:]:
            XY[i] += rng.normal(0, frac * span, size=2)
    return XY


def cluster_keywords(texts, labels, top=3):
    """Top TF-IDF terms per (non-noise) cluster, used as readable legend labels."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    ks = sorted(set(labels) - {-1})
    if not ks:
        return {}
    docs = [" ".join(t for t, l in zip(texts, labels) if l == k) for k in ks]
    vec = TfidfVectorizer(stop_words="english")
    M = vec.fit_transform(docs).toarray()
    terms = np.array(vec.get_feature_names_out())
    return {k: ", ".join(terms[np.argsort(M[i])[::-1][:top]]) for i, k in enumerate(ks)}


def wrap(s, width=40):
    words, lines, cur = s.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    return "<br>".join(lines)


MAX_SHOWN = 10  # clusters shown individually in legend / labels
COLORBLIND = ["#0173b2", "#de8f05", "#029e73", "#d55e00", "#cc78bc",
              "#ca9161", "#fbafe4", "#ece133", "#56b4e9", "#000000"]  # seaborn 'colorblind'


def run_analysis(texts):
    import plotly.graph_objects as go
    t0 = time.time()
    texts_arr = np.array(texts)

    X, emb_name = embed(texts)                 # <- swap in your embedding model in embed()
    labels = np.asarray(cluster_ideas(X))      # -1 = noise
    XY = jitter_duplicates(project_2d(X), texts)   # separate repeated ideas visually
    kw = cluster_keywords(texts, labels)

    real = [c for c in sorted(set(labels)) if c != -1]
    sizes = {c: int((labels == c).sum()) for c in real}
    largest = max(sizes, key=sizes.get) if sizes else None
    n_noise = int((labels == -1).sum())

    fig = go.Figure()
    # noise first so it sits behind the clusters
    if n_noise:
        m = labels == -1
        fig.add_trace(go.Scatter(
            x=XY[m, 0], y=XY[m, 1], mode="markers", name=f"Unclustered (n={n_noise})",
            marker=dict(size=12, color="lightgray", opacity=0.7),
            text=[wrap(t) for t in texts_arr[m]], hovertemplate="%{text}<extra></extra>"))
    # With hundreds of ideas HDBSCAN can return many clusters. Give the 10 largest their own
    # colour/legend entry/label; lump the rest into one group so the plot stays readable.
    real_sorted = sorted(real, key=lambda c: -sizes[c])
    top, rest = real_sorted[:MAX_SHOWN], real_sorted[MAX_SHOWN:]
    for i, c in enumerate(top):
        m = labels == c
        col = COLORBLIND[i % len(COLORBLIND)]
        star = "\u2605 " if c == largest else ""
        fig.add_trace(go.Scatter(
            x=XY[m, 0], y=XY[m, 1], mode="markers",
            name=f"{star}{kw[c]}  (n={sizes[c]})",
            marker=dict(size=15 if c == largest else 13, color=col, opacity=0.85,
                        line=dict(width=2.5 if c == largest else 1,
                                  color="black" if c == largest else "white")),
            text=[wrap(t) for t in texts_arr[m]], hovertemplate="%{text}<extra></extra>"))
        fig.add_annotation(x=XY[m, 0].mean(), y=XY[m, 1].mean(), text=f"<b>{kw[c]}</b>",
                           showarrow=False, font=dict(size=14, color=col),
                           bgcolor="rgba(255,255,255,0.75)", borderpad=3, yshift=26)
    if rest:
        m = np.isin(labels, rest)
        fig.add_trace(go.Scatter(
            x=XY[m, 0], y=XY[m, 1], mode="markers",
            name=f"{len(rest)} smaller clusters (n={int(m.sum())})",
            marker=dict(size=11, color="#6b7f95", opacity=0.75, line=dict(width=1, color="white")),
            text=[wrap(t) for t in texts_arr[m]], hovertemplate="%{text}<extra></extra>"))

    fig.update_layout(
        title=dict(text=f"Creative uses for {html.escape(CFG['object'])}: {len(texts)} ideas, "
                        f"{len(real)} clusters, {n_noise} unclustered",
                   font=dict(size=22)),
        template="plotly_white", height=680,
        font=dict(family="Avenir, Helvetica, Arial, sans-serif"),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        legend=dict(font=dict(size=14), title="Cluster themes (\u2605 = largest)"),
        margin=dict(l=20, r=20, t=70, b=30))
    fig.add_annotation(
        xref="paper", yref="paper", x=0, y=-0.03, showarrow=False, font=dict(size=11, color="gray"),
        text=(f"{emb_name} \u2192 HDBSCAN (min_cluster_size={CFG['min_cluster_size']}, "
              f"min_samples={CFG['min_samples']}, eps={CFG['epsilon']}) \u2192 t-SNE"))

    # representative ideas from the largest cluster (same as the notebook: sample up to 10, seed 42)
    if largest is not None:
        pool = texts_arr[labels == largest]
        rng = np.random.RandomState(42)
        rep = rng.choice(pool, size=min(10, len(pool)), replace=False)
        items = "".join(f"<li>{html.escape(str(t))}</li>" for t in rep)
        extra = (f"<h3>Largest cluster: {html.escape(kw[largest])} (n={sizes[largest]})</h3>"
                 f"<ul>{items}</ul>")
    else:
        extra = ("<h3>No clusters found</h3><p>Everything was labelled noise. "
                 "Try a smaller --min-cluster-size or wait for more ideas.</p>")

    page = ("<!doctype html><meta charset=utf-8><body style=\"font-family:Avenir,Helvetica,Arial,"
            "sans-serif;margin:0\">" + fig.to_html(full_html=False, include_plotlyjs=True) +
            f"<div style='padding:0 32px 32px;font-size:1.05rem'>{extra}</div></body>")
    print(f"[analysis] n={len(texts)} clusters={len(real)} noise={n_noise} "
          f"largest={sizes.get(largest)} {emb_name} in {time.time() - t0:.1f}s")
    return page


# --------------------------------------------------------------------------
# Web app
# --------------------------------------------------------------------------
app = Flask(__name__)


def presenter_only():
    if request.remote_addr not in ("127.0.0.1", "::1"):
        abort(403)


AUDIENCE_HTML = """<!doctype html><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Creative uses</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:520px;margin:0 auto;padding:24px;background:#fafafa}
 h1{font-size:1.5rem;margin:.2em 0} p{color:#555}
 textarea{width:100%;font-size:1.1rem;padding:12px;border:2px solid #ccc;border-radius:10px;box-sizing:border-box}
 button{width:100%;margin-top:12px;padding:14px;font-size:1.1rem;border:0;border-radius:10px;background:#4E79A7;color:#fff}
 #msg{margin-top:14px;font-weight:600;color:#2a7}
</style>
<h1>Think of creative uses for <u>__OBJECT__</u></h1>
<p>For this task, you'll be asked to come up with creative uses for everyday objects. When we say "creative" we mean how original and useful the idea is. Submit your ideas one at a time below. </p>
<textarea id=t rows=3 maxlength=200 placeholder="e.g. ..." autofocus></textarea>
<button onclick=send()>Submit idea</button>
<div id=msg></div>
<script>
let n=0;
async function send(){
  const t=document.getElementById('t');
  const text=t.value.trim(); if(!text) return;
  const r=await fetch('/submit',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
  const j=await r.json();
  if(j.ok){n++;t.value='';t.focus();document.getElementById('msg').textContent='Got it! ('+n+' sent) Another?';}
  else document.getElementById('msg').textContent=j.error||'Error';
}
document.getElementById('t').addEventListener('keydown',e=>{if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();send();}});
</script>"""

PRESENTER_HTML = """<!doctype html><meta charset=utf-8><title>Presenter</title>
<style>
 body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee}
 #top{display:flex;gap:24px;align-items:center;padding:14px 24px;background:#1b1b1b;flex-wrap:wrap}
 #qr{width:150px;height:150px;background:#fff;padding:6px;border-radius:8px;box-sizing:border-box;display:flex;align-items:center;justify-content:center}
 #url{font-size:1.6rem;font-weight:700;color:#8fd}
 #count{font-size:3rem;font-weight:800}
 button{font-size:1.2rem;padding:12px 22px;border:0;border-radius:10px;margin-right:8px;cursor:pointer}
 #run{background:#59A14F;color:#fff} #clr{background:#444;color:#fff}
 #main{display:flex;height:calc(100vh - 180px)}
 #list{width:300px;overflow:auto;padding:10px;font-size:.95rem;background:#161616}
 .it{padding:5px 8px;border-bottom:1px solid #2a2a2a;display:flex;justify-content:space-between;gap:8px}
 .it a{color:#e66;cursor:pointer;text-decoration:none}
 iframe{flex:1;border:0;background:#fff}
</style>
<div id=top>
 __QR__
 <div><div style="color:#aaa">Join at</div><div id=url>__URL__</div></div>
 <div><div id=count>0</div><div style="color:#aaa">ideas</div></div>
 <div><button id=run onclick=runA()>Run analysis</button>
      <button id=clr onclick=clr()>Clear</button>
      <span id=st style="color:#aaa"></span></div>
</div>
<div id=main><div id=list></div><iframe id=f></iframe></div>
<script>
async function poll(){
  const j=await (await fetch('/ideas')).json();
  document.getElementById('count').textContent=j.length;
  document.getElementById('list').innerHTML=j.slice().reverse().map(i=>
    '<div class=it><span>'+i.text.replace(/[<>&]/g,c=>({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))+
    '</span><a onclick="del('+i.id+')">✕</a></div>').join('');
}
async function del(id){await fetch('/delete/'+id,{method:'POST'});poll();}
async function runA(){
  const st=document.getElementById('st'); st.textContent='Analysing...';
  const r=await fetch('/analyze',{method:'POST'}); const j=await r.json();
  if(j.ok){document.getElementById('f').src='/plot?ts='+Date.now();st.textContent='';}
  else st.textContent=j.error;
}
async function clr(){if(confirm('Delete all ideas?')){await fetch('/clear',{method:'POST'});poll();}}
setInterval(poll,2000);poll();
</script>"""


def qr_svg(url):
    try:
        import qrcode
        import qrcode.image.svg
        img = qrcode.make(url, image_factory=qrcode.image.svg.SvgPathImage, box_size=10)
        import io
        buf = io.BytesIO()
        img.save(buf)
        return f'<div id=qr>{buf.getvalue().decode()}</div>'
    except Exception:  # noqa: BLE001
        return ""


@app.get("/")
def audience():
    return AUDIENCE_HTML.replace("__OBJECT__", html.escape(CFG["object"]))


@app.post("/submit")
def submit():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()[:200]
    if len(text) < 2:
        return jsonify(ok=False, error="Too short")
    with LOCK:
        IDEAS.append({"id": NEXT_ID[0], "text": text})
        NEXT_ID[0] += 1
        with open(CFG["csv"], "a", newline="") as f:
            csv.writer(f).writerow([time.strftime("%H:%M:%S"), text])
    return jsonify(ok=True)


@app.get("/present")
def present():
    presenter_only()
    url = f"http://{app.config['LAN_IP']}:{app.config['PORT']}"
    return PRESENTER_HTML.replace("__URL__", url).replace("__QR__", qr_svg(url))


@app.get("/ideas")
def ideas():
    presenter_only()
    return jsonify(IDEAS)


@app.post("/delete/<int:i>")
def delete(i):
    presenter_only()
    with LOCK:
        IDEAS[:] = [x for x in IDEAS if x["id"] != i]
    return jsonify(ok=True)


@app.post("/clear")
def clear():
    presenter_only()
    with LOCK:
        IDEAS.clear()
    return jsonify(ok=True)


@app.post("/analyze")
def analyze():
    presenter_only()
    with LOCK:
        texts = [i["text"] for i in IDEAS]
    if len(texts) < 6:
        return jsonify(ok=False, error="Need at least 6 ideas")
    try:
        LAST_PLOT["html"] = run_analysis(texts)
    except Exception as e:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        return jsonify(ok=False, error=str(e))
    return jsonify(ok=True)


@app.get("/plot")
def plot():
    presenter_only()
    return Response(LAST_PLOT["html"], mimetype="text/html")


def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except Exception:  # noqa: BLE001
        return "127.0.0.1"
    finally:
        s.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default="bubble wrap")
    ap.add_argument("--port", type=int, default=5000)
    ap.add_argument("--min-cluster-size", type=int, default=6)
    ap.add_argument("--min-samples", type=int, default=1)
    ap.add_argument("--epsilon", type=float, default=0.25, help="cluster_selection_epsilon")
    ap.add_argument("--seed", action="store_true", help="preload sample ideas")
    a = ap.parse_args()

    CFG.update(object=a.object, min_cluster_size=a.min_cluster_size,
               min_samples=a.min_samples, epsilon=a.epsilon)
    if a.seed:
        for t in SEED_IDEAS:
            IDEAS.append({"id": NEXT_ID[0], "text": t})
            NEXT_ID[0] += 1
    app.config.update(LAN_IP=lan_ip(), PORT=a.port)

    threading.Thread(target=_load_model, daemon=True).start()  # warm up the model
    print(f"\n  Audience URL : http://{app.config['LAN_IP']}:{a.port}")
    print(f"  Presenter    : http://localhost:{a.port}/present\n")
    app.run(host="0.0.0.0", port=a.port, threaded=True)


if __name__ == "__main__":
    main()
