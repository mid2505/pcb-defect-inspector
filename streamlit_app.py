"""PCB Defect Inspector: web demo for the reference-guided PCB defect detection project.

Run locally:  streamlit run streamlit_app.py   (needs the model files in ./weights, see code/deploy_web_app.ipynb)
"""
import os
import random
import numpy as np
import cv2
import streamlit as st

import pcb_core as core
import pipelines as P

st.set_page_config(page_title="PCB Defect Inspector", page_icon="🔍", layout="wide")

EXTRA_CSS = """
.block-container {max-width: 1360px; padding-top: 2.2rem;}
#hdr h1 {font-size: 28px; margin: 0 0 4px 0; font-weight: 700; color: #111827; padding: 0;}
#hdr p {margin: 0 0 6px 0; color: #4b5563; font-size: 15px;}
.stephead {font-weight: 600; font-size: 15px; color: #1f2937; margin: 10px 0 -6px 0;}
"""
st.html(f"<style>{P.CSS}{EXTRA_CSS}</style>")
st.html('<div id="hdr"><h1>PCB Defect Inspector</h1><p>Finds open, short, mousebite, spur, spurious copper and pin-hole '
        'defects by comparing a board image with its defect-free template, even when the image is blurry.</p></div>')


@st.cache_resource(show_spinner="Getting the models ready (only on the first start, about a minute)...")
def setup():
    core.ensure_weights()
    for n in ["final"]:            # the main model, so the first Run is quick
        if core.available(n):
            core.get(n)
    return True


setup()

SCAN, PHOTO = "Black-and-white scan (like DeepPCB)", "Colour photo (like PKU)"


def read_upload(f):
    if f is None:
        return None
    img = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def label_for(data, i):
    it = data[i]
    return f"{it['id']}  ·  " + ", ".join(sorted({P.NAMES[l] for l in it["labels"]}))


left, right = st.columns([4, 7], gap="large")

with left:
    # ---- step 1
    st.html('<div class="stephead">1 · What do you want to see?</div>')
    mode = st.radio("mode", [P.MODE_SYS, P.MODE_EXP], horizontal=True, label_visibility="collapsed")
    exp = st.selectbox("Experiment", P.EXPS) if mode == P.MODE_EXP else None
    st.markdown(P.desc_for(mode, exp))

    # ---- step 2
    st.html('<div class="stephead">2 · Choose an image</div>')
    if mode == P.MODE_SYS:
        sources = [P.SRC_DEEP, P.SRC_PKU, P.SRC_UP]
    elif exp == P.EXP_PKU:
        sources = [P.SRC_PKU]
    elif exp == P.EXP_TPL:
        sources = [P.SRC_DEEP, P.SRC_UP]
    else:
        sources = [P.SRC_DEEP]
    source = st.radio("source", sources, horizontal=True, label_visibility="collapsed", key=f"src_{mode}_{exp}")

    sel, up_test, up_temp, photo = None, None, None, False
    if source != P.SRC_UP:
        data = P.PKU if source == P.SRC_PKU else P.DEEP
        c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
        defect = c1.selectbox("Show images with", ["All types"] + P.NAMES[1:], key=f"def_{source}")
        idxs = [i for i, it in enumerate(data) if defect == "All types" or defect in {P.NAMES[l] for l in it["labels"]}]
        pick_key = f"pick_{source}_{defect}"
        if c2.button("Random", width="stretch") and idxs:
            st.session_state[pick_key] = random.choice(idxs)
        if idxs:
            sel = st.selectbox(f"Image ({len(idxs)} available)", idxs, key=pick_key,
                               format_func=lambda i: label_for(data, i))
            s = P.sample(source, sel)
            p1, p2 = st.columns(2)
            p1.image(s["test"], caption="Test image", width="stretch")
            p2.image(s["temp"], caption="Template (defect-free)", width="stretch")
        else:
            st.info("No images with this defect type.")
    else:
        if mode == P.MODE_SYS:
            photo = st.radio("Image type", [SCAN, PHOTO], horizontal=True) == PHOTO
        st.caption("Both images are required: the system finds defects by comparing the board with its "
                   "defect-free template. Use a template of the same board, lined up with the test image.")
        u1, u2 = st.columns(2)
        f_test = u1.file_uploader("Board to inspect", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
        f_temp = u2.file_uploader("Defect-free template", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
        up_test, up_temp = read_upload(f_test), read_upload(f_temp)
        if up_test is not None:
            u1.image(up_test, width="stretch")
        if up_temp is not None:
            u2.image(up_temp, width="stretch")

    # ---- step 3
    no_blur_default = source in (P.SRC_PKU, P.SRC_UP) or exp in (P.EXP_ABL, P.EXP_PKU)
    st.html('<div class="stephead">3 · Simulate camera blur (optional)</div>')
    ctx = f"{mode}_{exp}_{source}"
    b1, b2 = st.columns([3, 2])
    kind = b1.selectbox("Blur type", core.BLUR_TYPES, index=0 if no_blur_default else 1, key=f"kind_{ctx}",
                        help="Adds blur to the test image to show how the system copes. "
                             "Keep 'None' if your image is already blurry.")
    level = b2.slider("Level", 1, 5, 4, key=f"lvl_{ctx}",
                      disabled=kind == "None" or "unseen" in kind)
    with st.expander("Advanced"):
        thr = st.slider("Confidence threshold", 0.05, 0.95, 0.5 if mode == P.MODE_EXP else 0.8, 0.05, key=f"thr_{mode}",
                        help="Boxes below this confidence are hidden. Our system uses 0.8.")
    run = st.button("Run", type="primary", width="stretch")

with right:
    if run:
        with st.spinner("Running the models..."):
            html = P.on_run(mode, exp, source, sel, up_test, up_temp, kind, level, thr, photo)
        st.session_state["result"] = (ctx, html)
    res = st.session_state.get("result")
    st.html(res[1] if res and res[0] == ctx else P.PLACEHOLDER)

with st.expander("About this project"):
    st.image(os.path.join(P.ROOT, "assets", "system_design.jpg"), width="stretch")
    st.html(P.table([["DeepPCB (official split)", "-", "99.43"], ["DeepPCB (8:1:1, as base paper)", "99.00", "<b>99.59</b>"],
                     ["PKU-Market-PCB (8:1:1)", "96.4", "<b>99.24</b>"], ["DeepPCB, heavy blur (level 5)", "-", "96.5"]],
                    ["dataset", "MGRE-Net (base paper) mAP@0.5", "ours mAP@0.5"]))
