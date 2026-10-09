# PCB Defect Inspector

Web demo of a final-year project: reference-guided PCB defect detection (template comparison inside a Faster R-CNN
with NNVR) plus a learned deblur + binarize stage for blurry images.

- **Our system:** blurry image → deblur + binarize (U-Net) → template-comparison detector → defects with type and confidence.
- **Experiments:** ablation (plain / + NNVR / + template), four ways to handle blur, the cleaner with and without the
  template, and real colour photos (PKU-Market-PCB).
- **Your own images:** upload a board image and its defect-free template.

Sample images are from the DeepPCB and PKU-Market-PCB **test** splits (never used in training).

Run locally: `pip install -r requirements.txt` then `streamlit run streamlit_app.py`.
The model files are downloaded on first start from the address in `weights_url.txt`.
