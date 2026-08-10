"""
split_dashboard.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Memisahkan dua dashboard besar menjadi panel-panel individual
dengan tema terang (light theme) yang siap cetak.

Dashboard 1 — iot_slm_evaluation.png  (5 row × beberapa panel)
Dashboard 2 — iot_slm_xai_dashboard.png (6 row × beberapa panel)

Output: /mnt/user-data/outputs/panels/  (PNG, 150 DPI, light bg)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
import numpy as np
import os, textwrap

# ─── Output directory ────────────────────────────────────────────────────────
OUT_DIR = os.getcwd()
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Print-friendly light theme colours ──────────────────────────────────────
BG        = (255, 255, 255)          # white page
PANEL_BG  = (245, 248, 252)          # very light blue-grey
BORDER    = (180, 195, 215)          # cool grey border
TITLE_BG  = (26,  72, 138)           # navy header bar
TITLE_FG  = (255, 255, 255)          # white text on header
LABEL_FG  = (30,  50,  90)           # dark navy body text
CAP_FG    = (90, 110, 140)           # muted caption
ACCENT    = (13, 122, 110)           # teal accent
MARGIN    = 36                        # outer padding (px)
HEADER_H  = 52                        # title bar height
FOOTER_H  = 38                        # caption bar height
CORNER_R  = 12                        # rounded corner radius

PRINT_DPI = 150

# ─── Font helpers ─────────────────────────────────────────────────────────────
def _font(size, bold=False):
    """Try to load a system font; fall back to PIL default."""
    candidates_bold = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    candidates_reg = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]
    paths = candidates_bold if bold else candidates_reg
    for p in paths:
        if os.path.exists(p):
            try: return ImageFont.truetype(p, size)
            except: pass
    return ImageFont.load_default()

FONT_TITLE  = _font(22, bold=True)
FONT_LABEL  = _font(16, bold=True)
FONT_BODY   = _font(14)
FONT_CAP    = _font(13)
FONT_SMALL  = _font(12)

# ─── Core utilities ───────────────────────────────────────────────────────────
def rounded_rect(draw, xy, radius, fill, outline=None, width=1):
    x0, y0, x1, y1 = xy
    r = radius
    draw.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=fill,
                           outline=outline, width=width)

def invert_dark_region(img_crop):
    """
    Convert a dark-background plot region to a light-background version.
    Strategy:
      1. Convert to numpy
      2. Invert the image
      3. Adjust contrast / brightness so axes stay legible
    """
    arr = np.array(img_crop).astype(np.float32)

    # Detect if background is dark (mean luminance < 100)
    gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
    mean_lum = gray.mean()

    if mean_lum > 140:
        # Already light-ish; just return as-is
        return img_crop

    # --- invert ---
    rgb = arr[:, :, :3]
    inv = 255.0 - rgb

    # Stretch contrast
    lo, hi = inv.min(), inv.max()
    if hi > lo:
        inv = (inv - lo) / (hi - lo) * 230.0

    # Reconstruct
    out = np.clip(inv, 0, 255).astype(np.uint8)
    if arr.shape[2] == 4:
        alpha = arr[:, :, 3:4].astype(np.uint8)
        out = np.concatenate([out, alpha], axis=2)
        return Image.fromarray(out, 'RGBA').convert('RGB')
    return Image.fromarray(out, 'RGB')


def make_panel(crop_img, title, caption, panel_w=None, prefix=""):
    """
    Wrap a cropped image region into a print-ready card with
    navy title bar, white background, and caption footer.
    Returns a PIL Image.
    """
    light_crop = invert_dark_region(crop_img.convert("RGB"))

    # Enhance contrast after inversion
    light_crop = ImageEnhance.Contrast(light_crop).enhance(1.15)
    light_crop = ImageEnhance.Sharpness(light_crop).enhance(1.10)

    src_w, src_h = light_crop.size
    if panel_w is None:
        panel_w = src_w + 2 * MARGIN

    # Scale crop to fit panel width
    scale  = (panel_w - 2 * MARGIN) / src_w
    dst_w  = panel_w - 2 * MARGIN
    dst_h  = int(src_h * scale)
    resized = light_crop.resize((dst_w, dst_h), Image.LANCZOS)

    total_h = HEADER_H + MARGIN//2 + dst_h + MARGIN//2 + FOOTER_H
    canvas  = Image.new("RGB", (panel_w, total_h), BG)
    draw    = ImageDraw.Draw(canvas)

    # ── Navy title bar
    draw.rectangle([0, 0, panel_w, HEADER_H], fill=TITLE_BG)

    # Teal accent strip
    draw.rectangle([0, HEADER_H-4, panel_w, HEADER_H], fill=ACCENT)

    # Title text (wrap if long)
    max_chars = (panel_w - 2*MARGIN) // 11
    wrapped   = textwrap.shorten(title, width=max_chars, placeholder="…")
    tw = draw.textlength(wrapped, font=FONT_TITLE)
    tx = (panel_w - tw) // 2
    draw.text((tx, (HEADER_H - 22)//2), wrapped, font=FONT_TITLE, fill=TITLE_FG)

    # ── Plot area with light panel background and border
    plot_y = HEADER_H + MARGIN // 2
    rounded_rect(draw,
                 [MARGIN//2, plot_y - 4,
                  panel_w - MARGIN//2, plot_y + dst_h + 4],
                 CORNER_R, fill=PANEL_BG, outline=BORDER, width=1)

    canvas.paste(resized, (MARGIN, plot_y))

    # ── Caption bar
    cap_y = plot_y + dst_h + MARGIN // 2
    draw.rectangle([0, cap_y, panel_w, total_h], fill=(235, 240, 248))
    draw.line([(0, cap_y), (panel_w, cap_y)], fill=BORDER, width=1)

    cap_chars = (panel_w - 2*MARGIN) // 8
    cap_text  = textwrap.shorten(caption, width=cap_chars, placeholder="…")
    draw.text((MARGIN, cap_y + (FOOTER_H - 13)//2),
              cap_text, font=FONT_CAP, fill=CAP_FG)

    return canvas


def add_page_number(img, num, total, label=""):
    """Stamp a small page-number badge bottom-right."""
    draw = ImageDraw.Draw(img)
    text = f"{label}  {num}/{total}" if label else f"{num}/{total}"
    tw   = draw.textlength(text, font=FONT_SMALL)
    x    = img.width - int(tw) - MARGIN
    y    = img.height - FOOTER_H + (FOOTER_H - 12)//2
    draw.text((x, y), text, font=FONT_SMALL, fill=ACCENT)
    return img


def save(img, filename):
    path = os.path.join(OUT_DIR, filename)
    img.save(path, dpi=(PRINT_DPI, PRINT_DPI), optimize=True)
    kb = os.path.getsize(path) // 1024
    print(f"  ✓  {filename:<55s}  {img.size[0]}×{img.size[1]}px  {kb} KB")
    return path


# ════════════════════════════════════════════════════════════════════════════════
#  DASHBOARD 1 — iot_slm_evaluation.png
#  Layout: 5 rows
#   Row 0: 3 panels  (loss curve | mae/rmse | r2)
#   Row 1: 4 panels  (pred vs actual × 4 sensors)
#   Row 2: 4 panels  (raw ts + anomaly × 4)
#   Row 3: 3 panels  (daily | weekly | monthly)
#   Row 4: 3 panels  (residual | scatter | anomaly count)
# ════════════════════════════════════════════════════════════════════════════════

def split_eval_dashboard():
    print("\n── Dashboard 1: iot_slm_evaluation.png ──────────────────────────")
    src = Image.open("/mnt/user-data/outputs/iot_slm_evaluation.png").convert("RGB")
    W, H = src.size                      # 2367 × 3041
    PANEL_OUT_W = 900                    # output width per single panel

    # ── Row boundaries (approximate, as fraction of H)
    # Determined by the 5-row GridSpec with hspace=0.44, top=0.970, bottom=0.02
    # Row separators: row_start, mid-gap, ..., row_end (tight crop)
    row_tops = [91, 612, 1227, 1843, 2458, 2980]

    panels = []  # (crop_pil, title, caption, width_hint)

    # ── ROW 0 – 3 panels (left=0.06, right=0.97, wspace=0.35)
    r0_y0, r0_y1 = row_tops[0], row_tops[1]
    # Exact 3-col positions from GridSpec(1,3, wspace=0.35, left=0.06, right=0.97)
    col3_xs = [(142, 724), (927, 1510), (1713, 2295)]
    r0_titles   = ["Training Loss Curve (MSE)", "MAE & RMSE per Sensor", "R² Score per Sensor"]
    r0_captions = [
        "Training (red) and validation (blue) MSE over 60 epochs; both converge without overfitting.",
        "Bar chart comparing Mean Absolute Error and RMSE for each of the four sensor variables.",
        "Coefficient of determination (R²) per sensor; values closer to 1.0 indicate better fit.",
    ]
    for i, (x0, x1) in enumerate(col3_xs):
        crop = src.crop((x0, r0_y0, x1, r0_y1))
        panels.append((crop, f"1.{i+1} {r0_titles[i]}", r0_captions[i], PANEL_OUT_W))

    # ── ROW 1 – 4 panels (pred vs actual)
    r1_y0, r1_y1 = row_tops[1], row_tops[2]
    sensors  = ["Temperature (BMP280)", "Pressure (BMP280)", "Humidity (AHT20)", "Light (BH1750)"]
    # Exact 4-col positions from GridSpec(1,4, wspace=0.35, left=0.06, right=0.97)
    col4_xs = [(142, 568), (717, 1144), (1293, 1720), (1869, 2295)]
    for i, (x0, x1) in enumerate(col4_xs):
        crop = src.crop((x0, r1_y0, x1, r1_y1))
        cap  = f"Predicted (white dashed) vs actual (colored) values for {sensors[i]} over 200 validation samples."
        panels.append((crop, f"2.{i+1} Pred vs Actual — {sensors[i]}", cap, PANEL_OUT_W))

    # ── ROW 2 – 4 panels (raw ts + anomaly)
    r2_y0, r2_y1 = row_tops[2], row_tops[3]
    for i, (x0, x1) in enumerate(col4_xs):
        crop = src.crop((x0, r2_y0, x1, r2_y1))
        cap  = f"Raw time-series for {sensors[i]}; red markers indicate statistically irregular readings (>2.5σ)."
        panels.append((crop, f"3.{i+1} Anomaly Overlay — {sensors[i]}", cap, PANEL_OUT_W))

    # ── ROW 3 – 3 panels (daily | weekly | monthly)
    r3_y0, r3_y1 = row_tops[3], row_tops[4]
    r3_titles   = ["Daily Profile — Temperature", "Weekly Pattern — Temp & Humidity", "Monthly Trend — Temp & Light"]
    r3_captions = [
        "Average 24-hour temperature cycle (line) with ±1 standard deviation shading.",
        "Mean temperature (left axis) and humidity (right axis) for each day of the week.",
        "Month-by-month average for temperature and light intensity showing seasonal drift.",
    ]
    for i, (x0, x1) in enumerate(col3_xs):
        crop = src.crop((x0, r3_y0, x1, r3_y1))
        panels.append((crop, f"4.{i+1} {r3_titles[i]}", r3_captions[i], PANEL_OUT_W))

    # ── ROW 4 – 3 panels (residual | scatter | anomaly count)
    r4_y0, r4_y1 = row_tops[4], row_tops[5]
    r4_titles   = ["Residual Distribution", "Scatter: Actual vs Predicted (Temp)", "Anomaly Count per Sensor"]
    r4_captions = [
        "Histogram of prediction errors (pred − actual) for all four sensors; zero-centred = unbiased.",
        "Each point is one validation sample; points on the diagonal indicate perfect agreement.",
        "Total number of statistically irregular readings detected per sensor using the 2.5σ threshold.",
    ]
    for i, (x0, x1) in enumerate(col3_xs):
        crop = src.crop((x0, r4_y0, x1, r4_y1))
        panels.append((crop, f"5.{i+1} {r4_titles[i]}", r4_captions[i], PANEL_OUT_W))

    # ── Also save full rows as combined sheets
    row_meta = [
        ("Row 1 — Training & Accuracy Metrics",       row_tops[0], row_tops[1]),
        ("Row 2 — Predicted vs Actual (All Sensors)", row_tops[1], row_tops[2]),
        ("Row 3 — Raw Time-Series + Anomaly Overlay", row_tops[2], row_tops[3]),
        ("Row 4 — Temporal Pattern Comparison",       row_tops[3], row_tops[4]),
        ("Row 5 — Residual Analysis & Summary",       row_tops[4], row_tops[5]),
    ]

    # Save individual panels
    saved = []
    total = len(panels)
    for idx, (crop, title, cap, pw) in enumerate(panels, 1):
        img = make_panel(crop, title, cap, pw)
        img = add_page_number(img, idx, total, "Eval")
        fname = f"eval_{idx:02d}_{title[:35].replace(' ','_').replace('/','').replace(':','')}.png"
        saved.append(save(img, fname))

    # Save full-row sheets (useful for side-by-side printing)
    for ri, (row_title, y0, y1) in enumerate(row_meta, 1):
        crop = src.crop((100, y0, 2310, y1))
        img  = make_panel(crop, f"Dashboard 1 — {row_title}", "IoT-SLM Evaluation Dashboard · Universitas Mataram", 1600)
        img  = add_page_number(img, ri, len(row_meta), "Eval-Row")
        fname = f"eval_row{ri:02d}_{row_title[:30].replace(' ','_').replace('—','').strip()}.png"
        saved.append(save(img, fname))

    return saved


# ════════════════════════════════════════════════════════════════════════════════
#  DASHBOARD 2 — iot_slm_xai_dashboard.png
#  Layout: 6 rows
#   Row 0: 3 panels  (global importance | concept scores | method agreement)
#   Row 1: 2 panels  (saliency normal | saliency anomaly)
#   Row 2: 2 panels  (IG normal | IG anomaly)
#   Row 3: 3 panels  (attention curve | attn matrix | LIME coef)
#   Row 4: 3 panels  (SHAP compare | CF delta | all concepts)
#   Row 5: 1 panel   (NL narrative text box)
# ════════════════════════════════════════════════════════════════════════════════

def split_xai_dashboard():
    print("\n── Dashboard 2: iot_slm_xai_dashboard.png ───────────────────────")
    src = Image.open("/mnt/user-data/outputs/iot_slm_xai_dashboard.png").convert("RGB")
    W, H = src.size                       # 2347 × 3239
    PANEL_OUT_W = 900

    # Row boundaries (GridSpec 6 rows, hspace=0.52, top=0.978, bottom=0.022)
    # Row separators: row_start, mid-gap, ..., row_end
    row_tops = [71, 524, 1071, 1619, 2166, 2713, 3167]

    panels = []

    # ── ROW 0 – 3 panels
    r0_y0, r0_y1 = row_tops[0], row_tops[1]
    # Exact 3-col positions from GridSpec(1,3, wspace=0.35, left=0.06, right=0.97)
    col3_xs = [(140, 718), (920, 1497), (1699, 2276)]
    r0_meta = [
        ("1.1 Global Feature Importance (SHAP vs IG)",
         "SHAP permutation (red) vs Integrated Gradients (cyan) averaged over 80 samples. Both methods agree: light intensity dominates."),
        ("1.2 Concept Scores — Normal vs Anomaly",
         "Six physics-grounded concept indicators compared between a normal window (green) and an anomaly window (red)."),
        ("1.3 Method Agreement Chart",
         "SHAP, IG, and LIME importance rankings for each sensor; convergence of lines indicates cross-method agreement."),
    ]
    for i, (x0, x1) in enumerate(col3_xs):
        crop = src.crop((x0, r0_y0, x1, r0_y1))
        panels.append((crop, r0_meta[i][0], r0_meta[i][1], PANEL_OUT_W))

    # ── ROW 1 – 2 panels (saliency heatmaps)
    r1_y0, r1_y1 = row_tops[1], row_tops[2]
    # Exact 2-col positions from GridSpec(1,2, wspace=0.25, left=0.06, right=0.97)
    col2_xs = [(140, 1049), (1367, 2276)]
    r1_meta = [
        ("2.1 Gradient Saliency — Normal Window",
         "Heatmap of |∂output/∂input|; brighter cells indicate timesteps and sensors that most directly shift the prediction. White star = peak."),
        ("2.2 Gradient Saliency — Anomaly Window",
         "Saliency concentrates around specific timesteps and sensors during the anomaly window, pinpointing the disturbance location."),
    ]
    for i, (x0, x1) in enumerate(col2_xs):
        crop = src.crop((x0, r1_y0, x1, r1_y1))
        panels.append((crop, r1_meta[i][0], r1_meta[i][1], PANEL_OUT_W))

    # ── ROW 2 – 2 panels (IG maps)
    r2_y0, r2_y1 = row_tops[2], row_tops[3]
    r2_meta = [
        ("3.1 Integrated Gradients — Normal Window",
         "Path-integral attribution map (blue=negative, red=positive). Satisfies the Completeness Axiom; sum equals f(x)−f(baseline)."),
        ("3.2 Integrated Gradients — Anomaly Window",
         "Strong red and blue regions identify which sensors and timesteps most positively and negatively drove the anomaly prediction."),
    ]
    for i, (x0, x1) in enumerate(col2_xs):
        crop = src.crop((x0, r2_y0, x1, r2_y1))
        panels.append((crop, r2_meta[i][0], r2_meta[i][1], PANEL_OUT_W))

    # ── ROW 3 – 3 panels (attention curve | attn matrix | LIME)
    r3_y0, r3_y1 = row_tops[3], row_tops[4]
    r3_meta = [
        ("4.1 Attention Rollout — Temporal Curve",
         "Per-timestep attention weight for normal (cyan) and anomaly (red) windows. Peak near t=21 ≈ 1.4h before forecast."),
        ("4.2 Attention Weight Matrix — Anomaly Window",
         "Full T×T attention heatmap; diagonal concentration shows the model attends primarily to recent timesteps for each query position."),
        ("4.3 LIME Surrogate Coefficients — Anomaly Window",
         "Average linear surrogate coefficients per sensor. Red bars push prediction upward; cyan bars pull it down."),
    ]
    for i, (x0, x1) in enumerate(col3_xs):
        crop = src.crop((x0, r3_y0, x1, r3_y1))
        panels.append((crop, r3_meta[i][0], r3_meta[i][1], PANEL_OUT_W))

    # ── ROW 4 – 3 panels (SHAP | CF | all concepts)
    r4_y0, r4_y1 = row_tops[4], row_tops[5]
    r4_meta = [
        ("5.1 SHAP Permutation — Normal vs Anomaly",
         "Importance from randomly shuffling each sensor; green=normal, red=anomaly. Humidity rises notably in anomaly windows."),
        ("5.2 Counterfactual Delta Map",
         "Minimal input perturbation required to reduce anomaly score to 0.1. Red=increase needed; blue=decrease. Score achieved: 0.1009."),
        ("5.3 All Concept Attribution Scores",
         "All six physics-grounded concept scores side-by-side for normal and anomaly windows. Light Anomaly reaches 1.0 in anomaly window."),
    ]
    for i, (x0, x1) in enumerate(col3_xs):
        crop = src.crop((x0, r4_y0, x1, r4_y1))
        panels.append((crop, r4_meta[i][0], r4_meta[i][1], PANEL_OUT_W))

    # ── ROW 5 – 1 panel (NL narrative full width)
    r5_y0, r5_y1 = row_tops[5], row_tops[6]
    crop = src.crop((80, r5_y0, 2290, r5_y1))
    panels.append((
        crop,
        "6.1 XAI Natural Language Explanation",
        "Automated narrative summarising all 7 XAI methods in plain English for the flagged anomaly sample.",
        1600
    ))

    # Save individual panels
    saved = []
    total = len(panels)
    for idx, (crop, title, cap, pw) in enumerate(panels, 1):
        img   = make_panel(crop, title, cap, pw)
        img   = add_page_number(img, idx, total, "XAI")
        slug  = title[:38].replace(' ','_').replace('/','').replace(':','').replace('—','')
        fname = f"xai_{idx:02d}_{slug}.png"
        saved.append(save(img, fname))

    # Full-row sheets
    row_meta = [
        ("Row 1 — Global Feature Importance & Method Agreement", row_tops[0], row_tops[1]),
        ("Row 2 — Gradient Saliency Heatmaps",                  row_tops[1], row_tops[2]),
        ("Row 3 — Integrated Gradients Attribution Maps",        row_tops[2], row_tops[3]),
        ("Row 4 — Attention Rollout, Matrix & LIME",             row_tops[3], row_tops[4]),
        ("Row 5 — SHAP, Counterfactual & Concept Scores",        row_tops[4], row_tops[5]),
        ("Row 6 — Natural Language XAI Narrative",               row_tops[5], row_tops[6]),
    ]
    for ri, (row_title, y0, y1) in enumerate(row_meta, 1):
        crop = src.crop((80, y0, 2290, y1))
        img  = make_panel(crop, f"Dashboard 2 — {row_title}",
                          "IoT-SLM XAI Dashboard · Universitas Mataram", 1600)
        img  = add_page_number(img, ri, len(row_meta), "XAI-Row")
        fname = f"xai_row{ri:02d}_{row_title[:30].replace(' ','_').replace('—','').strip()}.png"
        saved.append(save(img, fname))

    return saved


# ════════════════════════════════════════════════════════════════════════════════
#  COMBINED SUMMARY SHEET
# ════════════════════════════════════════════════════════════════════════════════

def make_index_sheet(all_saved):
    """
    Create an A4-style index page listing all generated panel filenames
    grouped by dashboard, suitable as a print cover page.
    """
    line_h  = 24
    n_lines = len(all_saved) + 8
    H_idx   = HEADER_H + MARGIN + n_lines * line_h + MARGIN
    W_idx   = 1240
    canvas  = Image.new("RGB", (W_idx, H_idx), BG)
    draw    = ImageDraw.Draw(canvas)

    # Header
    draw.rectangle([0, 0, W_idx, HEADER_H], fill=TITLE_BG)
    draw.rectangle([0, HEADER_H-4, W_idx, HEADER_H], fill=ACCENT)
    draw.text((MARGIN, (HEADER_H-22)//2),
              "Panel Index — IoT-SLM Dashboard Split", font=FONT_TITLE, fill=TITLE_FG)

    y = HEADER_H + MARGIN
    font_section = _font(15, bold=True)

    sections = {
        "Dashboard 1 — RNN Evaluation (eval_*)": [f for f in all_saved if "/eval_" in f],
        "Dashboard 2 — XAI Explainability (xai_*)": [f for f in all_saved if "/xai_" in f],
    }
    for sec_title, files in sections.items():
        draw.text((MARGIN, y), sec_title, font=font_section, fill=TITLE_BG)
        y += line_h + 4
        draw.line([(MARGIN, y), (W_idx - MARGIN, y)], fill=BORDER, width=1)
        y += 8
        for path in files:
            name = os.path.basename(path)
            draw.text((MARGIN + 16, y), f"• {name}", font=FONT_BODY, fill=LABEL_FG)
            y += line_h
        y += line_h // 2

    # Footer
    draw.rectangle([0, H_idx - FOOTER_H, W_idx, H_idx], fill=(235, 240, 248))
    draw.line([(0, H_idx - FOOTER_H), (W_idx, H_idx - FOOTER_H)], fill=BORDER, width=1)
    draw.text((MARGIN, H_idx - FOOTER_H + (FOOTER_H-13)//2),
              f"Total panels: {len(all_saved)}  ·  Print at 150 DPI  ·  Universitas Mataram — Teknik Informatika",
              font=FONT_CAP, fill=CAP_FG)

    path = os.path.join(OUT_DIR, "00_panel_index.png")
    canvas.save(path, dpi=(PRINT_DPI, PRINT_DPI))
    kb = os.path.getsize(path) // 1024
    print(f"\n  ✓  00_panel_index.png   {canvas.size[0]}×{canvas.size[1]}px  {kb} KB")
    return path


# ════════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("  IoT-SLM Dashboard Splitter  ·  Light Print Theme")

    saved_eval = split_eval_dashboard()
    saved_xai  = split_xai_dashboard()
    all_saved  = saved_eval + saved_xai

    index = make_index_sheet(all_saved)

    print(f"  Done.  {len(all_saved)} panels saved to:")
    print(f"  {OUT_DIR}/")
