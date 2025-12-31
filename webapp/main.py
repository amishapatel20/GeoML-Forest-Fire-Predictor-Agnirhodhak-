import os
import glob
from datetime import datetime

import numpy as np
import rasterio
import matplotlib.pyplot as plt
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from tensorflow import keras


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(PROJECT_ROOT)

MODEL_DIR = os.path.join(ROOT_DIR, "models")
STACK_DIR = os.path.join(ROOT_DIR, "daily_stacks")
FORECAST_DIR = os.path.join(ROOT_DIR, "future_forecasts")

os.makedirs(FORECAST_DIR, exist_ok=True)

B_TEMP, B_U, B_V, B_WS, B_NDVI, B_BURN = 0, 1, 2, 3, 4, 5
B_LULC, B_DEM, B_SLOPE, B_ASPECT, B_HILL = 6, 7, 8, 9, 10

UNET_FEATURE_IDXS = [
    B_TEMP,
    B_U,
    B_V,
    B_WS,
    B_NDVI,
    B_LULC,
    B_DEM,
    B_SLOPE,
    B_ASPECT,
    B_HILL,
]

PATCH_SIZE = 128

UNET_MODEL_PATH = os.path.join(MODEL_DIR, "unet_scenario1_tplus1.keras")
UNET_META_PATH = os.path.join(MODEL_DIR, "unet_scenario1_tplus1_meta.json")


_unet_model = None
_best_thr_u = 0.5


def load_unet_model():
    global _unet_model, _best_thr_u
    if _unet_model is not None:
        return _unet_model, _best_thr_u

    if not os.path.exists(UNET_MODEL_PATH) or not os.path.exists(UNET_META_PATH):
        raise RuntimeError(
            "Saved U-Net Scenario 1 model or meta JSON not found. "
            "Run the save-model cell in Model.ipynb first to create them."
        )

    _unet_model = keras.models.load_model(UNET_MODEL_PATH, compile=False)

    import json

    with open(UNET_META_PATH, "r") as f:
        meta = json.load(f)
    _best_thr_u = float(meta.get("best_thr_u", 0.5))

    return _unet_model, _best_thr_u


def _parse_date_from_stack_name(path: str) -> str:
    name = os.path.basename(path)
    # Expected: Uttarakhand_stack_YYYY-MM-DD.tif
    try:
        date_str = name.split("_")[-1].replace(".tif", "")
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except Exception:
        return name


def get_stack_path_for_date(date_str: str | None) -> tuple[str, str]:
    """Return (path, resolved_date_str).

    - If date_str is None or "latest", picks the latest available stack file.
    - Otherwise, tries to use Uttarakhand_stack_<date_str>.tif.
    """

    pattern = os.path.join(STACK_DIR, "Uttarakhand_stack_*.tif")
    all_files = sorted(glob.glob(pattern))
    if not all_files:
        raise RuntimeError(
            f"No daily stacks found in {STACK_DIR}. "
            "Use the notebook or script to export daily stacks first."
        )

    if date_str is None or date_str.lower() == "latest":
        path = all_files[-1]
        resolved_date = _parse_date_from_stack_name(path)
        return path, resolved_date

    # Specific date requested
    target = os.path.join(STACK_DIR, f"Uttarakhand_stack_{date_str}.tif")
    if not os.path.exists(target):
        raise FileNotFoundError(
            f"Stack for date {date_str} not found at {target}. "
            "Ensure this date has been exported."
        )
    return target, date_str


def run_unet_inference_on_stack(stack_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run U-Net Scenario 1 on a single-day stack (B, H, W).

    Returns (prob_full, risk_mask) with shapes (H, W) and (H, W).
    """

    model, best_thr_u = load_unet_model()

    if stack_arr.ndim != 3:
        raise ValueError(f"Expected (B,H,W) array, got shape {stack_arr.shape}")

    feat_chw = stack_arr[UNET_FEATURE_IDXS]  
    C, H, W = feat_chw.shape

    prob_full = np.zeros((H, W), dtype="float32")

    for y0 in range(0, H, PATCH_SIZE):
        for x0 in range(0, W, PATCH_SIZE):
            y1 = min(y0 + PATCH_SIZE, H)
            x1 = min(x0 + PATCH_SIZE, W)

            sub = feat_chw[:, y0:y1, x0:x1]
            h_sub = y1 - y0
            w_sub = x1 - x0

            pad_h = PATCH_SIZE - h_sub
            pad_w = PATCH_SIZE - w_sub

            if pad_h > 0 or pad_w > 0:
                sub = np.pad(
                    sub,
                    ((0, 0), (0, pad_h), (0, pad_w)),
                    mode="constant",
                    constant_values=0.0,
                )

            x_patch = np.transpose(sub, (1, 2, 0))[None, ...].astype("float32")

            y_pred = model.predict(x_patch, batch_size=1, verbose=0)[0, ..., 0]

            prob_full[y0:y1, x0:x1] = y_pred[:h_sub, :w_sub]

    prob_full = np.nan_to_num(prob_full, nan=0.0, posinf=1.0, neginf=0.0)
    prob_full = np.clip(prob_full, 0.0, 1.0)
    risk_mask = (prob_full >= best_thr_u).astype("uint8")

    return prob_full, risk_mask



def save_visualizations(
    stack_arr: np.ndarray,
    prob_full: np.ndarray,
    risk_mask: np.ndarray,
    date_str: str,
) -> tuple[str, str]:
    """Save probability map and NDVI+mask overlay PNGs.

    Returns (prob_png_path, overlay_png_path).
    """

    H, W = prob_full.shape
    # Probability map
    plt.figure(figsize=(6, 5))
    im = plt.imshow(prob_full, cmap="inferno", vmin=0.0, vmax=1.0)
    plt.title(f"U-Net Scenario 1 risk (t+1) for {date_str}")
    plt.colorbar(im, fraction=0.046, pad=0.02, label="Probability")
    plt.axis("off")
    prob_png = os.path.join(FORECAST_DIR, f"unet_scenario1_prob_tplus1_{date_str}.png")
    plt.tight_layout()
    plt.savefig(prob_png, dpi=150)
    plt.close()

    # NDVI + high-risk overlay (if NDVI band available)
    overlay_png = ""
    try:
        base_ndvi = stack_arr[B_NDVI]
        vmin_base = np.nanpercentile(base_ndvi, 5)
        vmax_base = np.nanpercentile(base_ndvi, 95)
    except Exception:
        base_ndvi = None

    if base_ndvi is not None:
        plt.figure(figsize=(6, 5))
        plt.imshow(base_ndvi, cmap="Greens", vmin=vmin_base, vmax=vmax_base)
        plt.imshow(
            np.ma.masked_where(risk_mask == 0, risk_mask),
            cmap="autumn",
            alpha=0.6,
        )
        plt.title(
            f"NDVI + predicted high-risk mask (t+1) for {date_str}"
        )
        plt.axis("off")
        overlay_png = os.path.join(
            FORECAST_DIR,
            f"unet_scenario1_overlay_tplus1_{date_str}.png",
        )
        plt.tight_layout()
        plt.savefig(overlay_png, dpi=150)
        plt.close()

    return prob_png, overlay_png


app = FastAPI(title="Agnirodhak Uttarakhand Fire Risk Forecast")

# Allow cross-origin requests so a separate frontend (e.g., hosted on
# another domain) can call this API safely in production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static assets for frontend and forecast images
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(PROJECT_ROOT, "static")),
    name="static",
)
app.mount(
    "/forecasts",
    StaticFiles(directory=FORECAST_DIR),
    name="forecasts",
)


@app.get("/api/available-dates")
async def api_available_dates() -> JSONResponse:
    """List all dates for which daily stacks exist.

    This is useful for frontends to know what dates can be
    queried for predictions before calling /api/predict.
    """

    pattern = os.path.join(STACK_DIR, "Uttarakhand_stack_*.tif")
    all_files = sorted(glob.glob(pattern))
    if not all_files:
        raise HTTPException(
            status_code=500,
            detail=(
                f"No daily stacks found in {STACK_DIR}. "
                "Use the notebook or script to export daily stacks first."
            ),
        )

    dates = [_parse_date_from_stack_name(p) for p in all_files]
    return JSONResponse({"dates": dates, "count": len(dates)})


@app.get("/")
async def index() -> FileResponse:
    """Serve the main web UI."""

    index_path = os.path.join(PROJECT_ROOT, "static", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=500, detail="index.html not found.")
    return FileResponse(index_path, media_type="text/html")


@app.get("/about")
async def about_page() -> FileResponse:
    """Serve the About page."""

    about_path = os.path.join(PROJECT_ROOT, "static", "about.html")
    if not os.path.exists(about_path):
        raise HTTPException(status_code=500, detail="about.html not found.")
    return FileResponse(about_path, media_type="text/html")


@app.get("/methodology")
async def methodology_page() -> FileResponse:
    """Serve the Methodology page."""

    method_path = os.path.join(PROJECT_ROOT, "static", "methodology.html")
    if not os.path.exists(method_path):
        raise HTTPException(
            status_code=500, detail="methodology.html not found."
        )
    return FileResponse(method_path, media_type="text/html")


@app.get("/api/predict")
async def api_predict(date: str | None = Query(default="latest")) -> JSONResponse:
    """Run fire-risk prediction for a given date or latest stack.

    - date: "YYYY-MM-DD" or "latest" (default).
    Returns JSON with URLs to PNG visualizations.
    """

    try:
        stack_path, resolved_date = get_stack_path_for_date(date)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    try:
        with rasterio.open(stack_path) as src:
            stack_arr = src.read().astype("float32")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to read stack file {stack_path}: {e}",
        ) from e

    try:
        prob_full, risk_mask = run_unet_inference_on_stack(stack_arr)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {e}",
        ) from e

    prob_png, overlay_png = save_visualizations(
        stack_arr=stack_arr,
        prob_full=prob_full,
        risk_mask=risk_mask,
        date_str=resolved_date,
    )

    # Summaries to help non-technical users interpret the map
    total_pixels = int(risk_mask.size)
    if total_pixels:
        # Use model probability field to derive three bands
        # High: prob >= best_thr_u (already used for risk_mask)
        # Moderate: 0.3 <= prob < best_thr_u
        # Low: prob < 0.3
        # These thresholds are heuristic but give an intuitive split
        _, best_thr_u = load_unet_model()
        high_mask = prob_full >= best_thr_u
        moderate_mask = (prob_full >= 0.3) & (prob_full < best_thr_u)
        low_mask = prob_full < 0.3

        high_pixels = int(high_mask.sum())
        moderate_pixels = int(moderate_mask.sum())
        low_pixels = int(low_mask.sum())

        # Normalise to fractions that sum to ~1.0
        denom = float(high_pixels + moderate_pixels + low_pixels) or float(total_pixels)
        high_fraction = float(high_pixels / denom)
        moderate_fraction = float(moderate_pixels / denom)
        low_fraction = float(low_pixels / denom)
    else:
        high_pixels = 0
        high_fraction = 0.0
        moderate_fraction = 0.0
        low_fraction = 0.0

    # Approximate focus area label with Uttarakhand-style region names.
    # This is heuristic (based on the grid), not an exact district lookup.
    focus_label = "No high-risk pixels detected"
    ys, xs = np.where(risk_mask == 1)
    if ys.size > 0:
        cy = float(ys.mean())
        cx = float(xs.mean())
        H, W = risk_mask.shape

        # Map center-of-mass into 3x3 bands (row_band, col_band)
        row_band = 1
        col_band = 1
        if cy < H / 3:
            row_band = 0
        elif cy > 2 * H / 3:
            row_band = 2
        if cx < W / 3:
            col_band = 0
        elif cx > 2 * W / 3:
            col_band = 2

        region_names = {
            # row 0 = northern high ranges
            (0, 0): "north‑west high ranges (around Uttarkashi belt)",
            (0, 1): "upper Garhwal (Chamoli–Uttarkashi belt)",
            (0, 2): "north‑east high ranges (towards Pithoragarh)",
            # row 1 = central hill districts
            (1, 0): "western Garhwal (Dehradun–Tehri side)",
            (1, 1): "central Garhwal (around Tehri–Pauri belt)",
            (1, 2): "eastern Kumaon (Almora–Pithoragarh belt)",
            # row 2 = southern Shivalik / Terai
            (2, 0): "south‑west Shivalik foothills", 
            (2, 1): "central Shivalik belt", 
            (2, 2): "south‑east Terai forests",
        }

        focus_label = region_names.get((row_band, col_band), "central Uttarakhand")

    prob_url = "/forecasts/" + os.path.basename(prob_png)
    overlay_url = "/forecasts/" + os.path.basename(overlay_png) if overlay_png else ""

    return JSONResponse(
        {
            "date": resolved_date,
            "stack_path": stack_path,
            "probability_png": prob_url,
            "overlay_png": overlay_url,
            "high_risk_fraction": high_fraction,
            "moderate_risk_fraction": moderate_fraction,
            "low_risk_fraction": low_fraction,
            "high_risk_focus": focus_label,
        }
    )


if __name__ == "__main__":
    import uvicorn

    # On platforms like Render, the port is provided via the PORT
    # environment variable. Fall back to 8000 for local runs.
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
