# Agnirodhak – Forest Fire Early Warning for Uttarakhand

Agnirodhak is a research project that predicts short-term (next 1–3 days)
wildfire spread / burn probability over a Himalayan region (Uttarakhand)
using daily Earth Engine datasets, stacked GeoTIFFs, and machine learning.
It combines a full modeling pipeline in a Jupyter notebook with a small
FastAPI web dashboard so the results can be explored interactively.

## 🎯Highlights
- End-to-end ML pipeline: data ingestion from Google Earth Engine →
  preprocessing → feature engineering → model training → evaluation.
- Spatial & temporal modeling: Random Forest baseline, multiple U-Net
  segmentation models, and a ConvLSTM sequence model for fire spread.
- Class-imbalance handling, custom metrics (G-mean, CBA, ROC/PR AUC) and
  threshold tuning for operational decision support.
- Remote sensing experience: ERA5 reanalysis, MODIS NDVI / burned area,
  ESA WorldCover land cover, SRTM terrain derivatives.
- Deployment: FastAPI backend + HTML/CSS/JS frontend for an interactive
  “Agnirodhak” fire-risk dashboard.

## 🚀 Core Technical Capabilities
- Automated daily data export (stacked + per-band) from Google Earth Engine.
- Consistent multi-band daily stacks: TempC, U10, V10, WindSpeed, NDVI,
  BurnDate, LULC, DEM, Slope, Aspect, Hillshade.
- Leakage-safe feature engineering (uses historical + current state only).
- Lag-window temporal features with optional day-to-day differences.
- Dual targets:
  - ANY_BURN: any pixel that will be (or continue to be) burned the next day.
  - IGNITION: new burned pixels (unburned today → burned tomorrow).
- Class imbalance mitigation (negative downsampling).
- Automatic decision-threshold selection from validation curves.
- Next-day risk map visualization and GIF animations.

## 🧠 Models Implemented
- **Tabular baselines** (ANY_BURN / optional IGNITION):
  - HistGradientBoostingClassifier (main baseline in the notebook).
  - RandomForestClassifier (alternative tree ensemble).
- **U-Net segmentation (Keras / TensorFlow)**:
  - Scenario 1: t → t+1 next-day burn probability (operational model used
    in the web app and inference scripts).
  - Scenario 2: any burn in the next 2 days (t+1..t+2).
  - Scenario 3: any burn in the next 5 days (t+1..t+5).
- **ConvLSTM**:
  - Sequence model over 5-day stacks capturing temporal evolution of
    environmental features.

The interactive dashboard and standalone inference code use the
U-Net Scenario 1 model together with the best decision threshold
learned on the test set.

## 📊 Model Comparison (test set)

| Model                          | G-mean  | CBA      | Precision | Accuracy |
|--------------------------------|---------|----------|-----------|----------|
| U-Net Scenario 1 (t+1)         | 0.9391  | 0.7166   | 0.5015    | 0.9327   |
| U-Net Scenario 3 (t+1..t+5)    | 0.9287  | 0.6797   | 0.4418    | 0.9191   |
| U-Net Scenario 2 (t+1..t+2)    | 0.9178  | 0.6602   | 0.4204    | 0.9027   |
| RF ANY_BURN                    | 0.8824  | 0.7325   | 0.6104    | 0.8659   |
| ConvLSTM (5-day seq, t+1)      | 0.8350  | 0.5326   | 0.2448    | 0.8223   |

## 🧱 Repository Structure
```
Agnirodhak_D/
  Model.ipynb                  # Main RF + U-Net + ConvLSTM workflow
  ForestFire_Detector_Model.py # Scripted RF / GIF utilities
  webapp/                      # FastAPI backend + static web UI
    main.py
    static/
      index.html               # Dashboard
      about.html               # Project overview
      methodology.html         # Methods & data
  daily_stacks/                # (Created) Stacked daily GeoTIFFs (pauri_stack_YYYY-MM-DD.tif)
  daily_bands/                 # (Optional) Per-day subdirectories with single-band GeoTIFFs
  figures/                     # (Optional) Saved static figures
  *.gif / *.mp4                # (Generated) Animations
  requirements.txt             # Python deps (install before running)
  README.md                    # This document
```

## 🌐 Web Application (Agnirodhak Dashboard)

This repository includes a minimal web application so you can publish
an interactive dashboard for the U-Net Scenario 1 next-day fire risk
model.

Backend:
- FastAPI app in `webapp/main.py`.
- Serves a JSON prediction API (`/api/predict`).
- Serves a small static site from `webapp/static`.

Frontend:
- Single-page dashboard (`index.html`) with date picker and map.
- Additional pages:
  - `/about` – high-level project description.
  - `/methodology` – model + data documentation.

### 🔌 Run Locally (Windows)

From the project root (Agnirodhak_D/):

```powershell
python -m venv venv
./venv/Scripts/Activate.ps1   # activate virtual environment
pip install -r requirements.txt

# Start the web API + static frontend
python .\webapp\main.py
```

Then open:
- Dashboard: http://127.0.0.1:8000/
- About: http://127.0.0.1:8000/about
- Methodology: http://127.0.0.1:8000/methodology

> The API expects daily stacks under `daily_stacks/` named
> `Uttarakhand_stack_YYYY-MM-DD.tif`. Use Model.ipynb to export
> additional dates before serving predictions for them.

### 🚀 Publishing Ideas

- **Single-container deployment**: package this project in a Docker
  image and run the FastAPI app behind a production ASGI server
  (e.g., uvicorn or gunicorn) on any cloud VM or PaaS (Render,
  Railway, Fly.io, etc.).
- **Separate frontend**: thanks to enabled CORS, you can build a
  richer React / Next.js site that calls this API (e.g., hosted on
  Vercel) while keeping the FastAPI app as a backend service.
- **Scheduled updates**: run a daily job (notebook or script) to
  export new stacks into `daily_stacks/` so the web app always has
  recent data.

## 🗂 Data Sources (Google Earth Engine Collections)
| Variable | Source Collection | Notes |
|----------|-------------------|-------|
| TempC, U10, V10 | `ECMWF/ERA5/DAILY` | Temperature converted to °C; winds at 10 m; WindSpeed derived. |
| NDVI | `MODIS/061/MOD13Q1` | 16-day composite; scaled by 0.0001. |
| BurnDate | `MODIS/061/MCD64A1` | Burn date mosaic (recent ~32 days). Converted to mask (burn >0). |
| LULC | `ESA/WorldCover/v100` | Land cover category. |
| DEM | `USGS/SRTMGL1_003` | Elevation (m); used to derive slope, aspect, hillshade. |

> You must authenticate with Earth Engine (Cell 1) first run.

## 🔧 Environment Setup
PowerShell (Windows):
```powershell
python -m venv venv
./venv/Scripts/Activate.ps1
pip install -r requirements.txt
```
If `geemap` requests additional auth steps, follow on-screen browser login.

## 🔁 Pipeline Overview (Notebook Cells)
| Cell | Purpose |
|------|---------|
| 1 | EE init, AOI, date range, export toggles. |
| 2 | Daily export loop (stacked + optional per-band). Idempotent. |
| 3 | Load stacked files or reconstruct from per-band directories. |
| 3b | (Optional) Write reconstructed stacked GeoTIFFs locally. |
| 3c | Quick exploratory visualization of key bands (first & last day). |
| 4 | Spatio-temporal feature + target construction (lags, diffs, labels, split). |
| 5 | Model training (ANY_BURN + optional IGNITION), metrics, threshold selection. |
| 6 | Single-horizon rolling next-day prediction GIF (forward risk evolution). |
| 6b | Multi-horizon (1–3 day) forecast per frame with side-by-side horizons (GIF/MP4). |
| 7 | Original simple GIF modes (actual burn / risk_any / risk_ign). |

## 🧪 Features & Targets
This section provides detailed definitions, preprocessing steps, and rationale for every feature / engineered input.

### Base Dynamic Environmental Features (Daily)
| Name | Bands / Symbol | Source | Unit | Preprocessing | Rationale |
|------|----------------|--------|------|---------------|-----------|
| Air Temperature | TempC (T) | ERA5 DAILY mean_2m_air_temperature | °C | Kelvin → °C by subtracting 273.15; daily mean | Warmer air can dry fuels & increase combustion / spread potential. |
| Zonal Wind | U10 (U) | ERA5 DAILY u_component_of_wind_10m | m/s | Daily mean | Wind drives directional flame spread; included for vector resolution. |
| Meridional Wind | V10 (V) | ERA5 DAILY v_component_of_wind_10m | m/s | Daily mean | See above; combined with U for speed + direction. |
| Wind Speed (derived) | WindSpeed (W) | sqrt(U^2+V^2) | m/s | Computed per day | Magnitude of advective forcing on fire fronts. |
| Vegetation Index | NDVI (N) | MODIS MOD13Q1 NDVI | unitless (-1..1) | 16-day composite mean over window [t-16, t]; scaled by 0.0001 | Proxy for live green biomass; influences fuel availability & moisture. |
| Burn Date | BurnDate (BD) | MODIS MCD64A1 | days since start of year (int) | Mosaic last 32 days; used only to derive masks, not as raw numeric feature | Identifies already burned pixels to exclude from ignition risk. |

### Static Physiographic / Fuel Context
| Name | Bands | Source | Unit | Notes / Derivation | Rationale |
|------|-------|--------|------|--------------------|-----------|
| Land Use / Land Cover | LULC | ESA WorldCover v100 | categorical | First image clipped; no remap applied | Different fuel beds & ignition likelihood. |
| Elevation | DEM | SRTMGL1_003 | m | Raw DEM clipped | Influences microclimate & slope. |
| Slope | Slope | Terrain.slope(DEM) | degrees | Derived from DEM | Fire spreads faster upslope (preheats fuel). |
| Aspect | Aspect | Terrain.aspect(DEM) | degrees (0–360) | Derived; left raw | Orientation affects insolation / drying. |
| Hillshade | Hillshade | Terrain.hillshade(DEM) | intensity (0–255) | Derived; relative illumination | Correlates with microclimate / dryness proxies. |

### Binary Burn State Feature
| Name | Symbol | Definition | Encoding | Purpose |
|------|--------|------------|----------|---------|
| Current Burn Mask | burn_now (B₀) | 1 if BurnDate > 0 on day t else 0 | float32 | Prevents model predicting ignition on already burned pixels; informs spread containment. |

### Temporal Engineering
Configuration: `LAG_DAYS = L` (e.g., 3). For each prediction reference day t (predicting t+1):
1. Collect dynamic slices for time indices (t-L+1 .. t) → shape (L, 5, H, W).
2. Flatten across temporal dimension: yields L*5 channels.
3. (Optional) Differences: For k=1..L-1 compute Δ_k = slice_k - slice_{k-1} per dynamic band → (L-1)*5 channels capturing temporal trend / acceleration (e.g., warming rate, wind shift, NDVI change).
4. Append `burn_now` (1 channel) and static context (5 channels).

Final channel count F:
F = (L * 5) + ( (L-1)*5 if INCLUDE_DIFFS else 0 ) + 1 (burn_now) + 5 (static)

Example (L=3, diffs on): F = 3*5 + 2*5 + 1 + 5 = 31.

### Spatial Label Dilation (Optional)
If `DILATE_LABELS=True`, ANY_BURN labels undergo a morphological dilation with 3x3 structuring element (radius ~1). This:
* Provides a margin around burns to capture near-front areas.
* Encourages conservative over-prediction beneficial for early warning use-cases.
Trade-off: Inflates positive count; may slightly reduce precision while increasing recall.

### Formal Target Definitions
Let B_t(x) ∈ {0,1} be burn mask at day t for pixel x (1 if burned). The model predicts probability P(B_{t+1}(x)=1 | history up to t).

1. ANY_BURN (AB):  AB_{t+1}(x) = 1 iff B_{t+1}(x) = 1. (Optionally dilated post hoc.)
2. NEW_IGNITION (IGN): IGN_{t+1}(x) = 1 iff B_t(x)=0 and B_{t+1}(x)=1.

Properties:
* IGN ⊂ AB always.
* IGN positives are usually rarer -> may skip training if < `MIN_TRAIN_POS_IGNITION`.

### Leakage Avoidance Principles
| Potential Leakage Source | Mitigation |
|--------------------------|-----------|
| Direct numeric BurnDate future value | Not used; only binary burn_now from current day t. |
| Using day t+1 dynamic variables while predicting t+1 | Strictly assemble features using indices ≤ t. |
| Using dilated future burn in features | Dilation applied only to labels after feature computation. |
| Temporal look-ahead in differences | Differences computed only within window ending at t. |

### Negative Downsampling Strategy
For training set: retain all positives P; sample up to `MAX_NEG_PER_POS * |P|` negatives uniformly at random. Benefits:
* Controls memory & class imbalance.
* Reduces training time.
Edge Cases: If |P|=0 ⇒ cap to first min(50,000, total negatives) to allow metrics skip gracefully.

### Threshold Selection (Decision Layer)
Given predicted probabilities p_i and true labels y_i on test set, compute precision-recall curve. For threshold θ_k (k over unique prob splits):
F1_k = 2 * (Prec_k * Rec_k) / (Prec_k + Rec_k + ε). Choose θ maximizing F1. This yields balanced compromise under imbalance.

If `THRESH_OVERRIDE` supplied (Cells 6 / 6b), it supersedes learned θ for visualization (useful for scenario tuning, e.g., favor high recall).

### Multi-Horizon Approximation (Cell 6b)
Model was trained on next-day mapping (t→t+1). For horizon h>1 predictions, we reuse actual historical context at in_day = t+h-1 so that distribution shift is minimized (no recursive synthetic burns). Resulting frames thus represent *retrospective forecasts* given perfect knowledge up to that day—suitable for post-analysis & operational tuning, but not a pure forward simulation. A true simulation would instead iteratively update burn_now with predicted mask (future enhancement).

### Potential Additional Features (Not Yet Implemented)
| Candidate | Source | Value |
|----------|--------|-------|
| Relative Humidity | ERA5 | Moisture effect on ignition probability |
| Precipitation (24h) | ERA5 | Dampening / suppression of fire spread |
| Soil Moisture | ERA5-Land / SMAP | Fuel moisture proxy |
| Lightning Density | LIS/OTD | Natural ignition driver |
| Human Proximity | OSM roads / settlements | Anthropogenic ignition risk |
| Fuel Moisture Codes | NFDRS / Derived indices | Fire behavior potential |

Adding these would follow same pattern: integrate into daily stack (dynamic vs static), incorporate into lag & diff logic, retrain & reassess metrics.

## 🧮 Modeling
- Default classifier: `HistGradientBoostingClassifier` (fast, handles large tabular data).
- Alternative: RandomForest (set FAST_MODEL = 'RF').
- Imbalance handling: downsample negatives to ratio ≤ MAX_NEG_PER_POS.
- Threshold: choose by maximizing F1 from precision-recall curve (ANY_BURN & IGNITION separately).

## Models included 
- HistGradientBoostingClassifier (tabular baseline for ANY_BURN / IGNITION)
- RandomForestClassifier (tabular alternative)
- U-Net convolutional segmentation models (Scenarios 1, 2, 3)
- ConvLSTM sequence model (5-day temporal stacks)

Why these models:
* Tree-based models provide strong, interpretable tabular baselines for pixel-level
  risk using lagged environmental features.
* U-Net captures spatial patterns of burn spread and performs best for next-day
  risk mapping (Scenario 1 is used in the web app).
* ConvLSTM explicitly models temporal evolution across multiple days of inputs.

Training pipeline steps:
1. Assemble pixel–transition samples: features from day t (lag stack) -> label from day t+1.
2. Downsample negatives (retain all positives P; sample up to `MAX_NEG_PER_POS * |P|` negatives).
3. Fit HGB on downsampled training set; keep untouched full test set for unbiased metrics.
4. Generate probabilities for test; derive precision–recall curve; pick threshold maximizing F1 (stored as `THR_ANY` / `THR_IGN`).

Key Hyperparameters (`HGB_PARAMS`):
| Param | Value | Effect |
|-------|-------|--------|
| max_depth | 6 | Limits tree depth; controls interaction order & overfitting risk. |
| learning_rate | 0.15 | Step size per boosting iteration; higher → fewer trees needed; too high risks overshoot. |
| max_iter | 200 | Number of boosting stages (trees). |
| l2_regularization | 0.0 | Add >0 to smooth models if overfitting observed. |
| random_state | 42 | Reproducibility. |

Internal mechanics:
* Gradient boosting minimizes logistic loss (binomial deviance) by sequentially fitting trees to residual gradients.
* Feature values bucketed into discrete bins → faster histogram-based gain computation.
* Final log-odds = sum(tree_outputs); probability = sigmoid(log-odds).

### Alternative: Random Forest (RF)
Activated by setting `FAST_MODEL='RF'`.
Pros: Simpler tuning, robust to noise, parallelizable.
Cons: Typically less sharp probability estimates, may need more trees for comparable AUC, no stage-wise refinement.

### Probability Thresholding
Raw probabilities rarely align with operational goals. The notebook computes F1 across thresholds and selects the max (balanced trade‑off). Override with `THRESH_OVERRIDE` in prediction cells to bias toward:
* Higher recall (early warning) → lower threshold.
* Higher precision (resource focus) → higher threshold.

### Class Imbalance Strategy
Downsampling reduces training set size & skew; evaluation still on full distribution → reported precision/recall reflect real imbalance.
Edge case: zero positives in test → metrics skipped gracefully.

### Avoiding Data Leakage
| Risk | Guard |
|------|-------|
| Using future BurnDate numerics | Only current day burn mask `burn_now` included. |
| Including t+1 dynamic data in features | Features built strictly from indices ≤ t. |
| Label dilation leaking into features | Dilation applied post feature assembly. |

### Limitations of Current Modeling Layer
* Pixel independence: No explicit spatial interaction; spread learned implicitly via static + burn_now patterns.
* Next-day only training; multi-horizon (Cell 6b) is retrospective application of the same next-day model.
* Circular variables (Aspect) not encoded as sin/cos; model must infer wrap-around.
* LULC categorical not one-hot / embedded (HGB treats it as ordered numeric). CatBoost/LightGBM could help.

### Potential Improvements
| Area | Enhancement | Benefit |
|------|-------------|---------|
| Calibration | Isotonic / Platt scaling | More reliable probabilities for decision thresholds. |
| Spatial context | Add neighborhood stats / convolutional features | Capture fire-front geometry. |
| Feature encoding | One-hot / target encode LULC; sin/cos Aspect | Better categorical & cyclical handling. |
| Hyperparameter tuning | Grid / Bayesian search over depth, LR, iterations | Potential lift in PR AUC. |
| Advanced libraries | LightGBM / XGBoost / CatBoost | Faster training, native categorical support. |
| Frontier features | Distance to nearest active burn, perimeter orientation | Direct spread dynamics encoding. |
| Simulation | Autoregressive predicted burn updates | True forward scenario generation (vs retrospective). |

### Interpretability Toolkit (Suggested Additions)
| Method | Library | Usage |
|--------|---------|-------|
| Permutation Importance | sklearn.inspection | Global feature effect ranking. |
| Partial Dependence / ICE | sklearn.inspection | Marginal effect of single / pair features. |
| SHAP values | shap | Local pixel-level contribution explanations. |
| Spatial Error Maps | custom | Highlight systematic under/over-prediction zones. |

### Example: Adding Permutation Importance (ANY_BURN)
```python
from sklearn.inspection import permutation_importance
subset_idx = np.random.choice(X_test.shape[0], size=min(50000, X_test.shape[0]), replace=False)
perm = permutation_importance(MODEL_ANY, X_test[subset_idx], y_any_test[subset_idx], n_repeats=5, random_state=42)
import matplotlib.pyplot as plt
fi_order = np.argsort(perm.importances_mean)[::-1][:20]
plt.figure(figsize=(6,4))
plt.bar(range(len(fi_order)), perm.importances_mean[fi_order], yerr=perm.importances_std[fi_order])
plt.title('Permutation Importance (top 20)')
plt.xlabel('Feature index')
plt.tight_layout(); plt.show()
```

### When to Switch Models
Consider moving to spatial deep learning (ConvLSTM / U-Net temporal stacks) when:
* You have higher resolution inputs (e.g., VIIRS active fire, finer DEM derivatives).
* Need explicit modeling of contiguous spread fronts.
* Training dataset spans multiple seasons / regions (improved generalization needed).

Until then, the current HGB pipeline offers a strong, transparent baseline.

## 📊 Metrics Reported
- ROC AUC
- PR AUC (Average Precision)
- Best F1 + threshold + precision & recall at that F1 point.

## 🗓 Multi-Horizon Forecasting (Cell 6b)
For each reference day t, produce predictions for horizons H in {1,2,3}:
- For horizon h: build lag stack ending at day (t + h − 1) and predict day (t + h).
- Panels show: base NDVI, risk heatmap, high-risk mask (threshold), optional actual burn overlay, optional wind vectors.
- Outputs: GIF (`multi_horizon_risk.gif`) and MP4 if ffmpeg encoder is available.

### Why This Approach?
The model is trained only for next-day prediction; multi-day horizons are approximated by applying it recursively at successive days (using actual historical daily context rather than simulated future). This yields consistent short-term forecast snapshots while avoiding compounded simulation error.

## 🎞 GIF / MP4 Outputs
| Output | Source Cell | Description |
|--------|-------------|-------------|
| `fire_spread_animation.gif` | 7 | Actual burn OR single-horizon risk sequence. |
| `predicted_risk_sequence.gif` | 6 | Rolling next-day forecast risk. |
| `multi_horizon_risk.gif` | 6b | Multi-panel horizons per frame. |
| `multi_horizon_risk.mp4` | 6b | Optional MP4 (if encoder). |

## ⚙️ Key Configuration Flags
Edit directly in notebook cells:
- Cell 1: `START_DATE`, `END_DATE`, `EXPORT_DAILY_STACKS`.
- Cell 2: `EXPORT_SEPARATE_BANDS`.
- Cell 4: `LAG_DAYS`, `INCLUDE_DIFFS`, `DILATE_LABELS`, `MIN_TRAIN_POS_IGNITION`.
- Cell 5: `FAST_MODEL`, `MAX_NEG_PER_POS`, model param dicts.
- Cell 6: `PRED_TARGET`, `INCLUDE_ACTUAL_NEXT`, `THRESH_OVERRIDE`, `INCLUDE_WIND`.
- Cell 6b: `HORIZONS`, `TARGET`, `INCLUDE_ACTUAL`, `INCLUDE_THRESHOLD`, `THRESH_OVERRIDE`.

## 🧭 Interpreting Results
- Cyan contours (Cell 6 / 6b): actual future burn mask — qualitative validation.
- Yellow contours: predicted high-risk region (≥ decision threshold).
- Warm colormap (inferno/hot): higher predicted probability → higher spread likelihood.
- Wind vectors: direction and relative magnitude of wind influence for that day.

## 🛡 Leakage Avoidance
We never feed the future BurnDate numeric values into features; only present/past states and static context. Targets refer strictly to next-day burn state.

## 🔄 Extending / Next Steps
| Idea | Description |
|------|-------------|
| Probability calibration | Apply isotonic or Platt scaling on validation split. |
| SHAP / feature attribution | Explain key drivers of local risk hotspots. |
| Frontier modeling | Define ignition as burn emerging after multi-day inactivity or at perimeter only. |
| Larger AOI / tiling | Chunk large rasters to manage memory. |
| Model persistence | Save trained models (`joblib.dump`) and metadata JSON. |
| Parameter sweep | Grid / Bayesian search for depth, learning rate, lag length. |
| Cloud masking | Incorporate quality flags for NDVI to reduce noise. |
| Simulation | Autoregressive multi-day simulation using predicted burn mask updates. |

## ⚠ Limitations
Additional nuance:
* Static aspect & hillshade not cyclically encoded; directional wrap-around (359≈0) not modeled (future: sin/cos transform).
* LULC categorical currently used raw; could embed or one-hot for algorithms needing distance-aware metrics.
* Dilation parameter (radius=1) fixed; adaptive dilation by wind direction & speed could yield better frontier emphasis.
- Multi-horizon predictions reuse actual historical sequences (not a forward simulation of hypothetical burns evolving under predicted risk).
- IGNITION class can be extremely sparse; model may skip training.
- Coarse spatial resolution (ERA5, MODIS) may miss micro-scale drivers.
- No explicit fuel moisture, humidity, precipitation, or human ignition factors yet.
- Threshold optimized for F1; operational deployments might prefer high recall or precision depending on objective.

## 🧪 Quality / Validation Tips
- Inspect class balance printouts in Cell 4 and ensure test set contains positives.
- Compare predicted high-risk contour overlap with actual burn (visual, or add a confusion matrix script).
- Adjust `LAG_DAYS` and re-train; monitor PR AUC changes.

## 📝 Troubleshooting
| Issue | Cause | Fix |
|-------|-------|-----|
| No stacked files found | Exports not run | Set `EXPORT_DAILY_STACKS=True` and run Cells 1–2 once. |
| Ignition model skipped | Too few positives | Expand date range or AOI; lower `MIN_TRAIN_POS_IGNITION`. |
| GIF empty / 1 frame | Not enough days for lag/horizons | Reduce `LAG_DAYS` or collect more days. |
| MP4 not created | Missing encoder | Install `imageio-ffmpeg` (`pip install imageio-ffmpeg`). |
| Memory error | Large AOI + many days | Reduce date span or implement tile-wise processing. |

## 🔐 Earth Engine Notes
Usage is subject to Google Earth Engine terms. Heavy export loops should include throttling / task monitoring for large areas (this notebook uses client-side synchronous exports via `geemap.ee_export_image` for simplicity over limited time windows and moderate area size).

## 📦 Dependencies (summary)
See `requirements.txt` — typically includes: `earthengine-api`, `geemap`, `rasterio`, `numpy`, `scipy`, `scikit-learn`, `matplotlib`, `imageio`.


**Summary:** This project demonstrates an end-to-end, lag-aware, leakage-safe wildfire spread risk modeling workflow leveraging daily remote sensing + reanalysis data, with interpretable multi-horizon visualization outputs for rapid situational awareness.
