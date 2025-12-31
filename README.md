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
  # Agnirodhak – Forest Fire Early Warning for Uttarakhand

  Agnirodhak is a forest‑fire early‑warning dashboard for Uttarakhand.
  It combines a U‑Net segmentation model (next‑day burn probability) with a
  FastAPI backend and a modern web UI so that scientists, administrators,
  and citizens can explore where fire risk is higher on a given day.

  The project is intended as a **full‑stack portfolio piece**: it
  demonstrates geospatial ML, clean API design, and a production‑style
  dashboard rather than a raw notebook demo.

  ---

  ## 🌟 What the App Delivers

  - **Next‑day risk map** for Uttarakhand based on daily stacks of
    temperature, wind, NDVI, topography, and land cover.
  - **Pixel‑level probability map** from a Keras/TensorFlow U‑Net
    (Scenario 1: t → t+1) with a tuned decision threshold.
  - **Three‑level risk breakdown** (High / Moderate / Low) computed
    from the predicted probabilities and exposed in the API.
  - **Focus region summary** (e.g., Garhwal / Kumaon / Terai‑like
    patterns) derived from the spatial distribution of high‑risk pixels.
  - **Precautions & response guidance** view with:
    - Risk banner (Normal / Caution / High alert) and a preparedness score.
    - Do / Don’t guidance if you live near forests.
    - Simple recommendations for what to do if you see smoke or fire.
  - **Emergency contact hints** for key Uttarakhand districts and a
    small **“check an area” helper** that ties the map focus and
    helpline info together in plain language.
  - **Assistant panel** that answers common questions about how to
    read the map and what the colours mean.

  ---

  ## 🏗️ Architecture

  - **Backend – FastAPI** (webapp/main.py)
    - Loads the saved U‑Net model and metadata (threshold).
    - Reads daily stacked GeoTIFFs from `daily_stacks/`.
    - Runs tiled inference on CPU/GPU and saves PNG outputs under
      `future_forecasts/`.
    - Exposes two primary endpoints:
      - `GET /api/available-dates` – list of dates for which stacks exist.
      - `GET /api/predict?date=YYYY-MM-DD|latest` – runs U‑Net,
        saves PNGs, and returns:
        - `probability_png` and `overlay_png` URLs.
        - `high_risk_fraction`, `moderate_risk_fraction`, `low_risk_fraction`.
        - `high_risk_focus` text summary.
    - Serves static assets (legacy minimal UI and PNGs) via `/static`
      and `/forecasts`.

  - **Frontend – React (Vite)** (frontend/)
    - Single‑page dashboard that calls the FastAPI API.
    - Tabs for **Dashboard**, **Model details**, **Precautions & response**, and **About**.
    - Modern NASA/ISRO‑style UI with:
      - Date selection and “Use latest” toggle.
      - Call‑to‑action: “Generate fire risk map”.
      - Map card with skeleton state, a clear loading state while the
        U‑Net runs, and a smooth fade‑in when the map is ready.
      - Legend and summary cards for High / Moderate / Low risk with
        percentages sourced from the backend fractions.
      - A small floating “Help” button that opens the assistant.

  - **Research notebook – Model training** (Model.ipynb)
    - Google Earth Engine → daily export of multi‑band stacks.
    - Baseline tabular models (HistGradientBoosting, Random Forest).
    - U‑Net training, evaluation, and export of weights + meta JSON.

  ---

  ## 📁 Repository Layout (high‑level)

  ```text
  Agnirodhak_D/
    Model.ipynb                  # Main ML & data pipeline notebook
    ForestFire_Detector_Model.py # Scripted RF / GIF utilities (legacy)
    webapp/                      # FastAPI backend + static assets
      main.py
      static/
        index.html               # Minimal dashboard (legacy)
        style.css, app.js        # Static UI (kept as a fallback)
    frontend/                    # React (Vite) SPA dashboard
      src/App.jsx                # Main UI + tabs + assistant
    daily_stacks/                # Stacked daily GeoTIFFs (Uttarakhand_stack_YYYY-MM-DD.tif)
    future_forecasts/            # Generated PNGs from API calls
    models/                      # Saved U‑Net model + meta JSON
    figures/                     # Optional static figures / GIFs
    requirements.txt             # Python dependencies
    README.md                    # This document
  ```

  ---

  ## 🔌 Running the Backend API (FastAPI)

  From the project root (Agnirodhak_D/) on Windows PowerShell:

  ```powershell
  python -m venv venv
  ./venv/Scripts/Activate.ps1
  pip install -r requirements.txt

  # Start the FastAPI app (serves API + legacy static pages)
  python .\webapp\main.py
  ```

  The default server runs on:

  - Backend base URL: http://127.0.0.1:8000
  - API endpoints:
    - http://127.0.0.1:8000/api/available-dates
    - http://127.0.0.1:8000/api/predict?date=latest

  The API expects stacks under `daily_stacks/` named
  `Uttarakhand_stack_YYYY-MM-DD.tif`. Use `Model.ipynb` to export
  additional dates before serving predictions for them.

  The minimal FastAPI‑served UI is still available at:

  - Dashboard (legacy): http://127.0.0.1:8000/

  ---

  ## 💻 Running the React Dashboard

  The React app lives under `frontend/` and talks to the FastAPI API via
  an environment variable `VITE_API_BASE`.

  1. Start the backend as described above (port 8000).
  2. In a new terminal:

  ```bash
  cd frontend
  npm install

  # Option 1: set API base inline for local dev
  VITE_API_BASE=http://127.0.0.1:8000 npm run dev

  # Option 2 (Windows PowerShell):
  $env:VITE_API_BASE = "http://127.0.0.1:8000"; npm run dev
  ```

  3. Open the Vite dev URL shown in the terminal (usually
     http://127.0.0.1:5173).

  When deployed (e.g. on Render), you can set `VITE_API_BASE` to the
  public URL of the FastAPI service so the frontend uses the hosted API.

  ---

  ## ☁️ Deployment Notes (example: Render)

  You can deploy the project as **two services** backed by the same
  GitHub repository:

  1. **Backend – FastAPI web service**
     - Root directory: project root.
     - Build command: `pip install -r requirements.txt`.
     - Start command: `uvicorn webapp.main:app --host 0.0.0.0 --port $PORT`.

  2. **Frontend – Static site (React)**
     - Root directory: `frontend`.
     - Build command: `npm install && npm run build`.
     - Publish directory: `dist`.
     - Environment variable: `VITE_API_BASE=<backend public URL>`.

  Any other PaaS / container platform works as long as it can run a
  FastAPI app (ASGI) and serve a static build of the Vite app.

  ---

  ## 📊 Modeling & Data (short overview)

  The heavy research / experimentation lives in `Model.ipynb`. At a
  high level:

  - **Data sources (Google Earth Engine)**
    - ERA5 Daily reanalysis: temperature and winds (10 m).
    - MODIS NDVI and burned area.
    - ESA WorldCover land cover.
    - SRTM DEM and terrain derivatives (slope, aspect, hillshade).

  - **Feature design**
    - Daily dynamic features: TempC, U10, V10, WindSpeed, NDVI.
    - Static context: LULC, DEM, Slope, Aspect, Hillshade.
    - Current burn mask to avoid predicting new ignition on already
      burned pixels.
    - Lagged windows and (optionally) day‑to‑day differences.

  - **Targets**
    - ANY_BURN: any pixel that will be (or remain) burned the next day.
    - NEW_IGNITION: pixels newly burned at t+1.

  - **Models**
    - HistGradientBoosting and Random Forest baselines on tabular
      features.
    - U‑Net convolutional model (Scenario 1, used in the web app).
    - A ConvLSTM sequence model for experiments with temporal stacks.

  - **Example U‑Net (Scenario 1) test metrics**

    | Model                       | G‑mean | CBA    | Precision | Accuracy |
    |----------------------------|--------|--------|-----------|----------|
    | U‑Net Scenario 1 (t → t+1) | 0.94   | 0.72   | 0.50      | 0.93     |

  The notebook also demonstrates class‑imbalance handling, threshold
  selection from precision–recall curves, and GIF/MP4 visualizations of
  fire‑spread risk.

  ---

  ## 🔍 Why This Project is Interesting

  - Mixes **geospatial ML** (remote sensing + reanalysis) with
    **production‑style backend** (FastAPI, tiled U‑Net inference).
  - Provides a **modern dashboard UI** (React + API‑driven design)
    instead of a static plot dump.
  - Includes **safety / UX considerations**: clear risk language,
    emergency hints, and an assistant to help non‑experts read the map.

  For research or SDE/ML internship applications, you can point to both
  the modeling notebook and the deployed dashboard to showcase end‑to‑end
  skills.
