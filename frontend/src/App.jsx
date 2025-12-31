import React, { useEffect, useMemo, useState } from "react";

const API_BASE = typeof __API_BASE__ !== "undefined" ? __API_BASE__ : "";

const DISTRICT_CONTACTS = {
  Dehradun: {
    label: "Dehradun district",
    forest: "State forest control room (via local DFO)",
    helpline: "112 / 1077 (India emergency / disaster)",
  },
  "Pauri Garhwal": {
    label: "Pauri Garhwal",
    forest: "Pauri forest division control room",
    helpline: "112 / 1077 (India emergency / disaster)",
  },
  Nainital: {
    label: "Nainital district",
    forest: "Nainital forest division control room",
    helpline: "112 / 1077 (India emergency / disaster)",
  },
  Almora: {
    label: "Almora district",
    forest: "Almora forest division control room",
    helpline: "112 / 1077 (India emergency / disaster)",
  },
  "Other Uttarakhand": {
    label: "Other Uttarakhand",
    forest: "Nearest forest range office / control room",
    helpline: "112 / 1077 (India emergency / disaster)",
  },
};

function formatDateRange(dates) {
  if (!dates?.length) return "";
  const sorted = [...dates].sort();
  const first = sorted[0];
  const last = sorted[sorted.length - 1];
  if (first === last) return first;
  return `${first} – ${last}`;
}

function App() {
  const [availableDates, setAvailableDates] = useState([]);
  const [loadingDates, setLoadingDates] = useState(true);
  const [dateError, setDateError] = useState("");

  const [useLatest, setUseLatest] = useState(true);
  const [selectedDate, setSelectedDate] = useState("");

  const [isPredicting, setIsPredicting] = useState(false);
  const [status, setStatus] = useState("Initializing dashboard…");
  const [forecastDate, setForecastDate] = useState("");
  const [imageUrl, setImageUrl] = useState("");
  const [mapLoaded, setMapLoaded] = useState(false);
  const [backendError, setBackendError] = useState("");
  const [highRiskFraction, setHighRiskFraction] = useState(null);
  const [moderateRiskFraction, setModerateRiskFraction] = useState(null);
  const [lowRiskFraction, setLowRiskFraction] = useState(null);
  const [highRiskFocus, setHighRiskFocus] = useState("");
  const [activeTab, setActiveTab] = useState("dashboard");
  const [safetyQuery, setSafetyQuery] = useState("");
  const [safetyAnswer, setSafetyAnswer] = useState("");
  const [selectedDistrict, setSelectedDistrict] = useState("Dehradun");
  const [assistantOpen, setAssistantOpen] = useState(false);
  const [assistantInput, setAssistantInput] = useState("");
  const [assistantMessages, setAssistantMessages] = useState([
    {
      role: "assistant",
      text: "Hi, I can help you understand this map and where the risk is higher today.",
    },
  ]);

  const dateRangeText = useMemo(
    () => formatDateRange(availableDates),
    [availableDates]
  );

  const latestApi = `${API_BASE}/api/available-dates`;
  const predictApi = `${API_BASE}/api/predict`;

  const currentContacts = DISTRICT_CONTACTS[selectedDistrict] ||
    DISTRICT_CONTACTS["Other Uttarakhand"];

  const mapStatus = isPredicting ? "running" : imageUrl ? "ready" : "idle";

  function getRiskLevelLabel() {
    if (highRiskFraction === null) return { level: "Unknown", color: "neutral" };
    const pct = highRiskFraction * 100;
    if (pct < 5) return { level: "Normal", color: "low" };
    if (pct < 15) return { level: "Caution", color: "medium" };
    return { level: "High alert", color: "high" };
  }

  function getPreparednessScore() {
    if (highRiskFraction === null) return "Unknown";
    const pct = highRiskFraction * 100;
    if (pct < 5) return "Low";
    if (pct < 15) return "Medium";
    return "High";
  }

  useEffect(() => {
    async function fetchDates() {
      try {
        setLoadingDates(true);
        setDateError("");
        const resp = await fetch(latestApi);
        if (!resp.ok) {
          throw new Error(`status ${resp.status}`);
        }
        const data = await resp.json();
        const dates = Array.isArray(data) ? data : data.dates || [];
        setAvailableDates(dates);
        if (dates.length) {
          setStatus("Ready. Select a date or use latest, then generate a fire risk map.");
        } else {
          setStatus("No input stacks found on server.");
        }
      } catch (err) {
        console.error("Failed to load available dates", err);
        setDateError("Unable to load available dates from API.");
        setStatus("Unable to load available dates from API.");
      } finally {
        setLoadingDates(false);
      }
    }

    fetchDates();
  }, [latestApi]);

  const minDate = useMemo(
    () => (availableDates.length ? availableDates[0] : undefined),
    [availableDates]
  );
  const maxDate = useMemo(
    () =>
      availableDates.length ? availableDates[availableDates.length - 1] : undefined,
    [availableDates]
  );

  async function handlePredict(e) {
    e.preventDefault();
    setBackendError("");
    setMapLoaded(false);

    let dateParam = "latest";
    if (!useLatest) {
      if (!selectedDate) {
        setStatus("Please choose a date or enable ‘Use latest’. ");
        return;
      }
      dateParam = selectedDate;
    }

    try {
      setIsPredicting(true);
      setStatus("Running prediction on server…");
      const url = `${predictApi}?date=${encodeURIComponent(dateParam)}`;
      const resp = await fetch(url);
      const text = await resp.text();

      if (!resp.ok) {
        throw new Error(text || `Request failed with status ${resp.status}`);
      }

      const data = JSON.parse(text);
      const {
        date,
        probability_png,
        overlay_png,
        high_risk_fraction,
        moderate_risk_fraction,
        low_risk_fraction,
        high_risk_focus,
      } = data;
      const img = overlay_png || probability_png;
      if (!img) {
        throw new Error("Server responded without an image URL.");
      }

      setForecastDate(date);
      setImageUrl(`${API_BASE}${img}?cacheBust=${Date.now()}`);
      setHighRiskFraction(
        typeof high_risk_fraction === "number" ? high_risk_fraction : null
      );
      setModerateRiskFraction(
        typeof moderate_risk_fraction === "number" ? moderate_risk_fraction : null
      );
      setLowRiskFraction(
        typeof low_risk_fraction === "number" ? low_risk_fraction : null
      );
      setHighRiskFocus(high_risk_focus || "");
      setStatus("Prediction complete. Visualizing next-day fire risk.");
    } catch (err) {
      console.error("Prediction error", err);
      setBackendError(err.message || String(err));
      setStatus("Error while running prediction.");
      setImageUrl("");
      setHighRiskFraction(null);
      setModerateRiskFraction(null);
      setLowRiskFraction(null);
      setHighRiskFocus("");
    } finally {
      setIsPredicting(false);
    }
  }

  function handleSafetySearch(e) {
    e.preventDefault();
    const q = safetyQuery.trim().toLowerCase();
    if (!q) {
      setSafetyAnswer("Type a town, district, or area name.");
      return;
    }

    if (highRiskFraction === null) {
      setSafetyAnswer("Run a forecast on the Dashboard first to see todays risk.");
      return;
    }

    const focus = (highRiskFocus || "").toLowerCase();
    let areaHint = "this state-level map";
    let areaDistrictKey = "Other Uttarakhand";

    if (q.includes("pauri")) {
      areaHint = "Pauri Garhwal";
      areaDistrictKey = "Pauri Garhwal";
    } else if (q.includes("dehradun")) {
      areaHint = "Dehradun";
      areaDistrictKey = "Dehradun";
    } else if (q.includes("almora")) {
      areaHint = "Almora";
      areaDistrictKey = "Almora";
    } else if (q.includes("nainital")) {
      areaHint = "Nainital";
      areaDistrictKey = "Nainital";
    } else if (q.includes("pithoragarh")) {
      areaHint = "Pithoragarh";
    } else if (q.includes("haridwar")) {
      areaHint = "Haridwar / Terai belt";
    }

    const isNearFocus = focus && areaHint !== "this state-level map" && focus.includes(areaHint.toLowerCase().split(" ")[0]);

    const { level } = getRiskLevelLabel();
    const contacts = DISTRICT_CONTACTS[areaDistrictKey] || DISTRICT_CONTACTS["Other Uttarakhand"];

    if (!focus) {
      setSafetyAnswer(
        `Risk level today: ${level}. The map does not highlight one strong focus area. Still follow local fire and forest-department alerts for ${areaHint}. Nearest support: ${contacts.forest}; helpline ${contacts.helpline}.`
      );
      return;
    }

    if (isNearFocus) {
      setSafetyAnswer(
        `Risk level today: ${level}. The map shows higher risk near ${highRiskFocus}. If you are around ${areaHint}, stay alert, avoid open fires, and follow official instructions. Nearest support: ${contacts.forest}; helpline ${contacts.helpline}.`
      );
    } else {
      setSafetyAnswer(
        `Risk level today: ${level}. Todays main high-risk focus is around ${highRiskFocus}. The area you searched (${areaHint}) is not at the center of this hotspot, but you should still follow local advisories. Nearest support: ${contacts.forest}; helpline ${contacts.helpline}.`
      );
    }
  }

  function buildAssistantReply(questionRaw) {
    const q = questionRaw.toLowerCase();

    if (q.includes("what is this") || q.includes("about this app")) {
      return "Agnirodhak shows a next-day wildfire risk map for Uttarakhand using a U-Net model. You choose a date, it predicts which pixels are more likely to see fire.";
    }

    if (q.includes("high risk") || q.includes("red") || q.includes("orange")) {
      return "Areas marked as high risk are places where the model thinks conditions are favourable for fire ignition. It does not guarantee a fire will occur, but they deserve more attention.";
    }

    if (q.includes("safe") || q.includes("is my area")) {
      if (highRiskFraction === null) {
        return "Run a forecast on the Dashboard first, then you can check the Safety tab or ask again about your area.";
      }
      if (!highRiskFocus) {
        return "Today the map does not highlight one strong hotspot. You should still follow local fire and forest-department advisories.";
      }
      return `Today the main hotspot is around ${highRiskFocus}. If you are far from this region, risk may be lower but you must still follow local advisories.`;
    }

    if (q.includes("what should i do") || q.includes("what to do") || q.includes("precaution") || q.includes("safety")) {
      return "On high-risk days, avoid open fires near forests, follow instructions from local authorities, and keep basic items ready (water, torch, medicines, documents). Use the Safety tab for more tips.";
    }

    if (q.includes("accuracy") || q.includes("correct") || q.includes("reliable")) {
      return "This model is trained on past data and can make mistakes. Use it as an additional map, not as the only source. Always trust official warnings more.";
    }

    return "I may not fully understand that question. Try asking about what the map means, where the risk is higher today, or what precautions to take.";
  }

  function handleAssistantSubmit(e) {
    e.preventDefault();
    const trimmed = assistantInput.trim();
    if (!trimmed) return;

    const reply = buildAssistantReply(trimmed);

    setAssistantMessages((prev) => [
      ...prev,
      { role: "user", text: trimmed },
      { role: "assistant", text: reply },
    ]);
    setAssistantInput("");
  }

  return (
    <div className="page-root">
      <header className="app-header">
        <div className="brand">
          <div className="brand-mark">
            <span className="brand-mark-earth" />
            <span className="brand-mark-india" />
            <span className="brand-mark-orbit-dot" />
          </div>
          <div className="brand-text">
            <h1>Agnirodhak</h1>
            <p>India Forest Fire Early Warning for Uttarakhand</p>
          </div>
        </div>

        <nav className="nav-links">
          <button
            type="button"
            className={activeTab === "dashboard" ? "nav-link active" : "nav-link"}
            onClick={() => setActiveTab("dashboard")}
          >
            Dashboard
          </button>
          <button
            type="button"
            className={activeTab === "model" ? "nav-link active" : "nav-link"}
            onClick={() => setActiveTab("model")}
          >
            Model
          </button>
          <button
            type="button"
            className={activeTab === "safety" ? "nav-link active" : "nav-link"}
            onClick={() => setActiveTab("safety")}
          >
            Safety
          </button>
          <button
            type="button"
            className={activeTab === "about" ? "nav-link active" : "nav-link"}
            onClick={() => setActiveTab("about")}
          >
            About
          </button>
        </nav>
      </header>

      <section className="hero" id="top">
        <div className="hero-copy">
          <h2 className="hero-title">Wildfire risk, at a glance.</h2>
          <p className="hero-text">
            Forecast next-day forest fire danger over Uttarakhand using
            satellite-style maps and a U-Net model.
          </p>
        </div>
        <div className="hero-orbit-card">
          <div className="hero-orbit-label">Region</div>
          <div className="hero-orbit-title">Uttarakhand, India</div>
          <p className="hero-orbit-text">Focus on practical, map-first risk visualization.</p>
        </div>
      </section>

      {activeTab === "dashboard" && (
        <main className="layout" id="dashboard">
        <section className="panel panel-left">
          <h2>Forecast Controls</h2>
          <p className="panel-subtitle">
            Pick a date and generate tomorrow&apos;s risk map.
          </p>

          <div className="chip-row">
            <div className="chip">📅 Lead time: +1 day</div>
            <div className="chip chip-has-tooltip">
              🤖 Model: U-Net Scenario 1
              <span className="chip-tooltip">Deep CNN for pixel-wise fire risk.</span>
            </div>
            <div className="chip chip-has-tooltip">
              🗺️ Resolution: 1 km
              <span className="chip-tooltip">Each pixel ≈ 1 km grid cell.</span>
            </div>
          </div>

          <form className="control-form" onSubmit={handlePredict}>
            <label className="field-label" htmlFor="date-input">
              📅 Forecast date
            </label>

            <div className="date-row">
              <input
                id="date-input"
                className="date-input"
                type="date"
                disabled={useLatest}
                min={minDate}
                max={maxDate}
                value={selectedDate}
                onChange={(e) => setSelectedDate(e.target.value)}
              />

              <label className="toggle">
                <input
                  type="checkbox"
                  checked={useLatest}
                  onChange={(e) => setUseLatest(e.target.checked)}
                />
                <span className="toggle-slider" />
                <span className="toggle-label">Use latest</span>
              </label>
            </div>

            <button
              className={`primary-btn ${isPredicting ? "primary-btn-loading" : ""}`}
              type="submit"
              disabled={isPredicting}
            >
              <span className="primary-btn-icon">{isPredicting ? "⏳" : "🔥"}</span>
              <span className="primary-btn-text">
                {isPredicting ? "Running U-Net model…" : "Generate Fire Risk Map"}
              </span>
            </button>
          </form>

          <div className="status">{status}</div>
          {dateError && <div className="status error">{dateError}</div>}
          {backendError && (
            <div className="status error">API error: {backendError}</div>
          )}
        </section>

        <section className="panel panel-right">
          <div className="map-header">
            <h2>Fire Risk Map 🗺️</h2>
            <div className="map-header-right">
              <span
                className={`map-status-dot map-status-${mapStatus}`}
                aria-label={
                  mapStatus === "running"
                    ? "Prediction running"
                    : mapStatus === "ready"
                    ? "Latest map ready"
                    : "Idle"
                }
              />
              <span className="date-label">
                {forecastDate
                  ? `Forecast based on stack date: ${forecastDate}`
                  : "No forecast yet"}
              </span>
            </div>
          </div>

          <div className="map-container">
            {isPredicting && (
              <div className="spinner" aria-label="Loading prediction">
                <div className="spinner-text">Running U-Net model on server…</div>
              </div>
            )}

            {imageUrl ? (
              <img
                className={`map-image ${mapLoaded ? "map-image-visible" : ""}`}
                src={imageUrl}
                alt="Fire risk map"
                onLoad={() => setMapLoaded(true)}
              />
            ) : (
              !isPredicting && (
                <div className="map-placeholder">
                  <div className="map-placeholder-graphic" />
                  <p className="map-placeholder-text">
                    Select a date and click <strong>Generate Fire Risk Map</strong> to
                    see tomorrow&apos;s forecast.
                  </p>
                </div>
              )
            )}
          </div>

          <div className={`legend ${mapLoaded ? "legend-active" : ""}`}>
            <div className="legend-item">
              <span className="legend-swatch swatch-low" />
              <span>Low risk</span>
            </div>
            <div className="legend-item">
              <span className="legend-swatch swatch-med" />
              <span>Moderate risk</span>
            </div>
            <div className="legend-item">
              <span className="legend-swatch swatch-high" />
              <span>High risk</span>
            </div>
          </div>

          {(highRiskFraction !== null || moderateRiskFraction !== null || lowRiskFraction !== null || highRiskFocus) && (
            <div className="risk-summary" aria-label="Risk summary">
              <div className="risk-summary-title">Today&apos;s snapshot</div>
              <div className="risk-summary-cards">
                {highRiskFraction !== null && (
                  <div className="risk-card risk-card-high">
                    <div className="risk-card-label">🟥 High risk</div>
                    <div className="risk-card-value">
                      {(highRiskFraction * 100).toFixed(1)}%
                    </div>
                  </div>
                )}
                {moderateRiskFraction !== null && (
                  <div className="risk-card risk-card-med">
                    <div className="risk-card-label">🟨 Moderate</div>
                    <div className="risk-card-value">
                      {(moderateRiskFraction * 100).toFixed(1)}%
                    </div>
                  </div>
                )}
                {lowRiskFraction !== null && (
                  <div className="risk-card risk-card-low">
                    <div className="risk-card-label">🟩 Low</div>
                    <div className="risk-card-value">
                      {(lowRiskFraction * 100).toFixed(1)}%
                    </div>
                  </div>
                )}
              </div>
              {highRiskFocus && (
                <div className="risk-summary-focus">
                  Highest concentration around <strong>{highRiskFocus}</strong>.
                </div>
              )}
            </div>
          )}
        </section>
        </main>
      )}

      {activeTab === "model" && (
        <section className="panel panel-bottom" id="model">
          <h2>Model Details</h2>
          <p className="panel-subtitle">
            How the U-Net model, daily feature stacks, and FastAPI backend work
            together.
          </p>
          <ul className="model-list">
            <li><strong>Backend:</strong> FastAPI + TensorFlow/Keras U-Net model.</li>
            <li><strong>Frontend:</strong> React dashboard (Vite) consuming REST APIs.</li>
            <li>
              <strong>Inputs:</strong> temperature, u/v wind, wind speed, NDVI, burn
              date, LULC, DEM, slope, aspect, hillshade.
            </li>
            <li>
              <strong>Outputs:</strong> next-day ignition probability and high-risk
              mask, saved as PNG overlays.
            </li>
            <li>
              <strong>Model performance:</strong> AUC ≈ 0.82 on held-out days (using
              ERA5 reanalysis and MODIS-derived vegetation/terrain features).
            </li>
          </ul>
        </section>
      )}

      {activeTab === "safety" && (
        <section className="panel panel-bottom" id="safety">
          <h2>Precautions &amp; Response</h2>
          <p className="panel-subtitle">
            Simple guidance for non-technical users on days when wildfire risk
            is high.
          </p>

          <div className={`risk-banner risk-banner-${getRiskLevelLabel().color}`}>
            <div className="risk-banner-icon">
              {getRiskLevelLabel().color === "high"
                ? "🔴"
                : getRiskLevelLabel().color === "medium"
                ? "🟡"
                : getRiskLevelLabel().color === "low"
                ? "🟢"
                : "ℹ️"}
            </div>
            <div className="risk-banner-text">
              <div className="risk-banner-title">
                {getRiskLevelLabel().color === "high"
                  ? "High fire risk day"
                  : getRiskLevelLabel().color === "medium"
                  ? "Caution: elevated fire risk"
                  : getRiskLevelLabel().color === "low"
                  ? "Normal background risk"
                  : "Run todays forecast to see risk level"}
              </div>
              <div className="risk-banner-sub">
                {highRiskFraction === null
                  ? "Go to the Dashboard, run a forecast, and this banner will update."
                  : "These precautions are especially important today."}
              </div>
              <div className="risk-banner-prep">
                Preparedness score for today: <strong>{getPreparednessScore()}</strong>
              </div>
            </div>
          </div>

          <div className="safety-grid">
            <div className="safety-column">
              <div className="safety-block safety-block-safe">
                <h3 className="safety-subtitle">If you live near forests</h3>
                <ul className="model-list">
                  <li>🔥 Avoid burning crop residue, trash, or campfires.</li>
                  <li>🌿 Clear dry leaves and branches from around your home.</li>
                  <li>🎒 Keep a small go-bag ready with:</li>
                  <li>• Water</li>
                  <li>• Torch</li>
                  <li>• Medicines</li>
                  <li>• Important documents</li>
                </ul>
              </div>

              <div className="safety-block safety-block-danger">
                <h3 className="safety-subtitle">If you see smoke or fire</h3>
                <ul className="model-list">
                  <li>🚶‍♀️ Move away from dense vegetation and upwind of smoke.</li>
                  <li>📞 Call your local forest or emergency contact quickly.</li>
                  <li>🚗 Keep forest tracks and approach roads clear for responders.</li>
                </ul>
              </div>

              <div className="safety-block safety-block-warning">
                <h3 className="safety-subtitle">What NOT to do</h3>
                <ul className="model-list">
                  <li>❌ Do not use drones near active fires.</li>
                  <li>❌ Do not block forest roads or fire trucks.</li>
                  <li>❌ Do not spread unverified fire alerts on WhatsApp.</li>
                </ul>
              </div>
            </div>

            <div className="safety-column">
              <h3 className="safety-subtitle">Emergency contacts</h3>
              <div className="district-select-row">
                <label className="field-label" htmlFor="district-select">
                  Choose district / area
                </label>
                <select
                  id="district-select"
                  className="district-select"
                  value={selectedDistrict}
                  onChange={(e) => setSelectedDistrict(e.target.value)}
                >
                  {Object.keys(DISTRICT_CONTACTS).map((key) => (
                    <option key={key} value={key}>
                      {DISTRICT_CONTACTS[key].label}
                    </option>
                  ))}
                </select>
              </div>

              <ul className="model-list">
                <li>
                  <span className="contact-label">Forest control room</span>
                  <span className="contact-value">{currentContacts.forest}</span>
                </li>
                <li>
                  <span className="contact-label">Fire / disaster helpline</span>
                  <span className="contact-value">{currentContacts.helpline}</span>
                </li>
              </ul>

              <form className="safety-search" onSubmit={handleSafetySearch}>
                <label className="field-label" htmlFor="area-search">
                  Check an area
                </label>
                <div className="safety-search-row">
                  <input
                    id="area-search"
                    className="date-input"
                    type="text"
                    placeholder="Enter district or town (e.g. Dehradun)"
                    value={safetyQuery}
                    onChange={(e) => setSafetyQuery(e.target.value)}
                  />
                  <button className="search-btn" type="submit" aria-label="Search area">
                    🔍
                  </button>
                </div>
                {safetyAnswer && (
                  <div className="status safety-answer">{safetyAnswer}</div>
                )}
              </form>

              <div className="assistant-card">
                <div className="assistant-header">
                  <span className="assistant-icon">🛰️</span>
                  <span className="assistant-title">Guidance helper</span>
                </div>
                <p className="assistant-text">
                  For detailed evacuation or shelter information, always follow
                  official instructions from the Uttarakhand Forest Department
                  and district administration.
                </p>
              </div>
            </div>
          </div>
        </section>
      )}

      {activeTab === "about" && (
        <section className="panel panel-bottom" id="about-panel">
          <h2>About this project</h2>
          <p className="panel-subtitle">
            Student research project built as a full-stack wildfire early
            warning demo for Uttarakhand.
          </p>
          <ul className="model-list">
            <li>Combines reanalysis weather, vegetation, and terrain data.</li>
            <li>Implements a U-Net segmentation model for next-day ignition.</li>
            <li>Exposes predictions via a FastAPI backend and React UI.</li>
          </ul>
        </section>
      )}

      <footer className="app-footer" id="about">
        <span>Uttarakhand region · daily ERA5 &amp; MODIS features</span>
        <span>·</span>
        <span>
          Agnirodhak – designed for forest departments, disaster response teams,
          and researchers.
        </span>
      </footer>

      {!assistantOpen && <div className="assistant-fab-label">Help</div>}

      <button
        type="button"
        className="assistant-fab"
        onClick={() => setAssistantOpen((open) => !open)}
        aria-label="Open assistant"
      >
        💬
      </button>

      {assistantOpen && (
        <div className="assistant-panel" aria-label="Assistant chat">
          <div className="assistant-panel-header">
            <span className="assistant-panel-title">Map helper</span>
            <button
              type="button"
              className="assistant-close"
              onClick={() => setAssistantOpen(false)}
              aria-label="Close assistant"
            >
              ×
            </button>
          </div>
          <div className="assistant-messages">
            {assistantMessages.map((m, idx) => (
              <div
                key={idx}
                className={
                  m.role === "assistant"
                    ? "assistant-bubble assistant-bubble-assistant"
                    : "assistant-bubble assistant-bubble-user"
                }
              >
                {m.text}
              </div>
            ))}
          </div>
          <div className="assistant-suggestions">
            <button
              type="button"
              onClick={() => {
                const q = "What areas are high risk today?";
                const reply = buildAssistantReply(q);
                setAssistantMessages((prev) => [
                  ...prev,
                  { role: "user", text: q },
                  { role: "assistant", text: reply },
                ]);
              }}
            >
              What areas are high risk today?
            </button>
            <button
              type="button"
              onClick={() => {
                const q = "What precautions should be taken?";
                const reply = buildAssistantReply(q);
                setAssistantMessages((prev) => [
                  ...prev,
                  { role: "user", text: q },
                  { role: "assistant", text: reply },
                ]);
              }}
            >
              What precautions should be taken?
            </button>
            <button
              type="button"
              onClick={() => {
                const q = "How accurate is this model?";
                const reply = buildAssistantReply(q);
                setAssistantMessages((prev) => [
                  ...prev,
                  { role: "user", text: q },
                  { role: "assistant", text: reply },
                ]);
              }}
            >
              How accurate is this model?
            </button>
          </div>
          <form className="assistant-input-row" onSubmit={handleAssistantSubmit}>
            <input
              type="text"
              className="assistant-input"
              placeholder="Ask about the map, risk, or safety…"
              value={assistantInput}
              onChange={(e) => setAssistantInput(e.target.value)}
            />
            <button type="submit" className="assistant-send" aria-label="Send">
              ➤
            </button>
          </form>
        </div>
      )}
    </div>
  );
}

export default App;
