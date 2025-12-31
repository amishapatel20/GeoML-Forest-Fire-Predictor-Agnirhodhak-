const form = document.getElementById("control-form");
const dateInput = document.getElementById("date-input");
const latestToggle = document.getElementById("latest-toggle");
const statusText = document.getElementById("status-text");
const spinner = document.getElementById("spinner");
const mapImage = document.getElementById("map-image");
const dateLabel = document.getElementById("date-label");

function setLoading(isLoading) {
  if (isLoading) {
    spinner.classList.remove("hidden");
    mapImage.classList.add("hidden");
    statusText.textContent = "Running prediction...";
  } else {
    spinner.classList.add("hidden");
  }
}

latestToggle.addEventListener("change", () => {
  if (latestToggle.checked) {
    dateInput.disabled = true;
    dateInput.classList.add("disabled-input");
  } else {
    dateInput.disabled = false;
    dateInput.classList.remove("disabled-input");
  }
});

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  let dateParam = "latest";
  if (!latestToggle.checked) {
    const val = dateInput.value;
    if (!val) {
      statusText.textContent = "Please select a date or enable 'Use latest'.";
      return;
    }
    dateParam = val;
  }

  setLoading(true);

  try {
    const resp = await fetch(`/api/predict?date=${encodeURIComponent(dateParam)}`);
    if (!resp.ok) {
      const msg = await resp.text();
      throw new Error(msg || `Request failed with status ${resp.status}`);
    }

    const data = await resp.json();
    const { date, probability_png, overlay_png } = data;

    const imgUrl = overlay_png || probability_png;
    if (!imgUrl) {
      throw new Error("Server did not return an image URL.");
    }

    mapImage.src = imgUrl + `?cacheBust=${Date.now()}`;
    mapImage.onload = () => {
      mapImage.classList.remove("hidden");
      setLoading(false);
    };

    dateLabel.textContent = `Forecast based on stack date: ${date}`;
    statusText.textContent =
      "Prediction complete. Visualizing probability of next-day fire ignition.";
  } catch (err) {
    console.error(err);
    setLoading(false);
    statusText.textContent = `Error: ${err.message}`;
  }
});

// Initialize state
latestToggle.dispatchEvent(new Event("change"));
statusText.textContent = "Waiting for prediction…";
