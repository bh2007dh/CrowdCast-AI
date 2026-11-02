/**
 * CrowdCast AI — Shared Frontend Controller (Geolocation, Radius, Manual Search, Theme).
 *
 * Strictly enforces:
 * - No continuous background tracking without user action (explicit Locate / Refresh control)
 * - No fake GPS coordinates masquerading as browser geolocation
 * - Clear distinction between Browser GPS, Manual Search, and Demonstration Campus presets
 */

const CrowdCastState = {
  STORAGE_KEY: "crowdcast_active_location_v1",
  THEME_KEY: "crowdcast_theme_v1",
  defaultLocation: {
    latitude: 13.0108,
    longitude: 80.2354,
    radius_m: 1000,
    city: "Chennai",
    locality: "Guindy Institutional Area",
    area_name: "Anna University & IIT Madras Corridor",
    formatted_address: "Sardar Patel Road, Guindy, Chennai, Tamil Nadu 600025",
    acquisition_mode: "default_campus_preset",
    data_label: "Default Reference Campus (Click 'Detect My GPS' for Live Browser Location)",
  },

  getLocation() {
    try {
      const raw = localStorage.getItem(this.STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed.latitude === "number" && typeof parsed.longitude === "number") {
          return parsed;
        }
      }
    } catch (e) {
      console.warn("Could not read location state from localStorage:", e);
    }
    return { ...this.defaultLocation };
  },

  saveLocation(locObj) {
    const merged = { ...this.getLocation(), ...locObj };
    localStorage.setItem(this.STORAGE_KEY, JSON.stringify(merged));
    this.renderContextBar(merged);
    window.dispatchEvent(new CustomEvent("crowdcast:locationChanged", { detail: merged }));
    return merged;
  },

  renderContextBar(loc) {
    const current = loc || this.getLocation();
    const labelEl = document.getElementById("global-location-label");
    const coordsEl = document.getElementById("global-location-coords");
    const modeBadgeEl = document.getElementById("global-location-mode-badge");
    const radiusSelect = document.getElementById("global-radius-select");

    if (labelEl) {
      labelEl.textContent = `${current.locality || current.city || "Detected Sector"} (${current.city || ""})`;
    }
    if (coordsEl) {
      coordsEl.textContent = `${Number(current.latitude).toFixed(4)}, ${Number(current.longitude).toFixed(4)}`;
    }
    if (modeBadgeEl) {
      if (current.acquisition_mode === "browser_geolocation") {
        modeBadgeEl.className = "data-badge badge-real";
        modeBadgeEl.textContent = "LIVE BROWSER GPS";
      } else if (current.acquisition_mode === "manual_search") {
        modeBadgeEl.className = "data-badge badge-estimated";
        modeBadgeEl.textContent = "MANUAL SEARCH LOCATION";
      } else {
        modeBadgeEl.className = "data-badge badge-demo";
        modeBadgeEl.textContent = "REFERENCE CAMPUS PRESET";
      }
    }
    if (radiusSelect && current.radius_m) {
      radiusSelect.value = String(current.radius_m);
    }
  },

  async requestBrowserGeolocation(onStatusUpdate) {
    if (!("geolocation" in navigator)) {
      if (onStatusUpdate) {
        onStatusUpdate("Browser Geolocation API is not supported in this browser. Use Manual Search.", "error");
      }
      return null;
    }

    if (onStatusUpdate) {
      onStatusUpdate("Requesting browser GPS permission...", "info");
    }

    return new Promise((resolve) => {
      navigator.geolocation.getCurrentPosition(
        async (position) => {
          const lat = Number(position.coords.latitude.toFixed(6));
          const lon = Number(position.coords.longitude.toFixed(6));
          const radius_m = Number(document.getElementById("global-radius-select")?.value || this.getLocation().radius_m || 1000);

          try {
            const resp = await fetch("/api/location/detect", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                latitude: lat,
                longitude: lon,
                radius_m: radius_m,
                source_mode: "browser_geolocation",
              }),
            });
            const data = await resp.json();
            if (data.status === "ok" && data.location) {
              const saved = this.saveLocation(data.location);
              if (onStatusUpdate) {
                onStatusUpdate(`Detected GPS Location: ${saved.formatted_address}`, "success");
              }
              resolve(saved);
              return;
            }
          } catch (err) {
            console.error("Backend reverse geocoding error:", err);
          }

          const fallbackSaved = this.saveLocation({
            latitude: lat,
            longitude: lon,
            radius_m: radius_m,
            acquisition_mode: "browser_geolocation",
          });
          resolve(fallbackSaved);
        },
        (err) => {
          let msg = "Location permission denied or unavailable. Please use Manual Location Search.";
          if (err.code === 1) msg = "GPS Permission Denied by user. Use Manual Search below.";
          if (err.code === 2) msg = "GPS position unavailable. Use Manual Search below.";
          if (err.code === 3) msg = "GPS request timed out. Use Manual Search below.";
          if (onStatusUpdate) onStatusUpdate(msg, "warning");
          resolve(null);
        },
        {
          enableHighAccuracy: true,
          timeout: 9000,
          maximumAge: 60000,
        }
      );
    });
  },

  async setLocationFromCoordinates(lat, lon, mode = "manual_search", customRadius = null) {
    const radius_m = Number(customRadius || document.getElementById("global-radius-select")?.value || this.getLocation().radius_m || 1000);
    const resp = await fetch("/api/location/detect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        latitude: Number(lat),
        longitude: Number(lon),
        radius_m: radius_m,
        source_mode: mode,
      }),
    });
    const data = await resp.json();
    if (data.status === "ok" && data.location) {
      return this.saveLocation(data.location);
    }
    throw new Error(data.error || "Failed to resolve coordinates.");
  },
};

function initTheme() {
  const savedTheme = localStorage.getItem(CrowdCastState.THEME_KEY) || "dark";
  document.documentElement.setAttribute("data-theme", savedTheme);
  const btn = document.getElementById("theme-toggle-btn");
  if (btn) {
    btn.textContent = savedTheme === "dark" ? "Light Theme" : "Dark Theme";
    btn.addEventListener("click", () => {
      const cur = document.documentElement.getAttribute("data-theme") || "dark";
      const next = cur === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      localStorage.setItem(CrowdCastState.THEME_KEY, next);
      btn.textContent = next === "dark" ? "Light Theme" : "Dark Theme";
    });
  }
}

document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  CrowdCastState.renderContextBar();

  const gpsBtn = document.getElementById("global-detect-gps-btn");
  if (gpsBtn) {
    gpsBtn.addEventListener("click", async () => {
      gpsBtn.disabled = true;
      gpsBtn.textContent = "Locating...";
      await CrowdCastState.requestBrowserGeolocation((msg) => {
        const toast = document.getElementById("global-status-toast");
        if (toast) {
          toast.textContent = msg;
          toast.style.display = "inline-block";
        }
      });
      gpsBtn.disabled = false;
      gpsBtn.textContent = "Refresh GPS";
    });
  }

  const radiusSelect = document.getElementById("global-radius-select");
  if (radiusSelect) {
    radiusSelect.addEventListener("change", async () => {
      const loc = CrowdCastState.getLocation();
      const newR = Number(radiusSelect.value);
      await CrowdCastState.setLocationFromCoordinates(loc.latitude, loc.longitude, loc.acquisition_mode, newR);
    });
  }
});

