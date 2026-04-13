// SmartAir-Guardian — static/js/dashboard.js
// Polls /api/history every 3 seconds and updates the UI.
// Requires api.js to be loaded first.

const POLL_INTERVAL_MS = 3000;

const GAS_NAMES = {
    0: "Clean Air",
    1: "Smoke / CO",
    2: "Alcohol / VOC",
    3: "NH3 / Ammonia",
    4: "Fire / Flame",
    5: "Mixed / LPG",
};

const SEVERITY_COLOR = {
    SAFE:    "#10b981",
    WARNING: "#f59e0b",
    DANGER:  "#ef4444",
};

// ── Poll latest reading ──────────────────────────────────────
async function pollLatest() {
    try {
        const { readings } = await API.history(1);
        if (!readings || readings.length === 0) return;
        updateLatestCard(readings[0]);
    } catch (err) {
        console.warn("[Dashboard] Poll error:", err.message);
    }
}

// ── Update the "latest reading" card ────────────────────────
function updateLatestCard(entry) {
    const p = entry.prediction;
    const s = entry.sensors;

    setEl("latest-gas",      p.gas_name       ?? "—");
    setEl("latest-severity", p.severity        ?? "—");
    setEl("latest-ppm",      `${p.ppm_estimate ?? 0} ppm`);
    setEl("latest-conf",     `${p.confidence   ?? 0}%`);
    setEl("latest-ts",       formatTs(entry.timestamp));

    // Sensor values
    setEl("val-mq135", s.mq135?.toFixed(2) ?? "—");
    setEl("val-mq3",   s.mq3?.toFixed(2)   ?? "—");
    setEl("val-mq7",   s.mq7?.toFixed(2)   ?? "—");
    setEl("val-mq4",   s.mq4?.toFixed(2)   ?? "—");
    setEl("val-temp",  s.temp?.toFixed(1)   ?? "—");
    setEl("val-hum",   s.hum?.toFixed(1)    ?? "—");
    setEl("val-flame", s.flame === 1 ? "YES" : "no");

    // Severity colour
    const sevEl = document.getElementById("latest-severity");
    if (sevEl) sevEl.style.color = SEVERITY_COLOR[p.severity] ?? "#e8eaf0";
}

// ── Helpers ──────────────────────────────────────────────────
function setEl(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

function formatTs(iso) {
    try {
        return new Date(iso).toLocaleTimeString();
    } catch {
        return iso;
    }
}

// ── Start polling ────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
    pollLatest();
    setInterval(pollLatest, POLL_INTERVAL_MS);
});
