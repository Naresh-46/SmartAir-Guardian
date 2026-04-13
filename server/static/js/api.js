// SmartAir-Guardian — static/js/api.js
// Reusable fetch helpers used by dashboard.html

const API = {
    BASE: window.location.origin,

    async predict(sensorData) {
        const res = await fetch(`${API.BASE}/api/predict`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify(sensorData),
        });
        if (!res.ok) throw new Error(`Predict failed: ${res.status}`);
        return res.json();
    },

    async history(n = 100) {
        const res = await fetch(`${API.BASE}/api/history?n=${n}`);
        if (!res.ok) throw new Error(`History failed: ${res.status}`);
        return res.json();
    },

    async stats() {
        const res = await fetch(`${API.BASE}/api/stats`);
        if (!res.ok) throw new Error(`Stats failed: ${res.status}`);
        return res.json();
    },

    async alerts(n = 20) {
        const res = await fetch(`${API.BASE}/api/alerts?n=${n}`);
        if (!res.ok) throw new Error(`Alerts failed: ${res.status}`);
        return res.json();
    },

    async status() {
        const res = await fetch(`${API.BASE}/api/status`);
        if (!res.ok) throw new Error(`Status failed: ${res.status}`);
        return res.json();
    },
};
