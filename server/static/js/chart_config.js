// SmartAir-Guardian — static/js/chart_config.js
// Shared Chart.js defaults — load before any chart is created.

const CHART_DEFAULTS = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 400 },
    plugins: {
        legend: {
            labels: { color: "#6b7280", font: { family: "'JetBrains Mono', monospace", size: 11 } },
        },
        tooltip: {
            backgroundColor: "#141920",
            borderColor: "rgba(255,255,255,0.07)",
            borderWidth: 1,
            titleColor: "#e8eaf0",
            bodyColor: "#6b7280",
        },
    },
    scales: {
        x: {
            ticks: { color: "#6b7280", font: { family: "'JetBrains Mono', monospace", size: 10 } },
            grid:  { color: "rgba(255,255,255,0.04)" },
        },
        y: {
            ticks: { color: "#6b7280", font: { family: "'JetBrains Mono', monospace", size: 10 } },
            grid:  { color: "rgba(255,255,255,0.04)" },
        },
    },
};

// Helper — create a line chart with SmartAir defaults
function createLineChart(canvasId, labels, datasets) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    return new Chart(ctx, {
        type: "line",
        data: { labels, datasets },
        options: {
            ...CHART_DEFAULTS,
            elements: {
                point: { radius: 2, hoverRadius: 4 },
                line:  { tension: 0.3 },
            },
        },
    });
}

// Helper — create a doughnut chart
function createDoughnutChart(canvasId, labels, data, colors) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return null;
    return new Chart(ctx, {
        type: "doughnut",
        data: {
            labels,
            datasets: [{ data, backgroundColor: colors, borderWidth: 0 }],
        },
        options: {
            ...CHART_DEFAULTS,
            cutout: "70%",
            scales: {},   // doughnut has no axes
        },
    });
}
