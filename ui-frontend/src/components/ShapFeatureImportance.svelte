<script>
  import PlotlyChart from "./PlotlyChart.svelte";

  // explanation object from API response
  export let explanation = null;
  // number of top features to show
  export let maxFeatures = 15;
  // title override
  export let title = "Feature Importance (SHAP)";

  // build chart data from explanation
  $: chartData = buildChartData(explanation, maxFeatures);
  $: chartLayout = buildLayout(title, maxFeatures);

  function buildChartData(exp, max) {
    if (!exp || !exp.top_features || exp.top_features.length === 0) {
      return null;
    }

    // take top N features (already sorted by absolute SHAP value)
    const features = exp.top_features.slice(0, max);

    // reverse for horizontal bar chart (top feature at top)
    const reversed = [...features].reverse();

    return [
      {
        type: "bar",
        orientation: "h",
        x: reversed.map((f) => f.shap_value),
        y: reversed.map((f) => truncateLabel(f.feature, 25)),
        marker: {
          color: reversed.map((f) =>
            f.shap_value >= 0 ? "#10b981" : "#ef4444"
          ),
        },
        hovertemplate:
          "<b>%{y}</b><br>" +
          "SHAP: %{x:.4f}<br>" +
          "<extra></extra>",
        text: reversed.map((f) => formatShapValue(f.shap_value)),
        textposition: "outside",
        textfont: { size: 10 },
      },
    ];
  }

  function buildLayout(title, max) {
    return {
      title: {
        text: title,
        font: { size: 14, color: "#e2e8f0" },
      },
      xaxis: {
        title: { text: "SHAP Value", font: { size: 11, color: "#94a3b8" } },
        zeroline: true,
        zerolinecolor: "#475569",
        gridcolor: "#334155",
        tickfont: { size: 10, color: "#94a3b8" },
      },
      yaxis: {
        automargin: true,
        tickfont: { size: 10, color: "#94a3b8" },
      },
      height: Math.max(300, max * 22 + 80),
      margin: { l: 140, r: 50, t: 50, b: 50 },
      paper_bgcolor: "rgba(15, 23, 42, 0)",
      plot_bgcolor: "rgba(15, 23, 42, 0)",
      showlegend: false,
    };
  }

  function truncateLabel(label, maxLen) {
    if (label.length <= maxLen) return label;
    return label.substring(0, maxLen - 3) + "...";
  }

  function formatShapValue(val) {
    if (Math.abs(val) < 0.001) return "";
    return val >= 0 ? `+${val.toFixed(3)}` : val.toFixed(3);
  }
</script>

{#if chartData}
  <div class="shap-importance">
    <PlotlyChart data={chartData} layout={chartLayout} />

    {#if explanation && explanation.base_value !== undefined}
      <div class="base-info">
        <span class="label">Base value:</span>
        <span class="value">{explanation.base_value.toFixed(4)}</span>
        {#if explanation.predicted_probability !== undefined}
          <span class="separator">|</span>
          <span class="label">Prediction:</span>
          <span class="value highlight"
            >{(explanation.predicted_probability * 100).toFixed(1)}%</span
          >
        {/if}
      </div>
    {/if}
  </div>
{:else}
  <div class="no-data">
    <p>No feature importance data available</p>
  </div>
{/if}

<style>
  .shap-importance {
    width: 100%;
  }

  .base-info {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    margin-top: 0.75rem;
    padding: 0.5rem 1rem;
    background: rgba(30, 41, 59, 0.5);
    border-radius: 0.5rem;
    font-size: 0.85rem;
  }

  .base-info .label {
    color: #94a3b8;
  }

  .base-info .value {
    color: #e2e8f0;
    font-weight: 600;
    font-family: monospace;
  }

  .base-info .value.highlight {
    color: #38bdf8;
  }

  .base-info .separator {
    color: #475569;
    margin: 0 0.25rem;
  }

  .no-data {
    padding: 2rem;
    text-align: center;
    color: #64748b;
    background: rgba(30, 41, 59, 0.3);
    border-radius: 0.75rem;
    border: 1px dashed #334155;
  }

  .no-data p {
    margin: 0;
  }
</style>
