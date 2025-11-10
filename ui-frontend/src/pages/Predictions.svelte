<script>
  import { onMount } from "svelte";
  import PlotlyChart from "../components/PlotlyChart.svelte";
  import { fetchStatus, predictClassification, predictSurvival } from "../lib/api";

  const formatInputValue = (date = new Date()) => {
    const local = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
    return local.toISOString().slice(0, 16);
  };

  let status;
  let statusLoading = true;
  let statusError = "";

  let timestamp = formatInputValue(new Date());
  let regionNumber = "";

  // classification form
  let windowHours = 24;
  let modelType = "gradient_boosting";
  let classificationStatus = "";
  let classificationResult = null;
  let classificationLoading = false;
  let classificationChart = null;

  // survival form
  let survivalModel = "cox";
  let survivalStatus = "";
  let survivalResult = null;
  let survivalLoading = false;
  let survivalCurve = null;
  let survivalDistribution = null;

  const loadStatus = async () => {
    statusLoading = true;
    statusError = "";
    try {
      status = await fetchStatus();
    } catch (err) {
      statusError = err.message;
    } finally {
      statusLoading = false;
    }
  };

  onMount(loadStatus);

  const toISO = (value) => {
    if (!value) return new Date().toISOString();
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return new Date().toISOString();
    }
    return date.toISOString();
  };

  const parsedRegion = () => {
    const val = Number.parseInt(regionNumber, 10);
    return Number.isNaN(val) ? null : val;
  };

  const runClassification = async (forceRefresh = false) => {
    classificationLoading = true;
    classificationStatus = "";
    try {
      const payload = {
        timestamp: toISO(timestamp),
        window: Number(windowHours),
        regionNumber: parsedRegion(),
        modelType,
        forceRefresh
      };
      const result = await predictClassification(payload);
      classificationResult = result.result;
      classificationChart = [
        {
          type: "bar",
          x: result.result.orderedLabels,
          y: result.result.orderedValues,
          text: result.result.orderedValues.map((v) => `${v.toFixed(1)}%`),
          textposition: "outside",
          marker: { color: "#38bdf8" }
        }
      ];
      classificationStatus = result.status;
      if (forceRefresh) {
        await loadStatus();
      }
    } catch (err) {
      classificationStatus = err.message;
    } finally {
      classificationLoading = false;
    }
  };

  const runSurvival = async (forceRefresh = false) => {
    survivalLoading = true;
    survivalStatus = "";
    try {
      const payload = {
        timestamp: toISO(timestamp),
        regionNumber: parsedRegion(),
        modelType: survivalModel,
        forceRefresh
      };
      const result = await predictSurvival(payload);
      survivalResult = result.result;
      survivalCurve = [
        {
          type: "scatter",
          mode: "lines",
          x: result.result.survivalCurve?.time_points ?? [],
          y: result.result.survivalCurve?.probabilities ?? [],
          line: { color: "#a78bfa", width: 3 }
        }
      ];
      survivalDistribution = [
        {
          type: "bar",
          x: result.result.orderedBuckets,
          y: result.result.orderedValues,
          marker: { color: "#f97316" },
          text: result.result.orderedValues.map((v) => `${v.toFixed(1)}%`),
          textposition: "outside"
        }
      ];
      survivalStatus = result.status;
      if (forceRefresh) {
        await loadStatus();
      }
    } catch (err) {
      survivalStatus = err.message;
    } finally {
      survivalLoading = false;
    }
  };
</script>

<div class="page">
  <div class="card">
    <div class="section-title">Prediction confidence</div>
    {#if statusLoading}
      Loading guardrail status…
    {:else if statusError}
      {statusError}
    {:else if status}
      {#if status.connection.confidence}
        <p>
          <strong>Confidence:</strong> {status.connection.confidence}
          {#if status.connection.guardrailReason}
            — {status.connection.guardrailReason}
          {/if}
        </p>
      {:else}
        API did not publish confidence metadata. Ensure the API server is reachable.
      {/if}
    {/if}
  </div>

  <div class="grid two">
    <div class="card">
      <h2>Classification (24-48h window)</h2>
      <p>
        Predicts the maximum flare class (None/C/M/X) expected within the selected window. Use the force refresh button
        to bypass throttling if you just updated data.
      </p>

      <form on:submit|preventDefault={() => runClassification(false)}>
        <label>
          Observation timestamp
          <input type="datetime-local" bind:value={timestamp} max="9999-12-31T23:59" />
        </label>
        <label>
          Region number (optional)
          <input type="number" bind:value={regionNumber} placeholder="e.g., 3598" min="0" />
        </label>

        <div class="grid two">
          <label>
            Prediction window (hours)
            <select bind:value={windowHours}>
              <option value="24">24h</option>
              <option value="48">48h</option>
            </select>
          </label>
          <label>
            Model
            <select bind:value={modelType}>
              <option value="logistic">Logistic regression</option>
              <option value="gradient_boosting">Gradient boosting</option>
            </select>
          </label>
        </div>

        <div class="button-row">
          <button class="primary" type="submit" disabled={classificationLoading}>
            {classificationLoading ? "Predicting…" : "Predict"}
          </button>
          <button
            class="secondary"
            type="button"
            on:click={() => runClassification(true)}
            disabled={classificationLoading}
          >
            Force refresh
          </button>
        </div>
      </form>

      {#if classificationStatus}
        <p>{classificationStatus}</p>
      {/if}

      {#if classificationResult}
        <pre>{classificationResult.text}</pre>
      {/if}

      {#if classificationChart}
        <div class="card chart-card" style="margin-top: 1rem;">
          <PlotlyChart
            layout={{
              title: "Probability distribution",
              yaxis: {
                title: "Probability (%)",
                range: [
                  0,
                  Math.max(...(classificationResult?.orderedValues ?? [0]), 1) + 10
                ]
              }
            }}
            data={classificationChart}
          />
        </div>
      {/if}
    </div>

    <div class="card">
      <h2>Survival analysis (time-to-event)</h2>
      <p>
        Predicts when a C-class flare is likely to occur using survival analysis. The chart shows the survival curve and
        bucketed probabilities for different horizons.
      </p>

      <form on:submit|preventDefault={() => runSurvival(false)}>
        <label>
          Observation timestamp
          <input type="datetime-local" bind:value={timestamp} max="9999-12-31T23:59" />
        </label>
        <label>
          Region number (optional)
          <input type="number" bind:value={regionNumber} placeholder="e.g., 3598" min="0" />
        </label>
        <label>
          Model
          <select bind:value={survivalModel}>
            <option value="cox">Cox proportional hazards</option>
            <option value="gb">Gradient boosting survival</option>
          </select>
        </label>
        <div class="button-row">
          <button class="primary" type="submit" disabled={survivalLoading}>
            {survivalLoading ? "Predicting…" : "Predict"}
          </button>
          <button class="secondary" type="button" on:click={() => runSurvival(true)} disabled={survivalLoading}>
            Force refresh
          </button>
        </div>
      </form>

      {#if survivalStatus}
        <p>{survivalStatus}</p>
      {/if}

      {#if survivalResult}
        <pre>{survivalResult.plainText}</pre>
        <pre>{survivalResult.distributionText}</pre>
      {/if}

      {#if survivalCurve}
        <div class="card chart-card" style="margin-top: 1rem;">
          <PlotlyChart
            data={survivalCurve}
            layout={{
              title: "Survival curve",
              xaxis: { title: "Hours" },
              yaxis: { title: "Survival probability", range: [0, 1.05] }
            }}
          />
        </div>
      {/if}

      {#if survivalDistribution}
        <div class="card chart-card" style="margin-top: 1rem;">
          <PlotlyChart
            data={survivalDistribution}
            layout={{
              title: "Probability distribution by bucket",
              yaxis: { title: "Probability (%)" },
              xaxis: { tickangle: -30 }
            }}
          />
        </div>
      {/if}
    </div>
  </div>
</div>
