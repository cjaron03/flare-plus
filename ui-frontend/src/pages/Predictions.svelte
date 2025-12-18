<script>
  import { onMount } from "svelte";
  import PlotlyChart from "../components/PlotlyChart.svelte";
  import LoadingSpinner from "../components/LoadingSpinner.svelte";
  import ShapFeatureImportance from "../components/ShapFeatureImportance.svelte";
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
  let classificationExplanation = null;
  let includeExplanation = true;
  let showExplanation = false;

  // survival form
  let survivalModel = "cox";
  let survivalStatus = "";
  let survivalResult = null;
  let survivalLoading = false;
  let survivalCurve = null;
  let survivalDistribution = null;
  let survivalExplanation = null;

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
    classificationResult = null;
    classificationChart = null;
    classificationExplanation = null;
    showExplanation = false;
    try {
      const payload = {
        timestamp: toISO(timestamp),
        window: Number(windowHours),
        region_number: parsedRegion(),
        model_type: modelType,
        force_refresh: forceRefresh,
        include_explanation: includeExplanation
      };
      const result = await predictClassification(payload);
      if (result && result.success === false) {
        classificationStatus = result.message || "Prediction failed";
        classificationLoading = false;
        return;
      }
      if (!result || !result.result) {
        classificationStatus = "Invalid response from server";
        classificationLoading = false;
        return;
      }
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
      // extract SHAP explanation if available
      if (result.result.explanation && !result.result.explanation.error) {
        classificationExplanation = result.result.explanation;
        showExplanation = true;
      }
      classificationStatus = result.status || "Prediction completed successfully";
      if (forceRefresh) {
        await loadStatus();
      }
    } catch (err) {
      classificationStatus = err.message || err.data?.message || "Failed to run prediction. Check that the API server is running and models are loaded.";
      classificationResult = null;
      classificationChart = null;
      classificationExplanation = null;
    } finally {
      classificationLoading = false;
    }
  };

  const runSurvival = async (forceRefresh = false) => {
    survivalLoading = true;
    survivalStatus = "";
    survivalResult = null;
    survivalCurve = null;
    survivalDistribution = null;
    try {
      const payload = {
        timestamp: toISO(timestamp),
        regionNumber: parsedRegion(),
        modelType: survivalModel,
        forceRefresh
      };
      const result = await predictSurvival(payload);
      if (result && result.success === false) {
        survivalStatus = result.message || "Prediction failed";
        survivalLoading = false;
        return;
      }
      if (!result || !result.result) {
        survivalStatus = "Invalid response from server";
        survivalLoading = false;
        return;
      }
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
      survivalStatus = result.status || "Prediction completed successfully";
      if (forceRefresh) {
        await loadStatus();
      }
    } catch (err) {
      survivalStatus = err.message || err.data?.message || "Failed to run prediction. Check that the API server is running and models are loaded.";
      survivalResult = null;
      survivalCurve = null;
      survivalDistribution = null;
    } finally {
      survivalLoading = false;
    }
  };
</script>

<div class="page">
  <div class="card" style="margin-bottom: 1.5rem;">
    <h1 style="margin: 0 0 0.5rem 0; font-size: 1.5rem; font-weight: 700;">Model Testing</h1>
    <p style="margin: 0; color: rgba(226, 232, 240, 0.75); font-size: 0.95rem;">
      Advanced prediction interface for testing different models, timestamps, and solar regions. Use this page to compare model performance, run historical predictions, and analyze specific active regions.
    </p>
  </div>

  <div class="card">
    <div class="section-title">Prediction confidence</div>
    {#if statusLoading}
      Loading guardrail status…
    {:else if statusError}
      {statusError}
    {:else if status}
      {#if status.connection.confidence}
        <p>
          <strong>Confidence:</strong> 
          {#if status.connection.confidence === "normal" || status.connection.confidence === "high"}
            <span style="color: #10b981; font-weight: 600;">{status.connection.confidence.toUpperCase()}</span>
            {#if !status.connection.guardrailActive}
              <span style="color: #10b981; margin-left: 0.5rem;">✓ All validations passing</span>
            {/if}
          {:else if status.connection.confidence === "low"}
            <span style="color: #f59e0b;">{status.connection.confidence.toUpperCase()}</span>
            {#if status.connection.guardrailReason}
              — <span style="color: #f59e0b;">{status.connection.guardrailReason}</span>
            {/if}
          {:else}
            {status.connection.confidence}
          {/if}
        </p>
      {:else}
        API did not publish confidence metadata. Ensure the API server is reachable.
      {/if}
    {/if}
  </div>

  <div class="grid two">
    <div class="card">
      <h2>Classification Model Testing</h2>
      <p>
        Test classification models with custom parameters. Predict the maximum solar flare class (None, C, M, or X) expected within the next 24 or 48 hours for a specific
        solar region at a given time. Compare different model types and time windows.
      </p>

      <form on:submit|preventDefault={() => runClassification(false)}>
        <label>
          Observation time
          <input type="datetime-local" bind:value={timestamp} max="9999-12-31T23:59" />
          <small>The timestamp for which to generate the prediction</small>
        </label>
        <label>
          Solar region number (optional)
          <input type="number" bind:value={regionNumber} placeholder="e.g., 3598" min="0" />
          <small>Leave empty to predict for all active regions</small>
        </label>

        <div class="grid two">
          <label>
            Prediction window
            <select bind:value={windowHours}>
              <option value="24">24 hours</option>
              <option value="48">48 hours</option>
            </select>
            <small>Time horizon for the prediction</small>
          </label>
          <label>
            Model type
            <select bind:value={modelType}>
              <option value="logistic">Logistic regression</option>
              <option value="gradient_boosting">Gradient boosting</option>
            </select>
            <small>Machine learning model to use</small>
          </label>
        </div>

        <div class="button-row">
          <button class="primary" type="submit" disabled={classificationLoading}>
            {#if classificationLoading}
              <LoadingSpinner size={16} color="#ffffff" />
              <span>Predicting…</span>
            {:else}
              Run prediction
            {/if}
          </button>
          <button
            class="secondary"
            type="button"
            on:click={() => runClassification(true)}
            disabled={classificationLoading}
          >
            {#if classificationLoading}
              <LoadingSpinner size={16} />
              <span>Force refresh & predict</span>
            {:else}
              Force refresh & predict
            {/if}
          </button>
        </div>
      </form>

      {#if classificationStatus}
        <div class={classificationStatus.includes("error") || classificationStatus.includes("Error") || classificationStatus.includes("failed") || classificationStatus.includes("unavailable") ? "warning" : ""} style="margin-top: 1rem; padding: 0.75rem; border-radius: 0.375rem; {classificationStatus.includes('error') || classificationStatus.includes('Error') || classificationStatus.includes('failed') || classificationStatus.includes('unavailable') ? 'border-color: #f87171; background: rgba(248, 113, 113, 0.1);' : 'border-color: #60a5fa; background: rgba(96, 165, 250, 0.1);'}">
          {classificationStatus}
        </div>
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

      {#if classificationExplanation}
        <div class="card" style="margin-top: 1rem;">
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.75rem;">
            <h3 style="margin: 0; font-size: 1rem; font-weight: 600;">Feature Importance (SHAP)</h3>
            <button
              class="secondary small"
              on:click={() => showExplanation = !showExplanation}
            >
              {showExplanation ? 'Hide' : 'Show'} Details
            </button>
          </div>
          {#if showExplanation}
            <ShapFeatureImportance
              explanation={classificationExplanation}
              maxFeatures={15}
              title="Top features driving prediction"
            />
            <div class="explanation-info" style="margin-top: 1rem; font-size: 0.85rem; color: rgba(226, 232, 240, 0.65);">
              <p style="margin: 0.5rem 0;">
                <strong style="color: #10b981;">Green bars</strong> = features that increased the probability of {classificationExplanation.predicted_class} class.
                <strong style="color: #ef4444;">Red bars</strong> = features that decreased it.
              </p>
              <p style="margin: 0.5rem 0;">
                Explainer type: <code style="background: rgba(30, 41, 59, 0.8); padding: 0.1rem 0.4rem; border-radius: 0.25rem;">{classificationExplanation.explainer_type}</code>
              </p>
            </div>
          {/if}
        </div>
      {/if}
    </div>

    <div class="card">
      <h2>Survival Model Testing (M-class)</h2>
      <p>
        Test survival analysis models with custom parameters. Predict when an M-class flare is likely to occur using time-to-event analysis. Shows probability distributions
        across different time horizons (0-168 hours) and a survival curve indicating the likelihood of flare occurrence
        over time. Compare Cox and Gradient Boosting survival models.
      </p>
      <div class="info-box" style="margin: 1rem 0; padding: 0.75rem; background: rgba(96, 165, 250, 0.1); border-left: 3px solid #60a5fa; border-radius: 0.25rem;">
        <strong>Model Performance:</strong> F1: 0.867, Precision: 93%, Recall: 81%
      </div>

      <form on:submit|preventDefault={() => runSurvival(false)}>
        <label>
          Observation time
          <input type="datetime-local" bind:value={timestamp} max="9999-12-31T23:59" />
          <small>The timestamp for which to generate the prediction</small>
        </label>
        <label>
          Solar region number (optional)
          <input type="number" bind:value={regionNumber} placeholder="e.g., 3598" min="0" />
          <small>Leave empty to predict for all active regions</small>
        </label>
        <label>
          Model type
          <select bind:value={survivalModel}>
            <option value="cox">Cox proportional hazards</option>
            <option value="gb">Gradient boosting survival</option>
          </select>
          <small>Survival analysis model to use</small>
        </label>
        <div class="button-row">
          <button class="primary" type="submit" disabled={survivalLoading}>
            {#if survivalLoading}
              <LoadingSpinner size={16} color="#ffffff" />
              <span>Predicting…</span>
            {:else}
              Run prediction
            {/if}
          </button>
          <button class="secondary" type="button" on:click={() => runSurvival(true)} disabled={survivalLoading}>
            {#if survivalLoading}
              <LoadingSpinner size={16} />
              <span>Force refresh & predict</span>
            {:else}
              Force refresh & predict
            {/if}
          </button>
        </div>
      </form>

      {#if survivalStatus}
        <div class={survivalStatus.includes("error") || survivalStatus.includes("Error") || survivalStatus.includes("failed") || survivalStatus.includes("unavailable") ? "warning" : ""} style="margin-top: 1rem; padding: 0.75rem; border-radius: 0.375rem; {survivalStatus.includes('error') || survivalStatus.includes('Error') || survivalStatus.includes('failed') || survivalStatus.includes('unavailable') ? 'border-color: #f87171; background: rgba(248, 113, 113, 0.1);' : 'border-color: #60a5fa; background: rgba(96, 165, 250, 0.1);'}">
          {survivalStatus}
        </div>
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
