<script>
  import { onMount } from "svelte";
  import { fetchStatus, triggerIngestion } from "../lib/api";
  import StatusPill from "../components/StatusPill.svelte";

  let loading = true;
  let status;
  let error = "";
  let ingestMessage = "";
  let ingestSummary = "";
  let ingestLoading = false;

  const loadStatus = async () => {
    loading = true;
    error = "";
    try {
      status = await fetchStatus();
    } catch (err) {
      error = err.message;
    } finally {
      loading = false;
    }
  };

  const handleRefresh = async () => {
    ingestLoading = true;
    ingestMessage = "";
    ingestSummary = "";
    try {
      const result = await triggerIngestion({ useCache: true });
      ingestMessage = result.message;
      ingestSummary = result.summary ?? "";
      // reload full status to update all data freshness and connection info
      await loadStatus();
    } catch (err) {
      ingestMessage = err.message;
    } finally {
      ingestLoading = false;
    }
  };

  onMount(loadStatus);

  $: freshnessEntries = status?.dataFreshness
    ? Object.entries(status.dataFreshness)
    : [];
</script>

<div class="page">
  <section class="hero">
    <h1 class="hero-title">Solar Flare Prediction Dashboard</h1>
    <p class="hero-subtitle">
      Live monitoring of classification and survival models, NOAA data freshness, and ingestion activity. Built for
      research workflows — treat insights as advisory only.
    </p>
    <div class="warning">
      <strong>Prototype limitation:</strong> {status?.limitation ||
        "This build relies on limited training data. Predictions may be noisy until more flare events are ingested."}
    </div>
    {#if error}
      <div class="warning" style="border-color: #f87171;">
        {error}
      </div>
    {/if}
  </section>

  {#if loading}
    <div class="card">Loading status…</div>
  {:else if status}
    <div class="card">
      <div class="section-title">Ingestion & Refresh</div>
      <p>
        Trigger NOAA/SWPC ingestion via the API. The UI enforces a short cooldown to avoid hammering the pipeline.
      </p>
      <div class="button-row">
        <button class="primary" on:click={handleRefresh} disabled={ingestLoading}>
          {ingestLoading ? "Refreshing…" : "Refresh Data & Status"}
        </button>
        <button class="secondary" on:click={loadStatus} disabled={loading}>
          {loading ? "Loading…" : "Reload Snapshot"}
        </button>
      </div>
      {#if ingestMessage}
        <p style="margin-top: 1rem;">
          {ingestMessage}
        </p>
      {/if}
      {#if ingestSummary}
        <pre>{ingestSummary}</pre>
      {/if}
      {#if status.lastRefresh}
        <small>Last refresh marker: {new Date(status.lastRefresh).toLocaleString()}</small>
      {/if}
    </div>

    <div class="grid two">
      <div class="card">
        <div class="section-title">Connection & Models</div>
        <div class="status-grid">
          <div class="item">
            <div class="label">Mode</div>
            <div class="value">{status.connection.mode?.toUpperCase()}</div>
          </div>
          <div class="item">
            <div class="label">API URL</div>
            <div class="value">{status.connection.apiUrl ?? "N/A"}</div>
          </div>
          <div class="item">
            <div class="label">Models</div>
            <div class="value">
              {#if status.connection.modelsLoaded?.length}
                {status.connection.modelsLoaded.join(", ")}
              {:else}
                —
              {/if}
            </div>
          </div>
          <div class="item">
            <div class="label">Confidence</div>
            <div class="value">
              {status.connection.confidence ?? "Unavailable"}
              {#if status.connection.guardrailActive}
                <span class="badge red" style="margin-left: 0.5rem;">Guardrail active</span>
              {/if}
            </div>
            {#if status.connection.guardrailReason}
              <small>{status.connection.guardrailReason}</small>
            {/if}
          </div>
          <div class="item">
            <div class="label">Last validation</div>
            <div class="value">{status.connection.lastValidation ?? "Unknown"}</div>
          </div>
          <div class="item">
            <div class="label">Admin access</div>
            <div class="value">{status.connection.adminIndicator}</div>
            {#if status.connection.adminDisabledReason}
              <small>{status.connection.adminDisabledReason}</small>
            {/if}
          </div>
        </div>
      </div>

      <div class="card">
        <div class="section-title">Data Freshness</div>
        <div class="status-grid">
          {#each freshnessEntries as [key, info]}
            <div class="item">
              <div class="label">{info.label}</div>
              <div class="value">
                {#if info.hoursAgo !== null && info.hoursAgo !== undefined}
                  {info.hoursAgo.toFixed(1)}h ago
                {:else}
                  No data
                {/if}
              </div>
              <StatusPill
                label={`${info.count?.toLocaleString?.() ?? (info.count ?? 0)} records`}
                status={info.status}
              />
            </div>
          {/each}
        </div>
      </div>
    </div>
  {/if}
</div>
