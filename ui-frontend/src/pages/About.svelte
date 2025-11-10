<script>
  import { onMount } from "svelte";
  import { fetchAbout } from "../lib/api";

  let loading = true;
  let error = "";
  let about;

  onMount(async () => {
    loading = true;
    error = "";
    try {
      about = await fetchAbout();
    } catch (err) {
      error = err.message;
    } finally {
      loading = false;
    }
  });

  $: endpointList = about?.endpoints
    ? Object.entries(about.endpoints).filter(([_, url]) => Boolean(url))
    : [];
</script>

<div class="page">
  <div class="card">
    <h2>About flare+</h2>
    <p>
      flare+ is a research-grade solar flare prediction stack focused on short-term classification (24–48h) and
      time-to-event survival modelling. It ingests NOAA/SWPC data streams, engineers domain-specific features, and
      exposes predictions through the API and this UI.
    </p>
  </div>

  <div class="grid two">
    <div class="card">
      <h3>Limitations & disclaimers</h3>
      <ul>
        <li>This prototype relies on a limited training set — treat outputs as advisory only.</li>
        <li>Feature coverage varies by time range; occasional gaps are expected.</li>
        <li>Performance has not been benchmarked against official NOAA/SWPC forecasts.</li>
        <li>Availability depends on upstream NOAA endpoints and local infrastructure.</li>
      </ul>
    </div>

    <div class="card">
      <h3>Data sources</h3>
      {#if loading}
        Loading…
      {:else if error}
        {error}
      {:else if endpointList.length}
        <ul>
          {#each endpointList as [name, url]}
            <li>
              <strong>{name.replace("_", " ").toUpperCase()}:</strong>
              <a href={url} target="_blank" rel="noreferrer">{url}</a>
            </li>
          {/each}
        </ul>
      {:else}
        <p>Endpoint URLs not provided in config.yaml.</p>
      {/if}
      <ul>
        <li>
          <a href="https://www.swpc.noaa.gov/" target="_blank" rel="noreferrer">NOAA Space Weather Prediction Center</a>
        </li>
        <li>
          <a href="https://services.swpc.noaa.gov/json/goes/primary/xrays-7-day.json" target="_blank" rel="noreferrer"
            >GOES XRS Flux Data</a
          >
        </li>
        <li>
          <a href="https://services.swpc.noaa.gov/json/solar_regions.json" target="_blank" rel="noreferrer"
            >Solar Region Data</a
          >
        </li>
      </ul>
    </div>
  </div>

  <div class="grid two">
    <div class="card">
      <h3>Methodology</h3>
      <p>
        <strong>Classification:</strong> logistic regression and gradient boosting models predict the maximum flare class
        expected within 24 or 48 hours. Features include sunspot complexity metrics, x-ray flux trends, rolling
        statistics, and recency-weighted flare history.
      </p>
      <p>
        <strong>Survival analysis:</strong> Cox and gradient-boosting survival models output probability distributions
        for C-class flare timing across 0–168 hour buckets with hazard scores describing relative risk.
      </p>
    </div>
    <div class="card">
      <h3>Contact</h3>
      <p>
        Built by <strong>Jaron Cabral</strong>. See the repository README for roadmap details, operational notes, and the
        validation history.
      </p>
    </div>
  </div>
</div>
