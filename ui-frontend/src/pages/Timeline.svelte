<script>
  import { onMount } from "svelte";
  import PlotlyChart from "../components/PlotlyChart.svelte";
  import { fetchTimeline } from "../lib/api";

  const now = new Date();
  const thirtyDaysAgo = new Date(now.getTime() - 1000 * 60 * 60 * 24 * 30);

  const formatInputValue = (date) => {
    const local = new Date(date.getTime() - date.getTimezoneOffset() * 60000);
    return local.toISOString().slice(0, 16);
  };

  const toISO = (value) => {
    if (!value) return null;
    const date = new Date(value);
    return Number.isNaN(date.getTime()) ? null : date.toISOString();
  };

  let start = formatInputValue(thirtyDaysAgo);
  let end = formatInputValue(now);
  let minClass = "B";
  let regionNumber = "";

  let status = "";
  let loading = false;
  let events = [];
  let chartData = null;

  const buildChart = (items) => {
    if (!items.length) {
      chartData = [
        {
          x: [0],
          y: ["No events"],
          mode: "text",
          text: ["No flares found in range"],
        }
      ];
      return;
    }

    const classes = ["B", "C", "M", "X"];
    const colors = {
      B: "#fbbf24",
      C: "#fb923c",
      M: "#f97316",
      X: "#fb7185"
    };

    chartData = classes
      .map((cls) => {
        const filtered = items.filter((event) => event.classCategory === cls);
        if (!filtered.length) return null;
        return {
          x: filtered.map((event) => event.peakTime ?? event.startTime),
          y: filtered.map(() => cls),
          mode: "markers",
          name: cls,
          marker: {
            size: 12,
            color: colors[cls],
            line: { color: "#0f172a", width: 1.5 }
          },
          text: filtered.map(
            (event) =>
              `Class: ${event.flareClass || "?"}<br>Region: ${event.region || "?"}<br>Peak: ${
                event.peakTime ? new Date(event.peakTime).toLocaleString() : "N/A"
              }`
          ),
          hovertemplate: "%{text}<extra></extra>"
        };
      })
      .filter(Boolean);
  };

  const runQuery = async (forceRefresh = false) => {
    loading = true;
    status = "";
    try {
      const payload = {
        start: toISO(start),
        end: toISO(end),
        minClass,
        regionNumber: regionNumber ? Number(regionNumber) : null,
        forceRefresh
      };
      const result = await fetchTimeline(payload);
      events = result.events;
      status = result.message;
      buildChart(events);
    } catch (err) {
      status = err.message;
      events = [];
      chartData = null;
    } finally {
      loading = false;
    }
  };

  onMount(() => {
    runQuery(false);
  });
</script>

<div class="page">
  <div class="card">
    <h2>Historical flare timeline</h2>
    <p>
      Query stored flare events with optional filters. Use the force refresh button to bypass the 10 minute throttle if
      you need an immediate update.
    </p>
    <form on:submit|preventDefault={() => runQuery(false)}>
      <div class="grid two">
        <label>
          Start date
          <input type="datetime-local" bind:value={start} max={end} />
        </label>
        <label>
          End date
          <input type="datetime-local" bind:value={end} min={start} />
        </label>
      </div>

      <div class="grid two">
        <label>
          Minimum class
          <select bind:value={minClass}>
            <option value="B">B</option>
            <option value="C">C</option>
            <option value="M">M</option>
            <option value="X">X</option>
          </select>
        </label>
        <label>
          Region number (optional)
          <input type="number" bind:value={regionNumber} placeholder="e.g., 3598" min="0" />
        </label>
      </div>

      <div class="button-row">
        <button type="submit" class="primary" disabled={loading}>
          {loading ? "Updating…" : "Refresh timeline"}
        </button>
        <button type="button" class="secondary" on:click={() => runQuery(true)} disabled={loading}>
          Force refresh
        </button>
      </div>
    </form>

    {#if status}
      <p style="margin-top: 1rem;">{status}</p>
    {/if}
  </div>

  <div class="card chart-card">
    {#if chartData}
      <PlotlyChart
        data={chartData}
        layout={{
          title: "Flare events",
          xaxis: { title: "Time" },
          yaxis: { title: "Class", autorange: "reversed" },
          hovermode: "closest"
        }}
      />
    {:else}
      <p>No data to display yet.</p>
    {/if}
  </div>
</div>
