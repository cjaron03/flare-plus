<script>
  import { onDestroy, onMount } from "svelte";
  import Plotly from "plotly.js-dist-min";

  export let data = [];
  export let layout = {};
  export let config = {
    responsive: true,
    displayModeBar: false
  };

  let container;

  const render = () => {
    if (!container) return;
    const plotData = Array.isArray(data) ? data : [];
    Plotly.react(container, plotData, layout, config);
  };

  onDestroy(() => {
    if (container) {
      Plotly.purge(container);
    }
  });

  onMount(() => {
    render();
  });

  $: container, data, layout, config;
  $: if (container) {
    render();
  }
</script>

<div class="plotly-chart" bind:this={container}></div>

<style>
  .plotly-chart {
    width: 100%;
    min-height: 320px;
  }
</style>
