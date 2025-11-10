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

  onMount(() => {
    render();
    onDestroy(() => {
      if (container) {
        Plotly.purge(container);
      }
    });
  });

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
