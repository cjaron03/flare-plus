<script>
  import { onDestroy, onMount } from "svelte";

  export let data = [];
  export let layout = {};
  export let config = {
    responsive: true,
    displayModeBar: false
  };

  let container;
  let Plotly = null;

  // Load Plotly dynamically - it's a UMD bundle
  const loadPlotly = async () => {
    if (Plotly) return Plotly;
    
    try {
      // Try to get Plotly from window (if loaded via script tag)
      if (typeof window !== 'undefined' && window.Plotly) {
        Plotly = window.Plotly;
        return Plotly;
      }
      
      // Dynamic import - UMD bundle attaches to window.Plotly
      await import("plotly.js-dist-min");
      
      // After import, Plotly should be on window
      if (typeof window !== 'undefined' && window.Plotly) {
        Plotly = window.Plotly;
      } else {
        // Fallback: try to access from module
        const plotlyModule = await import("plotly.js-dist-min");
        Plotly = plotlyModule.Plotly || plotlyModule.default || plotlyModule;
      }
      
      if (!Plotly || typeof Plotly.newPlot !== 'function') {
        console.error('[PlotlyChart] Plotly.newPlot not available, Plotly:', Plotly);
        return null;
      }
      
      return Plotly;
    } catch (err) {
      console.error('[PlotlyChart] Failed to load Plotly:', err);
      return null;
    }
  };

  const render = async () => {
    if (!container) return;
    
    if (!Plotly) {
      Plotly = await loadPlotly();
      if (!Plotly) return;
    }
    
    const plotData = Array.isArray(data) ? data : [];
    
    try {
      // Clear existing plot if any
      if (container.data) {
        Plotly.purge(container);
      }
      
      // Use newPlot to create/update the plot
      Plotly.newPlot(container, plotData, layout, config);
    } catch (err) {
      console.error('[PlotlyChart] Render error:', err);
    }
  };

  onDestroy(() => {
    if (container && Plotly) {
      try {
        Plotly.purge(container);
      } catch (err) {
        console.error('[PlotlyChart] Purge error:', err);
      }
    }
  });

  onMount(() => {
    render();
  });

  // Reactive statement to re-render when data changes
  let renderTimeout = null;
  $: if (container && Plotly && data) {
    // Clear any pending render
    if (renderTimeout) {
      clearTimeout(renderTimeout);
    }
    // Debounce rapid changes
    renderTimeout = setTimeout(() => {
      render();
      renderTimeout = null;
    }, 50);
  }
  
  // Cleanup on destroy
  onDestroy(() => {
    if (renderTimeout) {
      clearTimeout(renderTimeout);
    }
  });
</script>

<div class="plotly-chart" bind:this={container}></div>

<style>
  .plotly-chart {
    width: 100%;
    min-height: 320px;
    max-width: 100%;
    overflow: hidden;
  }
  
  :global(.plotly) {
    width: 100% !important;
    max-width: 100% !important;
  }
</style>
