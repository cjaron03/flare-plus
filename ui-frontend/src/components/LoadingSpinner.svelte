<script>
  export let size = 20;
  export let color = "currentColor";
</script>

<div class="spinner" style="width: {size}px; height: {size}px;">
  <div class="spinner-inner" style="border-top-color: {color}; border-right-color: {color};"></div>
</div>

<style>
  .spinner {
    display: inline-block;
    vertical-align: middle;
    flex-shrink: 0;
    position: relative;
  }

  .spinner-inner {
    box-sizing: border-box;
    width: 100%;
    height: 100%;
    border: 2px solid rgba(255, 255, 255, 0.2);
    border-top-color: currentColor;
    border-right-color: currentColor;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
  }

  @keyframes spin {
    0% {
      transform: rotate(0deg);
    }
    100% {
      transform: rotate(360deg);
    }
  }
</style>

