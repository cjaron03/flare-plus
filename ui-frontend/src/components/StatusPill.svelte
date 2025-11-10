<script>
  export let label = "";
  export let status = "gray";

  const colors = {
    fresh: "green",
    stale: "yellow",
    outdated: "red",
    gray: "gray"
  };

  const classes = {
    green: "badge green",
    yellow: "badge yellow",
    red: "badge red",
    gray: "badge gray"
  };

  $: variant = colors[status] ?? "gray";
</script>

<span class={classes[variant]}>{label}</span>
