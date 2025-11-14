<script>
  import Router, { link, location } from "svelte-spa-router";
  import routes from "./routes";

  const navItems = [
    { href: "/", label: "Overview" },
    { href: "/predictions", label: "Model Testing" },
    { href: "/timeline", label: "Timeline" },
    { href: "/scenario", label: "Scenario" },
    { href: "/about", label: "About" },
    { href: "/admin", label: "Admin" },
    { href: "/login", label: "Login" }
  ];

  $: activePath = $location || "/";
  
  // preserve route on HMR updates
  if (import.meta.hot) {
    import.meta.hot.dispose(() => {
      // router will handle cleanup
    });
  }
</script>

<div class="app-shell">
  <header>
    <div class="nav-bar">
      <div class="brand">
        <span>flare+</span>
        <span>Solar flare prediction research dashboard</span>
      </div>
      <nav>
        {#each navItems as linkItem}
          <a
            href={linkItem.href}
            class:active={activePath === linkItem.href}
            use:link
            >{linkItem.label}</a
          >
        {/each}
      </nav>
    </div>
  </header>

  <main>
    <Router {routes} />
  </main>
</div>
