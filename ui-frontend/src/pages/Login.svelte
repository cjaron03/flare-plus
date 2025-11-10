<script>
  import { onMount } from "svelte";
  import { fetchAdminSession, loginAdmin, logoutAdmin } from "../lib/api";

  let session = null;
  let username = "";
  let password = "";
  let message = "";
  let loading = false;

  const loadSession = async () => {
    try {
      session = await fetchAdminSession();
    } catch (err) {
      message = err.message;
    }
  };

  const handleLogin = async () => {
    loading = true;
    message = "";
    try {
      const result = await loginAdmin({ username, password });
      message = result.message;
      session = result.session;
    } catch (err) {
      message = err.message;
    } finally {
      loading = false;
    }
  };

  const handleLogout = async () => {
    loading = true;
    message = "";
    try {
      const result = await logoutAdmin();
      message = result.message;
      session = result.session;
    } catch (err) {
      message = err.message;
    } finally {
      loading = false;
    }
  };

  onMount(loadSession);
</script>

<div class="page">
  <div class="card">
    <h2>Admin login</h2>
    <p>Authenticate to unlock the ingestion guardrail dashboard and validation tools.</p>

    <form on:submit|preventDefault={handleLogin}>
      <label>
        Username
        <input type="text" bind:value={username} placeholder="Username" autocomplete="username" />
      </label>
      <label>
        Password
        <input type="password" bind:value={password} placeholder="Password" autocomplete="current-password" />
      </label>
      <div class="button-row">
        <button class="primary" type="submit" disabled={loading}>
          {loading ? "Signing in…" : "Sign in"}
        </button>
        <button class="secondary" type="button" on:click={handleLogout} disabled={loading}>
          Sign out
        </button>
      </div>
    </form>

    {#if session}
      <p style="margin-top: 1rem;">
        <strong>Admin access:</strong> {session.indicator}
      </p>
      {#if session.disabledReason}
        <p>{session.disabledReason}</p>
      {/if}
    {/if}

    {#if message}
      <p style="margin-top: 0.5rem;">{message}</p>
    {/if}
  </div>
</div>
