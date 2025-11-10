<script>
  import { onMount } from "svelte";
  import { fetchAdminSession, fetchAdminPanel, runValidation } from "../lib/api";

  let session = null;
  let panelLoading = false;
  let panelError = "";
  let guardrailStatus = "";
  let validationHistory = [];
  let validationOutput = "";
  let validationRunning = false;

  const renderMarkdown = (text) =>
    (text || "")
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\n/g, "<br />");

  const loadPanel = async () => {
    panelLoading = true;
    panelError = "";
    try {
      const data = await fetchAdminPanel();
      guardrailStatus = data.guardrailStatus;
      validationHistory = data.validationHistory || [];
      panelError = data.error || "";
    } catch (err) {
      panelError = err.message;
    } finally {
      panelLoading = false;
    }
  };

  const refreshSession = async () => {
    try {
      session = await fetchAdminSession();
      if (session.hasAccess) {
        await loadPanel();
      }
    } catch (err) {
      panelError = err.message;
    }
  };

  const handleValidation = async () => {
    validationRunning = true;
    panelError = "";
    try {
      const result = await runValidation();
      validationOutput = result.output || result.message;
      guardrailStatus = result.guardrailStatus || guardrailStatus;
      validationHistory = result.validationHistory || validationHistory;
      if (!result.success && result.message) {
        panelError = result.message;
      }
    } catch (err) {
      panelError = err.message;
    } finally {
      validationRunning = false;
    }
  };

  onMount(refreshSession);
</script>

<div class="page">
  <div class="card">
    <h2>Admin console</h2>
    {#if !session}
      Checking admin session…
    {:else if !session.hasAccess}
      <p>
        Admin features are locked. {session.disabledReason || "Use the Login page to authenticate and unlock tools."}
      </p>
    {:else}
      <p>Admin tools are enabled for this session. Guardrail health and validation history are shown below.</p>
    {/if}
  </div>

  {#if session?.hasAccess}
    <div class="card">
      <div class="button-row">
        <button class="secondary" on:click={loadPanel} disabled={panelLoading}>
          {panelLoading ? "Refreshing…" : "Refresh status"}
        </button>
        <button class="primary" on:click={handleValidation} disabled={validationRunning}>
          {validationRunning ? "Running…" : "Run system validation"}
        </button>
      </div>
      {#if panelError}
        <p style="color: #fecaca;">{panelError}</p>
      {/if}
      <div style="margin-top: 1rem;">
        <h3>Guardrail status</h3>
        <div class="warning" style="background: rgba(37, 99, 235, 0.12);">
          {@html renderMarkdown(guardrailStatus)}
        </div>
      </div>
      <div style="margin-top: 1.5rem;">
        <h3>Recent validation runs</h3>
        {#if validationHistory.length}
          <table class="table">
            <thead>
              <tr>
                <th>Run time</th>
                <th>Status</th>
                <th>Guardrail</th>
                <th>Reason</th>
                <th>Initiated by</th>
              </tr>
            </thead>
            <tbody>
              {#each validationHistory as row}
                <tr>
                  <td>{row.runTime}</td>
                  <td>{row.status}</td>
                  <td>{row.guardrail}</td>
                  <td>{row.reason}</td>
                  <td>{row.initiatedBy}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        {:else}
          <p>No validation runs recorded.</p>
        {/if}
      </div>
      {#if validationOutput}
        <div style="margin-top: 1.5rem;">
          <h3>Validation output</h3>
          <pre>{validationOutput}</pre>
        </div>
      {/if}
    </div>
  {/if}
</div>
