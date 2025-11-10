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
    validationOutput = "";
    try {
      const result = await runValidation();
      
      // always update guardrail status and history
      guardrailStatus = result.guardrailStatus || guardrailStatus;
      validationHistory = result.validationHistory || validationHistory;
      
      // prioritize validationOutput, then output, then message
      if (result.validationOutput) {
        validationOutput = result.validationOutput;
        // if validation failed but we have output, don't show error message
        if (!result.success) {
          panelError = ""; // clear error since we have detailed output
        }
      } else if (result.output) {
        validationOutput = result.output;
        if (!result.success) {
          panelError = "";
        }
      } else if (result.message) {
        // if no detailed output, show message as both error and output
        if (!result.success) {
          validationOutput = result.message;
          panelError = "";
        } else {
          validationOutput = result.message;
        }
      }
      
      // only show error if we have no output at all
      if (!result.success && !validationOutput && result.message) {
        panelError = result.message;
      }
    } catch (err) {
      // extract error details from response if available
      let errorMsg = err.message;
      let errorOutput = err.message;
      
      if (err.data) {
        // check if there's validation output in the error response
        if (err.data.validationOutput) {
          errorOutput = err.data.validationOutput;
          errorMsg = "";
        } else if (err.data.message) {
          errorOutput = err.data.message;
          errorMsg = err.data.message;
        }
      }
      
      panelError = errorMsg;
      validationOutput = errorOutput;
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
      {#if panelError && !validationOutput}
        <div class="warning" style="border-color: #f87171; background: rgba(248, 113, 113, 0.1); margin-bottom: 1rem;">
          <strong>Error:</strong> {panelError}
        </div>
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
