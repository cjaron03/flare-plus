const API_BASE = import.meta.env.VITE_UI_API_URL ?? "/ui/api";

async function request(path, options = {}) {
  // add timeout to prevent hanging requests
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 60000); // 60 second timeout
  
  try {
    const response = await fetch(`${API_BASE}${path}`, {
      headers: {
        "Content-Type": "application/json",
        ...(options.headers || {})
      },
      signal: controller.signal,
      ...options
    });

    clearTimeout(timeoutId);
    const data = await response.json().catch(() => ({}));
    if (!response.ok) {
      // for validation errors, include the full response data
      const error = new Error(data.message || `Request to ${path} failed`);
      error.data = data;
      error.status = response.status;
      throw error;
    }
    return data;
  } catch (err) {
    clearTimeout(timeoutId);
    if (err.name === 'AbortError') {
      const error = new Error('Request timed out after 60 seconds');
      error.data = { message: 'Request timed out' };
      error.status = 408;
      throw error;
    }
    // preserve error data if it exists
    if (err.data) {
      throw err;
    }
    throw err;
  }
}

export const fetchStatus = () => request("/status");
export const triggerIngestion = (payload) =>
  request("/ingest", { method: "POST", body: JSON.stringify(payload ?? {}) });
export const predictClassification = (payload) =>
  request("/predict/classification", { method: "POST", body: JSON.stringify(payload) });
export const predictSurvival = (payload) =>
  request("/predict/survival", { method: "POST", body: JSON.stringify(payload) });
export const fetchTimeline = (payload) =>
  request("/timeline", { method: "POST", body: JSON.stringify(payload) });
export const fetchAbout = () => request("/about");

// admin
export const fetchAdminSession = () => request("/admin/session");
export const loginAdmin = (payload) =>
  request("/admin/login", { method: "POST", body: JSON.stringify(payload) });
export const logoutAdmin = () =>
  request("/admin/logout", { method: "POST" });
export const fetchAdminPanel = () => request("/admin/panel");
export const runValidation = async () => {
  try {
    return await request("/admin/validate", { method: "POST" });
  } catch (err) {
    // if the error has data with validation output, return it as a failed result
    // instead of throwing, so the UI can display the output
    if (err.data && (err.data.validationOutput || err.data.message)) {
      return {
        success: false,
        message: err.data.message || err.message,
        validationOutput: err.data.validationOutput,
        guardrailStatus: err.data.guardrailStatus,
        validationHistory: err.data.validationHistory || [],
      };
    }
    throw err;
  }
};
