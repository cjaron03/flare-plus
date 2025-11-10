const API_BASE = import.meta.env.VITE_UI_API_URL ?? "/ui/api";

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {})
    },
    ...options
  });

  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(data.message || `Request to ${path} failed`);
    error.data = data;
    error.status = response.status;
    throw error;
  }
  return data;
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
export const runValidation = () =>
  request("/admin/validate", { method: "POST" });
