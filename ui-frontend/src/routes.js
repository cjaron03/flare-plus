import Dashboard from "./pages/Dashboard.svelte";
import Predictions from "./pages/Predictions.svelte";
import Timeline from "./pages/Timeline.svelte";
import Scenario from "./pages/Scenario.svelte";
import About from "./pages/About.svelte";
import Admin from "./pages/Admin.svelte";
import Login from "./pages/Login.svelte";

export default {
  "/": Dashboard,
  "/predictions": Predictions,
  "/timeline": Timeline,
  "/scenario": Scenario,
  "/about": About,
  "/admin": Admin,
  "/login": Login,
  "*": Dashboard,
};
