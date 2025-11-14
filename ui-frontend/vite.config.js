import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  plugins: [svelte({
    hot: !process.env.VITEST,
    compilerOptions: {
      dev: process.env.NODE_ENV !== "production",
    },
  })],
  server: {
    host: "0.0.0.0",
    port: 5173,
    strictPort: false,
    hmr: {
      port: 5173,
      clientPort: 5173,
      protocol: "ws",
    },
    watch: {
      usePolling: true,
      interval: 1000,
      ignored: ["**/node_modules/**", "**/dist/**"],
    },
    proxy: {
      "/ui/api": {
        target: process.env.UI_API_TARGET || "http://127.0.0.1:7860",
        changeOrigin: true,
        secure: false,
        ws: false,
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  optimizeDeps: {
    exclude: ["plotly.js-dist-min"],
  },
  resolve: {
    alias: {
      "plotly.js-dist-min": "plotly.js-dist-min/plotly.min.js",
    },
  },
});
