import { defineConfig } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";

export default defineConfig({
  plugins: [svelte({
    hot: !process.env.VITEST,
  })],
  server: {
    host: "0.0.0.0",
    port: 5173,
    hmr: {
      port: 5173,
      host: "localhost",
    },
    watch: {
      usePolling: true,
    },
    proxy: {
      "/ui/api": {
        target: process.env.UI_API_TARGET || "http://127.0.0.1:7860",
        changeOrigin: true,
        secure: false,
      },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
