import path from "node:path";

import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src")
    }
  },
  server: {
    port: 5173,
    proxy: {
      "/health": "http://localhost:7860",
      "/tasks": "http://localhost:7860",
      "/state": "http://localhost:7860",
      "/step": "http://localhost:7860",
      "/reset": "http://localhost:7860"
    }
  },
  build: {
    outDir: "dist",
    sourcemap: false
  }
});
