import path from "path";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      "/api": {
        target: process.env.VITE_DEV_API_TARGET ?? "http://127.0.0.1:8060",
        changeOrigin: true,
      },
    },
  },
  build: {
    // Emit the compiled bundle directly into the Python package so it ships as
    // package data and is resolved at runtime via importlib.resources.
    outDir: path.resolve(__dirname, "../spar_datadash/_frontend"),
    emptyOutDir: true,
    target: "es2022",
    sourcemap: false,
    reportCompressedSize: false,
  },
});
