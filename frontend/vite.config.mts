import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";

// Use env var for backend URL at build time, default to local FastAPI
const backendUrl = process.env.VITE_API_BASE || "http://localhost:8000";

export default defineConfig({
  plugins: [react()],
  define: {
    __API_BASE__: JSON.stringify(backendUrl),
  },
  server: {
    port: 5173,
    strictPort: true,
  },
});
