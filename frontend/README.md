# Agnirodhak React Frontend

Modern React dashboard (Vite) for the Agnirodhak Uttarakhand forest fire early warning backend.

## Local development

```bash
cd frontend
npm install
npm run dev
```

By default the app expects the FastAPI backend at `http://localhost:8000`.

To point to a different backend (e.g. Render URL), set `VITE_API_BASE` when building:

```bash
VITE_API_BASE="https://your-backend.onrender.com" npm run build
```

Then deploy the generated `dist/` folder as a static site (e.g. Render static site).
