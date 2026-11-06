# PrismRAW frontend

Static HTML, CSS and browser JavaScript, with no Python dependencies.

Vercel Root Directory: frontend. Framework: Other. Set PUBLIC_API_BASE_URL to the actual HTTPS Render origin. Build: node build.mjs. Output: dist. vercel.json defines these settings.

Local combined serving uses ../run_local.ps1 without a frontend build; static/config.js defaults to same-origin API access. Browser uploads and result URLs point directly to Render in the Vercel build.
