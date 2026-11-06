# Render backend and Vercel frontend

Commit and push the prepared changes first. Cloud deployment is not yet verified.

## Render

Create a Docker Web Service from Hey-Ritik/PrismRAW, branch main, Docker build context ./backend, Dockerfile ./backend/Dockerfile (repository-relative), Free plan, health path /api/health. Leave Docker command override blank.

Set PORT=7860, PRISMRAW_HOSTED=1, PRISMRAW_MAX_CROP=256, PRISMRAW_TORCH_THREADS=1 and PRISMRAW_API_ONLY=1. Set PRISMRAW_FRONTEND_ORIGINS to the exact Vercel production origin, for example https://your-project.vercel.app. No path or wildcard; multiple explicit origins may be comma-separated.

The Blueprint prompts for the frontend origin. Alternatively create Render manually with this variable empty, then add the actual Vercel domain and restart/redeploy. CORS is a browser permission, not API authentication. After Render is Live, copy its actual HTTPS origin. Confirm /api/health responds. The root returns API information instead of the studio. Local runs without PRISMRAW_API_ONLY continue serving the studio.

## Vercel

Import the same GitHub repository. Choose Framework Preset Other, Root Directory set to frontend. vercel.json sets build command `node build.mjs`, output `dist`, and no dependency installation. No Python Vercel Functions are needed.

Set PUBLIC_API_BASE_URL to your actual Render HTTPS origin, for example https://your-backend.onrender.com. No path, credentials, query, or fragment. This public URL is baked into the static build; changing it requires redeployment. Set it for Production and any Preview environment you intend to test.

Deploy Vercel, add its actual production origin to PRISMRAW_FRONTEND_ORIGINS on Render, and restart/redeploy Render. Preview domains are distinct origins and must be explicitly allowed. Use the production URL for the demo.

## Verification and limits

Open Vercel, process a sample, check LPIPS and images, download the ZIP, open the evaluation report, and upload a small RAW file. Browser API requests, uploads and result URLs should point directly to Render, avoiding Vercel's function payload limit. No Vercel upload proxy is used.

If Render is asleep, open its health URL, wait for a response and reload Vercel. CORS errors usually mean the exact frontend origin is missing on Render. An incorrect PUBLIC_API_BASE_URL requires a new Vercel build.

Separation does not materially reduce backend RAM. LPIPS remains enabled; the approximately 492 MiB local peak is not a Render memory guarantee. Keep FP16 unchecked initially. Temporary results disappear on Render restarts, redeploys or idle shutdowns. Actual builds and runtime operation must still be verified on both services.
