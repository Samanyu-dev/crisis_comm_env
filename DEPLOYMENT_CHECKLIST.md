# Production Deployment Checklist for Hugging Face Spaces

## Pre-Deployment Verification

### 1. Frontend Build
- [x] `npm run build` completes successfully
- [x] `frontend/dist/index.html` exists
- [x] `frontend/dist/assets/` directory contains all chunks
- [x] All asset paths use relative paths (e.g., `./assets/...`)
- [x] No hardcoded URLs in built files

### 2. Router Configuration
- [x] Using `HashRouter` instead of `BrowserRouter`
- [x] Routes use hash-based navigation (`/#/dashboard`)
- [x] `vite.config.ts` has `base: "./"`
- [x] All routes render correctly with hash navigation

### 3. API Configuration
- [x] API service uses relative URLs (no localhost)
- [x] `VITE_API_BASE_URL` env var properly configured
- [x] Fallback to empty string for relative API calls
- [x] All endpoints use `/health`, `/tasks`, `/reset`, `/step`, `/state`

### 4. FastAPI Configuration
- [x] `/assets` static files mounted correctly
- [x] `/` route serves `index.html` if built
- [x] `/{full_path:path}` SPA fallback implemented
- [x] Proper 404 handling for API endpoints
- [x] Frontend dist files copied in Docker build

### 5. Docker Configuration
- [x] Multi-stage build with frontend builder
- [x] Frontend built in first stage
- [x] `frontend/dist` copied to runtime image
- [x] Python dependencies installed
- [x] `uvicorn` runs on `0.0.0.0:7860`
- [x] `EXPOSE 7860` directive present

### 6. Git Cleanup
- [x] `node_modules/` removed from git tracking
- [x] `frontend/node_modules/` removed from git tracking
- [x] `frontend/dist/` added to `.gitignore`
- [x] Only source files committed

### 7. Error Handling
- [x] `ErrorBoundary` component wraps app
- [x] Deployment diagnostics run on startup
- [x] API errors logged to console
- [x] Failed requests show user-friendly messages
- [x] Refresh mechanism provided

### 8. Production Readiness
- [x] Suspense boundaries for lazy routes
- [x] Loading states during initial render
- [x] Console logging for debugging
- [x] No hydration issues
- [x] All dynamic imports work

## Local Testing Steps

### Test Production Build Locally
```bash
# Build frontend
cd frontend && npm run build && cd ..

# Start Python server
cd server
source ../venv/bin/activate
python app.py
```

Then visit `http://localhost:7860` and verify:
1. Frontend renders (no blank screen)
2. All routes work with hash navigation
3. API calls succeed
4. Refresh works without breaking routing
5. No errors in browser console

### Test Docker Locally
```bash
docker build -t crisis-comm:test .
docker run -p 7860:7860 crisis-comm:test
```

Visit `http://localhost:7860` and verify same as above.

## Hugging Face Spaces Deployment

### Environment Setup
1. Create `.env` file in root with any needed vars
2. Ensure `.gitignore` excludes `node_modules` and build artifacts
3. Verify Dockerfile is at project root

### Repository Structure
```
.
├── frontend/
│   ├── src/          (source code)
│   ├── dist/         (built files - NOT in git)
│   ├── package.json
│   └── vite.config.ts
├── server/
│   ├── app.py        (FastAPI server)
│   ├── requirements.txt
│   └── ...
├── Dockerfile        (multi-stage build)
└── .gitignore       (includes frontend/dist, node_modules)
```

### Deploy Steps
1. Push code to GitHub (with `.gitignore` properly configured)
2. Create Hugging Face Space linked to GitHub repo
3. Select Docker runtime
4. Dockerfile path: `./Dockerfile`
5. Space runtime: Docker
6. Wait for build to complete
7. Access Space URL (hash-based routing will work)

## Troubleshooting

### Blank Screen on HF Spaces
**Check browser console for errors:**
```javascript
// Browser DevTools Console
// Should see deployment diagnostics logs
[ISO Timestamp] [Crisis-Comm] Starting deployment diagnostics...
```

**Solutions:**
- Verify `HashRouter` is used (check main.tsx)
- Verify `base: "./"` in vite.config.ts
- Verify `frontend/dist` exists with index.html
- Verify FastAPI serves static assets at `/assets`
- Check FastAPI logs for 404 errors

### API Calls Failing
**Check in browser console:**
```javascript
// Should show relative API URLs
[ISO Timestamp] [Crisis-Comm] API GET /health: 200
```

**Solutions:**
- Verify API base URL is empty string (relative)
- Verify backend is responding to API endpoints
- Check CORS if making cross-origin calls
- Verify endpoint paths match exactly

### Routes Not Working
**Check in browser console:**
```javascript
// Should show hash-based navigation
Router diagnostics: {
  usingHashRouter: true,
  currentPath: "#/dashboard"
}
```

**Solutions:**
- Verify HashRouter is used (not BrowserRouter)
- Check route paths match exactly
- Verify SPA fallback route in FastAPI

### Assets Not Loading
**Check in browser DevTools Network tab:**
- Should see `./assets/...` requests
- Should all return 200 status
- Check FastAPI logs for `/assets` mount

**Solutions:**
- Rebuild frontend: `npm run build`
- Verify `/assets` static mount in app.py
- Check frontend/dist/assets directory exists

## Monitoring in Production

### View Application Logs
In Hugging Face Space settings, check "App logs" to see:
1. FastAPI startup messages
2. API request logs
3. Error logs

### Browser Console Debugging
Open DevTools → Console to see:
1. Deployment diagnostics output
2. API request logs
3. Error messages
4. Route navigation

## Performance Optimizations

- [x] Build assets with `npm run build`
- [x] CSS minified automatically by Vite
- [x] JS minified automatically by Vite
- [x] Lazy-loaded pages reduce initial bundle
- [x] Relative asset paths avoid CDN lookups
- [x] Hash routing avoids server redirects

## Security Considerations

- ✓ No sensitive data in frontend code
- ✓ API calls use relative URLs
- ✓ CORS configured on backend if needed
- ✓ No environment secrets in frontend env vars
- ✓ Error messages don't expose sensitive info
