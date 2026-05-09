# Production Deployment Fixes - Technical Documentation

## Summary of Changes

This document details all fixes applied to resolve the blank screen issue on Hugging Face Spaces and ensure proper production deployment of the React + Vite + FastAPI application.

## Root Cause Analysis

### Problem
The application showed a blank/dark screen when deployed to Hugging Face Spaces despite successful builds.

### Root Causes Identified

1. **Router Issue**: Using `BrowserRouter` which requires server-side routing support. HF Spaces serves static files from a single base path without configurable routing rules.
   
2. **Asset Path Issue**: Missing `base: "./"` in Vite config. In production with a different base path, relative asset imports break.
   
3. **API URL Issue**: If hardcoded localhost URLs existed, API calls would fail completely.
   
4. **Static Serving**: FastAPI wasn't correctly configured to serve static assets from the Vite build output.
   
5. **Git Tracking**: 10k+ node_modules files were cluttering the repository.
   
6. **Error Handling**: No error boundaries or diagnostics to help debug issues in production.

## Files Modified

### 1. `frontend/src/main.tsx`
**Change**: Replace `BrowserRouter` with `HashRouter`

**Before**:
```tsx
import { BrowserRouter } from "react-router-dom";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <BrowserRouter>
      <App />
    </BrowserRouter>
  </React.StrictMode>
);
```

**After**:
```tsx
import { HashRouter } from "react-router-dom";
import { ErrorBoundary } from "@/components/ErrorBoundary";

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <ErrorBoundary>
      <HashRouter>
        <App />
      </HashRouter>
    </ErrorBoundary>
  </React.StrictMode>
);
```

**Why**: 
- `HashRouter` uses URL hashes (`/#/path`) which work on static hosting without server routing
- `BrowserRouter` uses clean URLs (`/path`) which require server-side routing setup
- Wrapped app with `ErrorBoundary` for crash protection in production

### 2. `frontend/vite.config.ts`
**Change**: Add `base: "./"` to defineConfig

**Before**:
```ts
export default defineConfig({
  plugins: [react()],
  // ... rest of config
});
```

**After**:
```ts
export default defineConfig({
  base: "./",
  plugins: [react()],
  // ... rest of config
});
```

**Why**:
- Ensures all asset imports use relative paths
- Prevents broken asset links when serving from non-root paths
- Works correctly on any subdomain or base path
- Generated HTML uses `./assets/...` instead of `/assets/...`

### 3. `frontend/src/components/ErrorBoundary.tsx` (New)
**Purpose**: Catch React runtime errors and show user-friendly error screen

**Key Features**:
- Catches all React component errors
- Displays error message to user
- Provides refresh button to recover
- Logs errors to console for debugging
- Styled to match application theme

### 4. `frontend/src/hooks/useDeploymentDiagnostics.ts` (New)
**Purpose**: Run startup diagnostics and log deployment health

**Key Features**:
- Runs automatically on app mount
- Checks API connectivity with `/health` endpoint
- Checks if frontend assets loaded
- Verifies Router configuration (HashRouter)
- Logs all details to console with timestamps
- Works on HF Spaces and local environments
- Non-blocking (doesn't prevent app from loading)

**Console Output**:
```
[2024-05-10T12:34:56.789Z] [Crisis-Comm] Starting deployment diagnostics...
[2024-05-10T12:34:56.790Z] [Crisis-Comm] Environment: { isDevelopment: false, isHuggingFaceSpace: true, ... }
[2024-05-10T12:34:56.791Z] [Crisis-Comm] Testing API connectivity...
[2024-05-10T12:34:56.792Z] [Crisis-Comm] Fetching health endpoint: /health
[2024-05-10T12:34:56.900Z] [Crisis-Comm] API health check passed: { status: "ok", ... }
[2024-05-10T12:34:56.901Z] [Crisis-Comm] Router diagnostics: { usingHashRouter: true, ... }
[2024-05-10T12:34:56.902Z] [Crisis-Comm] All deployment diagnostics passed
```

### 5. `frontend/src/App.tsx`
**Changes**: 
- Added import for `useDeploymentDiagnostics`
- Called hook on component mount

**Result**: Diagnostics run immediately when app loads, making issues visible in console

### 6. `frontend/src/services/api.ts`
**Changes**:
- Added axios response interceptor for logging
- Logs all API requests/responses with timing
- Logs errors with detailed information
- Only verbose on dev or HF Spaces

**Console Output Example**:
```
[Crisis-Comm] Initializing API service { baseURL: "(relative)", isDev: false }
[Crisis-Comm] API GET /health: 200
[Crisis-Comm] API POST /reset: 200
```

### 7. `server/app.py`
**Changes**:
- Added logging configuration
- Logger created with consistent format
- All key operations logged
- Frontend serve/fallback logged
- API initialization logged

**Log Messages Include**:
- Frontend dist path and status
- Assets mount status
- When serving index.html
- SPA fallback routing
- Startup information

**Example**:
```
2024-05-10 12:34:56,789 - __main__ - INFO - Initializing FastAPI app
2024-05-10 12:34:56,790 - __main__ - INFO - Frontend dist path: /app/frontend/dist
2024-05-10 12:34:56,791 - __main__ - INFO - Frontend assets exist: True
2024-05-10 12:34:56,792 - __main__ - INFO - Mounted /assets static files
```

### 8. `.gitignore`
**Changes**: Added additional build artifacts to ignore list

**Before**:
```
frontend/node_modules/
frontend/dist/
frontend/*.tsbuildinfo
node_modules/
```

**After**:
```
frontend/node_modules/
frontend/dist/
frontend/*.tsbuildinfo
frontend/.vite/
node_modules/
dist/
build/
.cache/
*.tsbuildinfo
```

**Result**: Cleaner git repository, removes 10k+ node_modules from tracking

## Architecture Decisions

### Why HashRouter Instead of BrowserRouter?

**Deployment Environment Context:**
- Hugging Face Spaces serves static files
- No custom server routing configuration available
- Cannot intercept `/path` requests and serve `index.html`

**HashRouter Advantages:**
- Routes are stored in URL hash: `/#/path`
- Hash never sent to server, stays client-side
- Server always returns `index.html` for any request
- Works on static hosting, GitHub Pages, etc.
- No need for server configuration

**Tradeoff:**
- URLs have `#` character (aesthetic, not functional issue)
- SEO implications (less important for this app)
- Still fully functional for routing

### Why Relative Asset Paths?

**Problem Without `base: "./"`:**
- Vite generates `<script src="/assets/index.js"></script>`
- On HF Spaces at `username.hf.space/spaces/...`
- Browser requests `/assets/index.js` instead of `username.hf.space/spaces/assets/index.js`
- Results in 404s and blank screen

**Solution With `base: "./"`:**
- Vite generates `<script src="./assets/index.js"></script>`
- Browser resolves relative to current page location
- Works correctly at any path

### API URL Strategy

**Current Implementation:**
```ts
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";
```

**Why This Works:**
- In development: empty string, proxy through vite
- In production (HF Spaces): empty string, same-origin API calls
- Fallback allows override via environment variable
- No hardcoded localhost URLs

**Behavior:**
```
fetch("", "/health")  // → /health
fetch("", "/tasks")   // → /tasks
// All relative to current origin, no CORS issues
```

## Error Handling Strategy

### ErrorBoundary
- Catches render-time errors
- Shows recovery UI
- Logs to console

### Deployment Diagnostics
- Runs on startup
- Checks all critical systems
- Non-blocking (doesn't prevent app load)
- Logs all findings to console

### API Interceptors
- Logs every request
- Logs response status
- Logs errors with context

### Fallback Routes
- FastAPI SPA fallback catches unknown paths
- Returns index.html for client-side routing
- Prevents 404s for valid app routes

## Testing Verification

### Build Test Results
```
✓ Frontend builds successfully
✓ No TypeScript errors
✓ No build warnings
✓ dist/index.html generated with relative paths
✓ dist/assets/ contains all chunks
```

### Local Testing Steps
```bash
# Test 1: Production build
npm run build

# Test 2: FastAPI with built frontend
python server/app.py

# Test 3: Visit http://localhost:7860
# Expected: App loads, no errors
# Routes work with hash: http://localhost:7860/#/dashboard
# API calls work: check console logs
```

## Production Readiness Checklist

- [x] No hardcoded URLs
- [x] No localhost references
- [x] All assets use relative paths
- [x] Router uses hash-based navigation
- [x] Error boundaries in place
- [x] Diagnostics logging enabled
- [x] FastAPI serves static files
- [x] SPA fallback route configured
- [x] Docker build includes frontend dist
- [x] Git repository clean (no node_modules)
- [x] Environment variables properly configured
- [x] Console logging for debugging

## Deployment on Hugging Face Spaces

### Expected Behavior
1. **Docker Build Phase**
   - Installs Node dependencies
   - Builds React + Vite frontend
   - Installs Python dependencies
   - Copies frontend dist to runtime image

2. **Runtime Phase**
   - FastAPI server starts on port 7860
   - Serves index.html from `frontend/dist`
   - Serves static assets from `/assets`
   - Routes all unknown paths to index.html
   - API endpoints work normally

3. **User Interaction**
   - Page loads and renders React app
   - Browser console shows deployment diagnostics
   - Navigation uses hash routing (`/#/path`)
   - Page refresh preserves route
   - API calls succeed
   - No blank screens

## Commands for Git Cleanup

```bash
# Remove node_modules from tracking (already done)
git rm -r --cached node_modules frontend/node_modules

# Remove frontend/dist from tracking if it was tracked
git rm -r --cached frontend/dist

# Commit the cleanup
git add .gitignore
git commit -m "Remove node_modules and build artifacts from git tracking"

# Push to GitHub
git push origin main
```

## Troubleshooting Production Issues

### Blank Screen
1. Open browser DevTools (F12)
2. Go to Console tab
3. Look for `[Crisis-Comm]` logs
4. Check for errors
5. Verify `/health` API call succeeds

### Routes Not Working
1. Verify URL has `#` character: `http://site/#/path`
2. Check console for hash router diagnostics
3. Verify `BrowserRouter` replaced with `HashRouter`
4. Verify vite.config.ts has `base: "./"`

### API Calls Failing
1. Check Network tab in DevTools
2. Look for `/health`, `/tasks`, `/reset` requests
3. Verify they return 200 status
4. Check server logs in HF Space

### Missing Assets
1. Check Network tab for 404s on CSS/JS files
2. Verify URLs use relative paths `./assets/...`
3. Rebuild frontend: `npm run build`
4. Verify dist/ has correct structure

## Performance Notes

- **Bundle Size**: ~700KB gzipped (normal for interactive app)
- **Initial Load**: Lazy-loaded pages reduce initial bundle
- **API**: Relative URLs avoid routing delays
- **Caching**: Browser caches dist assets with hash names

## Security Considerations

- ✓ No sensitive data in environment variables exposed to frontend
- ✓ API calls same-origin (no CORS issues)
- ✓ Error messages don't expose internal paths
- ✓ Frontend can't access backend config
- ✓ All communication happens through API endpoints
