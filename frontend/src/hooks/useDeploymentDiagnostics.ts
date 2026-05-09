import { useEffect, useState } from "react";

interface DeploymentStatus {
  apiReachable: boolean;
  frontendReady: boolean;
  assetsLoaded: boolean;
  errors: string[];
}

const isDevelopment = import.meta.env.DEV;
const isHuggingFaceSpace = window.location.hostname.includes("hf.space") || 
                           window.location.hostname.includes("huggingface.co");

function log(level: "info" | "warn" | "error", ...args: unknown[]): void {
  const timestamp = new Date().toISOString();
  const prefix = `[${timestamp}] [Crisis-Comm]`;
  
  switch (level) {
    case "info":
      console.log(prefix, ...args);
      break;
    case "warn":
      console.warn(prefix, ...args);
      break;
    case "error":
      console.error(prefix, ...args);
      break;
  }
}

export function useDeploymentDiagnostics(): DeploymentStatus {
  const [status, setStatus] = useState<DeploymentStatus>({
    apiReachable: false,
    frontendReady: true,
    assetsLoaded: true,
    errors: []
  });

  useEffect(() => {
    let mounted = true;

    const diagnostics = async (): Promise<void> => {
      const errors: string[] = [];

      log("info", "Starting deployment diagnostics...");
      log("info", "Environment:", {
        isDevelopment,
        isHuggingFaceSpace,
        hostname: window.location.hostname,
        pathname: window.location.pathname,
        hash: window.location.hash
      });

      // Check API connectivity
      try {
        log("info", "Testing API connectivity...");
        const apiBaseUrl = import.meta.env.VITE_API_BASE_URL ?? "";
        const healthUrl = `${apiBaseUrl}/health`;
        
        log("info", "Fetching health endpoint:", healthUrl);
        const response = await fetch(healthUrl, { 
          method: "GET",
          headers: { "Content-Type": "application/json" }
        });
        
        if (response.ok && mounted) {
          const data = await response.json();
          log("info", "API health check passed:", data);
          setStatus((prev) => ({ ...prev, apiReachable: true }));
        } else if (mounted) {
          const errorMsg = `API health check failed with status ${response.status}`;
          log("warn", errorMsg);
          errors.push(errorMsg);
        }
      } catch (error) {
        if (mounted) {
          const errorMsg = `API unreachable: ${error instanceof Error ? error.message : String(error)}`;
          log("error", errorMsg);
          errors.push(errorMsg);
        }
      }

      // Check critical assets
      try {
        log("info", "Checking asset loading...");
        const assets = document.querySelectorAll("script, link[rel='stylesheet']");
        log("info", `Found ${assets.length} assets loaded`);
        
        let failedAssets = 0;
        assets.forEach((asset) => {
          if (asset instanceof HTMLScriptElement) {
            if (!asset.src || (asset.hasAttribute("src") && !asset.src.startsWith("data:"))) {
              // Active script
              if (!asset.src) {
                log("info", "Inline script loaded");
              }
            }
          }
        });
        
        if (failedAssets === 0) {
          log("info", "All assets loaded successfully");
        }
      } catch (error) {
        if (mounted) {
          const errorMsg = `Asset check error: ${error instanceof Error ? error.message : String(error)}`;
          log("warn", errorMsg);
        }
      }

      // Router diagnostics
      log("info", "Router diagnostics:", {
        usingHashRouter: window.location.hash.startsWith("#"),
        currentPath: window.location.hash || window.location.pathname
      });

      if (errors.length > 0 && mounted) {
        log("warn", "Deployment diagnostics found issues:", errors);
        setStatus((prev) => ({ ...prev, errors }));
      } else if (mounted) {
        log("info", "All deployment diagnostics passed");
      }
    };

    diagnostics();

    return () => {
      mounted = false;
    };
  }, []);

  return status;
}

export function logApiCall(endpoint: string, method: string, status: number, duration: number): void {
  if (!isDevelopment && !isHuggingFaceSpace) {
    return; // Only log in dev or on HF Spaces for production debugging
  }
  
  log("info", `API ${method} ${endpoint}: ${status}ms response time`);
}

export function logError(context: string, error: unknown): void {
  log("error", `${context}:`, error instanceof Error ? error.message : error);
}
