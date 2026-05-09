import axios, { AxiosError, AxiosInstance } from "axios";

import { clearCache, getCached, setCached } from "@/services/cache";
import {
  type HealthResponse,
  type ResetResponse,
  type StateSnapshot,
  type StepResponse,
  type StepActionType,
  type TasksResponse
} from "@/types/api";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? "";
const isDev = import.meta.env.DEV;

// Log API initialization
console.log("[Crisis-Comm] Initializing API service", {
  baseURL: API_BASE_URL || "(relative)",
  isDev
});

const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  timeout: 8000,
  headers: {
    "Content-Type": "application/json"
  }
});

// Add response interceptor for logging
api.interceptors.response.use(
  (response) => {
    if (!isDev) {
      console.log(`[Crisis-Comm] API ${response.config.method?.toUpperCase()} ${response.config.url}: ${response.status}`);
    }
    return response;
  },
  (error) => {
    console.error("[Crisis-Comm] API Error", {
      method: error.config?.method?.toUpperCase(),
      url: error.config?.url,
      status: error.response?.status,
      message: error.message
    });
    return Promise.reject(error);
  }
);

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => {
    window.setTimeout(resolve, ms);
  });
}

async function withRetry<T>(fn: () => Promise<T>, retries = 2): Promise<T> {
  let attempt = 0;
  let latestError: unknown;

  while (attempt <= retries) {
    try {
      return await fn();
    } catch (error) {
      latestError = error;
      if (attempt === retries) {
        break;
      }
      await delay(350 * (attempt + 1));
      attempt += 1;
    }
  }

  throw latestError;
}

function handleApiError(error: unknown): never {
  if (error instanceof AxiosError) {
    const message =
      (error.response?.data as { detail?: string })?.detail ||
      error.message ||
      "Unknown API error";
    throw new Error(message);
  }
  throw error;
}

export async function fetchHealth(force = false): Promise<HealthResponse> {
  const cacheKey = "health";
  if (!force) {
    const cached = getCached<HealthResponse>(cacheKey);
    if (cached) {
      return cached;
    }
  }

  try {
    const response = await withRetry(() => api.get<HealthResponse>("/health"));
    setCached(cacheKey, response.data, 5000);
    return response.data;
  } catch (error) {
    handleApiError(error);
  }
}

export async function fetchTasks(includeChallenge = true, force = false): Promise<TasksResponse> {
  const cacheKey = `tasks:${includeChallenge}`;
  if (!force) {
    const cached = getCached<TasksResponse>(cacheKey);
    if (cached) {
      return cached;
    }
  }

  try {
    const response = await withRetry(() =>
      api.get<TasksResponse>("/tasks", {
        params: { include_challenge: includeChallenge }
      })
    );
    setCached(cacheKey, response.data, 15000);
    return response.data;
  } catch (error) {
    handleApiError(error);
  }
}

export async function fetchState(force = false): Promise<StateSnapshot> {
  const cacheKey = "state";
  if (!force) {
    const cached = getCached<StateSnapshot>(cacheKey);
    if (cached) {
      return cached;
    }
  }

  try {
    const response = await withRetry(() => api.get<StateSnapshot>("/state"));
    setCached(cacheKey, response.data, 4000);
    return response.data;
  } catch (error) {
    handleApiError(error);
  }
}

export async function resetSimulation(taskName?: string): Promise<ResetResponse> {
  clearCache("state");
  clearCache("health");
  try {
    const response = await withRetry(() =>
      api.post<ResetResponse>("/reset", taskName ? { task_name: taskName } : {})
    );
    return response.data;
  } catch (error) {
    handleApiError(error);
  }
}

function actionMessageTemplate(action: StepActionType, taskName: string): Record<string, string> {
  const scenario = taskName.replace(/-/g, " ");
  const base = {
    employees:
      "Internal update: Crisis team is active, verified facts only, and spokesperson protocol is in effect.",
    customers:
      "We are actively investigating and will provide factual updates with clear next steps and support channels.",
    regulators:
      "Formal status update: active incident handling, evidence retention initiated, and timeline notifications underway.",
    press:
      "We can confirm an active response operation. Further updates will be issued with verified facts and remediation progress."
  };

  switch (action) {
    case "contain":
      return {
        ...base,
        employees: `Containment protocol enabled for ${scenario}. Cross-functional approvals are required before any external statement changes.`,
        regulators: "Containment actions executed. Scope isolation and impact controls are in progress with auditable trail logging."
      };
    case "notify_public":
      return {
        ...base,
        customers: `Public notice for ${scenario}: we are sharing what is known, what is not yet confirmed, and what support is available.`,
        press: "Public notice issued with verified facts, impact estimate, and remediation commitments."
      };
    case "escalate":
      return {
        ...base,
        employees: "Escalation level increased. Executive command bridge is live and all business-unit liaisons must report status.",
        regulators: "Escalation declared with additional compliance oversight and decision log continuity."
      };
    case "investigate":
      return {
        ...base,
        employees: "Deep investigation mode active. Evidence chain-of-custody and forensic workstream assignments are locked.",
        customers: "Investigation is ongoing with specialist teams validating impact details before additional commitments."
      };
    case "rollback":
      return {
        ...base,
        employees: "Rollback protocol initiated for unstable systems. Recovery checkpoints and validation gates are now mandatory.",
        customers:
          "Service stabilization actions are underway. We are prioritizing safety and continuity while restoring normal operations."
      };
    case "deploy_team":
      return {
        ...base,
        employees: "Field and remote response teams deployed. Legal, PR, cybersecurity, and operations pods are synchronized.",
        regulators: "Additional response teams have been deployed and centralized governance remains active."
      };
    case "respond":
    default:
      return base;
  }
}

export async function stepSimulation(payload: {
  action: StepActionType;
  taskName: string;
  internalNotes?: string;
}): Promise<StepResponse> {
  clearCache("state");
  clearCache("health");
  const messages = actionMessageTemplate(payload.action, payload.taskName);

  try {
    const response = await withRetry(() =>
      api.post<StepResponse>("/step", {
        messages,
        internal_notes: payload.internalNotes ?? `Command action: ${payload.action}`
      })
    );
    return response.data;
  } catch (error) {
    handleApiError(error);
  }
}
