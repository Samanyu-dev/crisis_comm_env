import { create } from "zustand";

import {
  fetchHealth,
  fetchState,
  fetchTasks,
  resetSimulation,
  stepSimulation
} from "@/services/api";
import {
  type AgentNode,
  type AssistantMessage,
  type ChartPoint,
  type CrisisObservation,
  type HealthResponse,
  type LogEntry,
  type LogLevel,
  type StateSnapshot,
  type StepActionType,
  type TaskSummary,
  type ThreatSeverity,
  type TimelineEntry
} from "@/types/api";
import {
  generateChartPoint,
  severityFromState,
  severityValue
} from "@/utils/simulationMath";
import { formatTimestamp } from "@/lib/utils";
import { playCommandTone } from "@/utils/sound";

const MAX_LOGS = 260;
const MAX_CHART_POINTS = 36;

const actionInsights: Record<StepActionType, string> = {
  respond: "Crafting synchronized response statements for all stakeholders.",
  escalate: "Escalating incident to strategic command with legal oversight.",
  contain: "Containment protocol active. Limiting blast radius across channels.",
  notify_public: "Public disclosure cycle initiated with transparent fact framing.",
  investigate: "Forensic and operational investigation expanded.",
  rollback: "Rollback and service stabilization procedure executed.",
  deploy_team: "Cross-functional incident teams deployed to field operations."
};

const randomStreamMessages: Array<{ level: LogLevel; message: string; context?: string }> = [
  { level: "INFO", message: "Telemetry heartbeat stable", context: "SYSTEM" },
  { level: "WARNING", message: "Sentiment volatility spike detected", context: "PUBLIC SIGNAL" },
  { level: "CRITICAL", message: "Escalation vector crossing threshold", context: "RISK ENGINE" },
  { level: "SUCCESS", message: "Containment checkpoint verified", context: "RESPONSE OPS" },
  { level: "INFO", message: "Regulatory packet drafted", context: "LEGAL" },
  { level: "WARNING", message: "Press inquiry queue increasing", context: "PR" }
];

const defaultAgents: AgentNode[] = [
  { id: "pr", label: "PR Agent", status: "idle", confidence: 63, focus: "Narrative control" },
  { id: "cyber", label: "Cybersecurity Agent", status: "working", confidence: 78, focus: "Threat containment" },
  { id: "legal", label: "Legal Agent", status: "idle", confidence: 71, focus: "Disclosure compliance" },
  { id: "ops", label: "Operations Agent", status: "working", confidence: 68, focus: "Business continuity" }
];

function toLog(level: LogLevel, message: string, context?: string): LogEntry {
  return {
    id: crypto.randomUUID(),
    timestamp: formatTimestamp(),
    level,
    message,
    context
  };
}

function deriveThreatSeverity(
  currentState: StateSnapshot | null,
  observation: CrisisObservation | null
): ThreatSeverity {
  return severityFromState(currentState, observation?.events ?? []);
}

function buildAssistantReply(prompt: string, state: StateSnapshot | null, severity: ThreatSeverity): string {
  const step = state ? `${state.turn}/${state.max_turns}` : "unknown";
  const scenario = state?.scenario_name ?? "unassigned mission";

  if (prompt.toLowerCase().includes("predict")) {
    return `Projection for ${scenario}: severity trend is ${severity}. Prioritize regulator + customer sequence before turn ${step}.`;
  }
  if (prompt.toLowerCase().includes("strategy")) {
    return `Strategy: stabilize facts, publish synchronized audience messages, and log every decision with compliance rationale.`;
  }
  return `Recommendation: execute containment and investigation in parallel, then issue a verified response pack. Current mission turn: ${step}.`;
}

interface SimulationStore {
  bootstrapped: boolean;
  launched: boolean;
  loading: boolean;
  stepLoading: boolean;
  resetLoading: boolean;
  health: HealthResponse | null;
  tasks: TaskSummary[];
  currentState: StateSnapshot | null;
  currentObservation: CrisisObservation | null;
  selectedTask: string | null;
  threatSeverity: ThreatSeverity;
  lastReward: number;
  completionOverlayVisible: boolean;
  logs: LogEntry[];
  chartSeries: ChartPoint[];
  timeline: TimelineEntry[];
  assistantMessages: AssistantMessage[];
  agents: AgentNode[];
  soundEnabled: boolean;
  matrixEnabled: boolean;
  error: string | null;
  initialize: () => Promise<void>;
  launchSimulation: () => void;
  refreshSnapshot: () => Promise<void>;
  startScenario: (taskName: string) => Promise<void>;
  resetCurrentScenario: () => Promise<void>;
  executeAction: (action: StepActionType) => Promise<void>;
  streamTick: () => void;
  pushAssistantPrompt: (prompt: string) => void;
  dismissCompletionOverlay: () => void;
  toggleSound: () => void;
  toggleMatrix: () => void;
  clearError: () => void;
}

function initialChartSeries(): ChartPoint[] {
  const points: ChartPoint[] = [];
  for (let index = 0; index < 12; index += 1) {
    points.push(generateChartPoint(index + 1, points[index - 1]));
  }
  return points;
}

export const useSimulationStore = create<SimulationStore>((set, get) => ({
  bootstrapped: false,
  launched: false,
  loading: false,
  stepLoading: false,
  resetLoading: false,
  health: null,
  tasks: [],
  currentState: null,
  currentObservation: null,
  selectedTask: null,
  threatSeverity: "medium",
  lastReward: 0,
  completionOverlayVisible: false,
  logs: [toLog("INFO", "Command interface boot sequence complete", "SYSTEM")],
  chartSeries: initialChartSeries(),
  timeline: [],
  assistantMessages: [
    {
      id: crypto.randomUUID(),
      role: "assistant",
      text: "AI Command Assistant online. Ask for escalation prediction or response strategy.",
      timestamp: formatTimestamp()
    }
  ],
  agents: defaultAgents,
  soundEnabled: false,
  matrixEnabled: false,
  error: null,

  initialize: async () => {
    set({ loading: true, error: null });
    try {
      const [health, tasksResponse, state] = await Promise.all([
        fetchHealth(true),
        fetchTasks(true, true),
        fetchState(true)
      ]);

      const severity = deriveThreatSeverity(state, null);
      const launchStored = typeof window !== "undefined" ? localStorage.getItem("cce_launched") : null;
      const selected = state.scenario_name;

      set((prev) => ({
        bootstrapped: true,
        launched: launchStored === "1" || prev.launched,
        loading: false,
        health,
        tasks: tasksResponse.tasks,
        currentState: state,
        selectedTask: selected,
        threatSeverity: severity,
        chartSeries: [
          ...prev.chartSeries.slice(-MAX_CHART_POINTS + 1),
          {
            tick: prev.chartSeries.length + 1,
            threat: severityValue(severity),
            sentiment: prev.chartSeries.at(-1)?.sentiment ?? 58,
            financial: prev.chartSeries.at(-1)?.financial ?? 40,
            containment: prev.chartSeries.at(-1)?.containment ?? 46,
            stability: prev.chartSeries.at(-1)?.stability ?? 54
          }
        ],
        logs: [
          ...prev.logs,
          toLog("INFO", `Mission catalog loaded (${tasksResponse.tasks.length} scenarios)`, "DATA")
        ].slice(-MAX_LOGS)
      }));
    } catch (error) {
      set({
        loading: false,
        error: error instanceof Error ? error.message : "Unable to initialize dashboard"
      });
    }
  },

  launchSimulation: () => {
    if (typeof window !== "undefined") {
      localStorage.setItem("cce_launched", "1");
    }
    set({ launched: true });
  },

  refreshSnapshot: async () => {
    try {
      const [health, state] = await Promise.all([fetchHealth(), fetchState()]);
      const severity = deriveThreatSeverity(state, get().currentObservation);
      const nextTick = get().chartSeries.length + 1;
      const nextPoint = generateChartPoint(nextTick, get().chartSeries.at(-1));
      set((prev) => ({
        health,
        currentState: state,
        threatSeverity: severity,
        chartSeries: [
          ...prev.chartSeries.slice(-MAX_CHART_POINTS + 1),
          {
            tick: nextTick,
            threat: severityValue(severity),
            sentiment: nextPoint.sentiment,
            financial: nextPoint.financial,
            containment: nextPoint.containment,
            stability: nextPoint.stability
          }
        ]
      }));
    } catch (error) {
      set({ error: error instanceof Error ? error.message : "Snapshot refresh failed" });
    }
  },

  startScenario: async (taskName) => {
    set({ resetLoading: true, error: null });
    try {
      const response = await resetSimulation(taskName);
      const severity = deriveThreatSeverity(response.state, response.observation);

      set((prev) => ({
        selectedTask: taskName,
        currentState: response.state,
        currentObservation: response.observation,
        threatSeverity: severity,
        resetLoading: false,
        completionOverlayVisible: false,
        timeline: [],
        lastReward: 0,
        logs: [
          ...prev.logs,
          toLog("SUCCESS", `Scenario initialized: ${taskName}`, "RESET")
        ].slice(-MAX_LOGS)
      }));
    } catch (error) {
      set({
        resetLoading: false,
        error: error instanceof Error ? error.message : "Scenario reset failed"
      });
    }
  },

  resetCurrentScenario: async () => {
    const taskName = get().selectedTask ?? get().currentState?.scenario_name;
    if (!taskName) {
      return;
    }
    await get().startScenario(taskName);
  },

  executeAction: async (action) => {
    const taskName = get().selectedTask ?? get().currentState?.scenario_name;
    if (!taskName) {
      set({ error: "Select a scenario before dispatching actions." });
      return;
    }

    set({ stepLoading: true, error: null });

    try {
      playCommandTone(get().soundEnabled);
      const response = await stepSimulation({ action, taskName });
      const severity = deriveThreatSeverity(response.info.state_snapshot, response.observation);
      const nextTick = get().chartSeries.length + 1;
      const previousPoint = get().chartSeries.at(-1);
      const randomPoint = generateChartPoint(nextTick, previousPoint);
      const rewardPercent = Math.round(response.reward * 100);

      set((prev) => ({
        stepLoading: false,
        lastReward: response.reward,
        currentObservation: response.observation,
        currentState: response.info.state_snapshot,
        threatSeverity: severity,
        health: {
          status: "ok",
          task_name: taskName,
          turn: response.info.turn,
          done: response.done
        },
        completionOverlayVisible: response.done,
        chartSeries: [
          ...prev.chartSeries.slice(-MAX_CHART_POINTS + 1),
          {
            ...randomPoint,
            threat: severityValue(severity),
            containment: Math.min(100, randomPoint.containment + rewardPercent / 8),
            stability: Math.min(100, randomPoint.stability + rewardPercent / 10)
          }
        ],
        timeline: [
          ...prev.timeline,
          {
            id: crypto.randomUUID(),
            turn: response.info.turn,
            action,
            reward: response.reward,
            timestamp: formatTimestamp(),
            note: actionInsights[action]
          }
        ],
        logs: [
          ...prev.logs,
          toLog("INFO", `Action dispatched: ${action.toUpperCase()}`, "COMMAND"),
          toLog("SUCCESS", `Reward updated: ${rewardPercent}%`, "SCORING"),
          ...(response.info.reward_breakdown?.notes ?? []).map((note) =>
            toLog("WARNING", note, "AI GRADER")
          )
        ].slice(-MAX_LOGS),
        agents: prev.agents.map((agent) => {
          if (agent.id === "cyber" && (action === "contain" || action === "investigate")) {
            return {
              ...agent,
              status: "working",
              confidence: Math.min(99, agent.confidence + 4)
            };
          }
          if (agent.id === "legal" && action === "notify_public") {
            return {
              ...agent,
              status: "working",
              confidence: Math.min(99, agent.confidence + 3)
            };
          }
          if (agent.id === "pr" && action === "escalate") {
            return {
              ...agent,
              status: severity === "critical" ? "warning" : "working",
              confidence: Math.max(30, agent.confidence - (severity === "critical" ? 7 : -2))
            };
          }
          if (response.done) {
            return {
              ...agent,
              status: "idle"
            };
          }
          return agent;
        })
      }));
    } catch (error) {
      set({
        stepLoading: false,
        error: error instanceof Error ? error.message : "Action dispatch failed"
      });
    }
  },

  streamTick: () => {
    const item = randomStreamMessages[Math.floor(Math.random() * randomStreamMessages.length)];
    set((prev) => {
      const state = prev.currentState;
      const evolvingPoint = generateChartPoint(prev.chartSeries.length + 1, prev.chartSeries.at(-1));
      const severity = deriveThreatSeverity(state, prev.currentObservation);

      return {
        logs: [...prev.logs, toLog(item.level, item.message, item.context)].slice(-MAX_LOGS),
        chartSeries: [
          ...prev.chartSeries.slice(-MAX_CHART_POINTS + 1),
          {
            ...evolvingPoint,
            threat: severityValue(severity)
          }
        ],
        agents: prev.agents.map((agent) => {
          if (Math.random() > 0.7) {
            const statusPool: AgentNode["status"][] = ["idle", "working", "warning"];
            return {
              ...agent,
              status: statusPool[Math.floor(Math.random() * statusPool.length)],
              confidence: Math.max(25, Math.min(99, agent.confidence + (Math.random() - 0.5) * 5))
            };
          }
          return agent;
        })
      };
    });
  },

  pushAssistantPrompt: (prompt) => {
    const now = formatTimestamp();
    const replyText = buildAssistantReply(prompt, get().currentState, get().threatSeverity);

    set((prev) => ({
      assistantMessages: [
        ...prev.assistantMessages,
        {
          id: crypto.randomUUID(),
          role: "user",
          text: prompt,
          timestamp: now
        },
        {
          id: crypto.randomUUID(),
          role: "assistant",
          text: replyText,
          timestamp: formatTimestamp()
        }
      ],
      logs: [
        ...prev.logs,
        toLog("INFO", `AI assistant analyzed prompt: ${prompt}`, "ASSISTANT")
      ].slice(-MAX_LOGS)
    }));
  },

  dismissCompletionOverlay: () => {
    set({ completionOverlayVisible: false });
  },

  toggleSound: () => {
    set((state) => ({
      soundEnabled: !state.soundEnabled,
      logs: [
        ...state.logs,
        toLog("INFO", `Command audio ${state.soundEnabled ? "disabled" : "enabled"}`, "SETTINGS")
      ].slice(-MAX_LOGS)
    }));
  },

  toggleMatrix: () => {
    set((state) => ({
      matrixEnabled: !state.matrixEnabled,
      logs: [
        ...state.logs,
        toLog("INFO", `Matrix overlay ${state.matrixEnabled ? "disabled" : "enabled"}`, "SETTINGS")
      ].slice(-MAX_LOGS)
    }));
  },

  clearError: () => {
    set({ error: null });
  }
}));
