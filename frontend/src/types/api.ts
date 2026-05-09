export type ThreatSeverity = "low" | "medium" | "high" | "critical";

export interface TaskSummary {
  name: string;
  difficulty: "easy" | "medium" | "hard" | "challenge";
  description: string;
  max_turns: number;
  audiences: string[];
  disclosure_deadlines: Record<string, number>;
  required_disclosures: string[];
  forbidden_statements: string[];
  baseline_score_range: [number, number];
}

export interface TasksResponse {
  tasks: TaskSummary[];
}

export interface HealthResponse {
  status: string;
  task_name: string;
  turn: number;
  done: boolean;
}

export interface TurnEvent {
  turn: number;
  event_type: "new_fact" | "false_fact" | "stakeholder_pressure" | "stress_event";
  content: string;
  source: string;
  is_true: boolean;
  stress_level: "normal" | "escalation" | "crisis";
}

export interface StakeholderMessage {
  audience: "employees" | "customers" | "regulators" | "press";
  content: string;
}

export interface CrisisObservation {
  task_name: string;
  scenario_description: string;
  difficulty: "easy" | "medium" | "hard" | "challenge";
  turn: number;
  max_turns: number;
  events: TurnEvent[];
  available_audiences: Array<"employees" | "customers" | "regulators" | "press">;
  prior_statements: StakeholderMessage[];
  pending_deadlines: Record<string, number>;
  required_disclosures: string[];
  forbidden_statements: string[];
  done: boolean;
}

export interface StateSnapshot {
  episode_id: string;
  scenario_name: string;
  turn: number;
  max_turns: number;
  done: boolean;
  notified_audiences: string[];
  pending_deadlines: Record<string, number>;
  prior_statements: StakeholderMessage[];
  internal_notes_history: string[];
  transcript: Array<{ turn: number; audience: string; content: string }>;
  task_summary: TaskSummary;
}

export interface ResetResponse {
  observation: CrisisObservation;
  state: StateSnapshot;
}

export interface StepResponse {
  observation: CrisisObservation;
  reward: number;
  done: boolean;
  info: {
    episode_id: string;
    task_name: string;
    turn: number;
    max_turns: number;
    done: boolean;
    state_snapshot: StateSnapshot;
    reward_info?: Record<string, unknown>;
    reward_breakdown?: {
      factual_accuracy: number;
      audience_alignment: number;
      timeliness: number;
      consistency: number;
      legal_safety: number;
      proactive_disclosure: number;
      exploit_penalty: number;
      total: number;
      notes: string[];
    };
  };
}

export type StepActionType =
  | "respond"
  | "escalate"
  | "contain"
  | "notify_public"
  | "investigate"
  | "rollback"
  | "deploy_team";

export type LogLevel = "INFO" | "WARNING" | "CRITICAL" | "SUCCESS";

export interface LogEntry {
  id: string;
  timestamp: string;
  level: LogLevel;
  message: string;
  context?: string;
}

export interface TimelineEntry {
  id: string;
  turn: number;
  action: StepActionType;
  reward: number;
  timestamp: string;
  note: string;
}

export interface ChartPoint {
  tick: number;
  threat: number;
  sentiment: number;
  financial: number;
  containment: number;
  stability: number;
}

export interface AgentNode {
  id: "pr" | "cyber" | "legal" | "ops";
  label: string;
  status: "idle" | "working" | "warning" | "locked";
  confidence: number;
  focus: string;
}

export interface AssistantMessage {
  id: string;
  role: "user" | "assistant";
  text: string;
  timestamp: string;
}

export interface MetricCardData {
  id: string;
  label: string;
  value: number;
  suffix?: string;
  trend: number[];
  tone: "neutral" | "good" | "warning" | "danger";
}
