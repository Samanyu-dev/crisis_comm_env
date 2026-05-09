import { type ChartPoint, type StateSnapshot, type ThreatSeverity, type TurnEvent } from "@/types/api";
import { clamp } from "@/lib/utils";

export function severityFromState(state: StateSnapshot | null, events: TurnEvent[] = []): ThreatSeverity {
  if (!state) {
    return "medium";
  }

  const pressure = events.reduce((score, event) => {
    if (event.stress_level === "crisis") {
      return score + 2;
    }
    if (event.stress_level === "escalation") {
      return score + 1;
    }
    return score;
  }, 0);

  const progressRatio = state.max_turns > 0 ? state.turn / state.max_turns : 0;
  const pendingDeadlines = Object.keys(state.pending_deadlines ?? {}).length;
  const severityScore = progressRatio * 2 + pressure + pendingDeadlines * 0.8;

  if (severityScore >= 4.2) return "critical";
  if (severityScore >= 3) return "high";
  if (severityScore >= 1.8) return "medium";
  return "low";
}

export function randomWalk(prev: number, min = 10, max = 95, delta = 7): number {
  const change = (Math.random() - 0.5) * delta;
  return clamp(prev + change, min, max);
}

export function generateChartPoint(tick: number, prev?: ChartPoint): ChartPoint {
  const seed = prev ?? {
    tick,
    threat: 45,
    sentiment: 58,
    financial: 32,
    containment: 40,
    stability: 55
  };

  return {
    tick,
    threat: randomWalk(seed.threat, 18, 98, 12),
    sentiment: randomWalk(seed.sentiment, 5, 90, 9),
    financial: randomWalk(seed.financial, 10, 95, 8),
    containment: randomWalk(seed.containment, 8, 97, 10),
    stability: randomWalk(seed.stability, 5, 96, 10)
  };
}

export function severityValue(severity: ThreatSeverity): number {
  switch (severity) {
    case "low":
      return 28;
    case "medium":
      return 52;
    case "high":
      return 74;
    case "critical":
      return 92;
    default:
      return 50;
  }
}
