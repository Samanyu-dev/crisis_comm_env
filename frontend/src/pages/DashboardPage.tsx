import {
  Activity,
  AlertOctagon,
  BrainCircuit,
  HeartPulse,
  ListChecks,
  TrendingUp
} from "lucide-react";

import { MetricsOverview } from "@/components/dashboard/MetricsOverview";
import { SectionHeader } from "@/components/common/SectionHeader";
import { ThreatMap } from "@/components/visuals/ThreatMap";
import { AICommandAssistant } from "@/components/simulation/AICommandAssistant";
import { useSimulationStore } from "@/store/simulationStore";
import { formatPercent } from "@/lib/utils";

export function DashboardPage() {
  const {
    health,
    tasks,
    currentState,
    threatSeverity,
    lastReward,
    chartSeries,
    assistantMessages,
    pushAssistantPrompt
  } = useSimulationStore((state) => ({
    health: state.health,
    tasks: state.tasks,
    currentState: state.currentState,
    threatSeverity: state.threatSeverity,
    lastReward: state.lastReward,
    chartSeries: state.chartSeries,
    assistantMessages: state.assistantMessages,
    pushAssistantPrompt: state.pushAssistantPrompt
  }));

  const latestTrend = chartSeries.slice(-7).map((point) => point.threat);
  const sentimentTrend = chartSeries.slice(-7).map((point) => point.sentiment);
  const containmentTrend = chartSeries.slice(-7).map((point) => point.containment);

  const progress = currentState ? (currentState.turn / currentState.max_turns) * 100 : 0;

  const metricItems = [
    {
      metric: {
        id: "system-status",
        label: "System Status",
        value: health?.status === "ok" ? 100 : 32,
        suffix: "%",
        trend: latestTrend,
        tone: health?.status === "ok" ? ("good" as const) : ("danger" as const)
      },
      icon: HeartPulse
    },
    {
      metric: {
        id: "active-tasks",
        label: "Active Tasks",
        value: tasks.length,
        trend: sentimentTrend,
        tone: "neutral" as const
      },
      icon: ListChecks
    },
    {
      metric: {
        id: "threat-level",
        label: "Threat Level",
        value:
          threatSeverity === "critical"
            ? 95
            : threatSeverity === "high"
              ? 75
              : threatSeverity === "medium"
                ? 55
                : 32,
        suffix: "%",
        trend: latestTrend,
        tone:
          threatSeverity === "critical" || threatSeverity === "high"
            ? ("danger" as const)
            : threatSeverity === "medium"
              ? ("warning" as const)
              : ("good" as const)
      },
      icon: AlertOctagon
    },
    {
      metric: {
        id: "response-accuracy",
        label: "Response Accuracy",
        value: (lastReward || 0.67) * 100,
        suffix: "%",
        trend: containmentTrend,
        tone: lastReward > 0.7 ? ("good" as const) : ("warning" as const)
      },
      icon: BrainCircuit
    },
    {
      metric: {
        id: "simulation-progress",
        label: "Simulation Progress",
        value: progress,
        suffix: "%",
        trend: chartSeries.slice(-7).map((point) => point.stability),
        tone: "neutral" as const
      },
      icon: TrendingUp
    },
    {
      metric: {
        id: "api-health",
        label: "API Health",
        value: health?.status === "ok" ? 99 : 40,
        suffix: "%",
        trend: chartSeries.slice(-7).map((point) => point.financial),
        tone: health?.status === "ok" ? ("good" as const) : ("warning" as const)
      },
      icon: Activity
    }
  ];

  return (
    <div className="space-y-4">
      <SectionHeader
        title="Mission Dashboard"
        subtitle={`Current Mission: ${currentState?.scenario_name ?? "No scenario selected"} • Progress ${formatPercent(progress)}`}
      />

      <MetricsOverview items={metricItems} />

      <div className="grid gap-4 xl:grid-cols-[1.2fr_1fr]">
        <ThreatMap severity={threatSeverity} />
        <AICommandAssistant messages={assistantMessages} onSubmit={pushAssistantPrompt} />
      </div>
    </div>
  );
}
