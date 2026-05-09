import { Activity, BellRing, HeartPulse, ShieldCheck } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { useSimulationStore } from "@/store/simulationStore";

export function TopBar() {
  const { health, currentState, threatSeverity, tasks } = useSimulationStore((state) => ({
    health: state.health,
    currentState: state.currentState,
    threatSeverity: state.threatSeverity,
    tasks: state.tasks
  }));

  const activeScenarioCount = tasks.length;
  const completed = currentState?.done ? 1 : 0;

  return (
    <header className="glass-panel sticky top-0 z-30 mb-4 border border-slate-700/50 px-4 py-3">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-sm font-semibold tracking-wide text-slate-100">Crisis Command Environment</p>
          <p className="text-xs text-slate-400">
            AI-powered multi-scenario crisis response simulator
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2 text-xs">
          <Badge variant={health?.status === "ok" ? "success" : "danger"}>
            <HeartPulse className="mr-1 h-3 w-3" /> API {health?.status ?? "down"}
          </Badge>
          <Badge variant="secondary">
            <Activity className="mr-1 h-3 w-3" /> Threat {threatSeverity}
          </Badge>
          <Badge variant="default">
            <ShieldCheck className="mr-1 h-3 w-3" /> Scenarios {activeScenarioCount}
          </Badge>
          <Badge variant="warning">
            <BellRing className="mr-1 h-3 w-3" /> Completed {completed}
          </Badge>
        </div>
      </div>
    </header>
  );
}
