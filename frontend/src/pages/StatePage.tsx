import { Gauge, Radar, ShieldAlert, Timer } from "lucide-react";
import { useShallow } from "zustand/shallow";

import { SectionHeader } from "@/components/common/SectionHeader";
import { StateCharts } from "@/components/visuals/StateCharts";
import { ThreatMap } from "@/components/visuals/ThreatMap";
import { Progress } from "@/components/ui/progress";
import { useSimulationStore } from "@/store/simulationStore";

export function StatePage() {
  const { currentState, chartSeries, threatSeverity } = useSimulationStore(
    useShallow((state) => ({
      currentState: state.currentState,
      chartSeries: state.chartSeries,
      threatSeverity: state.threatSeverity
    }))
  );

  const completion = currentState ? (currentState.turn / currentState.max_turns) * 100 : 0;
  const pending = Object.keys(currentState?.pending_deadlines ?? {}).length;

  return (
    <div className="space-y-4">
      <SectionHeader
        title="Simulation State"
        subtitle="Realtime charts, gauges, node indicators, and timeline telemetry"
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <div className="glass-panel rounded-xl border border-slate-700/55 p-4">
          <p className="flex items-center gap-2 text-xs text-slate-300"><Gauge className="h-3.5 w-3.5 text-cyan-300" /> Progress</p>
          <p className="mt-2 text-2xl font-semibold text-white">{Math.round(completion)}%</p>
          <Progress className="mt-2" value={completion} />
        </div>
        <div className="glass-panel rounded-xl border border-slate-700/55 p-4">
          <p className="flex items-center gap-2 text-xs text-slate-300"><ShieldAlert className="h-3.5 w-3.5 text-cyan-300" /> Threat</p>
          <p className="mt-2 text-2xl font-semibold capitalize text-white">{threatSeverity}</p>
          <p className="mt-1 text-xs text-slate-400">Dynamic from events + deadlines.</p>
        </div>
        <div className="glass-panel rounded-xl border border-slate-700/55 p-4">
          <p className="flex items-center gap-2 text-xs text-slate-300"><Timer className="h-3.5 w-3.5 text-cyan-300" /> Deadlines</p>
          <p className="mt-2 text-2xl font-semibold text-white">{pending}</p>
          <p className="mt-1 text-xs text-slate-400">Pending stakeholder disclosures.</p>
        </div>
        <div className="glass-panel rounded-xl border border-slate-700/55 p-4">
          <p className="flex items-center gap-2 text-xs text-slate-300"><Radar className="h-3.5 w-3.5 text-cyan-300" /> Active Turn</p>
          <p className="mt-2 text-2xl font-semibold text-white">{currentState?.turn ?? 0}</p>
          <p className="mt-1 text-xs text-slate-400">Mission cycle index.</p>
        </div>
      </div>

      <StateCharts data={chartSeries} />
      <ThreatMap severity={threatSeverity} />
    </div>
  );
}
