import { AlertTriangle, CheckCheck, Database, RadioTower } from "lucide-react";

import { EventLogTerminal } from "@/components/logs/EventLogTerminal";
import { SectionHeader } from "@/components/common/SectionHeader";
import { useSimulationStore } from "@/store/simulationStore";

export function LogsPage() {
  const logs = useSimulationStore((state) => state.logs);

  const critical = logs.filter((log) => log.level === "CRITICAL").length;
  const warning = logs.filter((log) => log.level === "WARNING").length;
  const success = logs.filter((log) => log.level === "SUCCESS").length;

  return (
    <div className="space-y-4">
      <SectionHeader
        title="Event Logs"
        subtitle="Terminal-style realtime feed with streaming updates and level filters"
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <div className="glass-panel rounded-xl border border-slate-700/55 p-4">
          <p className="flex items-center gap-2 text-xs text-slate-300"><Database className="h-3.5 w-3.5 text-cyan-300" /> Entries</p>
          <p className="mt-2 text-2xl font-semibold text-white">{logs.length}</p>
        </div>
        <div className="glass-panel rounded-xl border border-slate-700/55 p-4">
          <p className="flex items-center gap-2 text-xs text-slate-300"><AlertTriangle className="h-3.5 w-3.5 text-rose-300" /> Critical</p>
          <p className="mt-2 text-2xl font-semibold text-rose-200">{critical}</p>
        </div>
        <div className="glass-panel rounded-xl border border-slate-700/55 p-4">
          <p className="flex items-center gap-2 text-xs text-slate-300"><RadioTower className="h-3.5 w-3.5 text-amber-300" /> Warning</p>
          <p className="mt-2 text-2xl font-semibold text-amber-200">{warning}</p>
        </div>
        <div className="glass-panel rounded-xl border border-slate-700/55 p-4">
          <p className="flex items-center gap-2 text-xs text-slate-300"><CheckCheck className="h-3.5 w-3.5 text-emerald-300" /> Success</p>
          <p className="mt-2 text-2xl font-semibold text-emerald-200">{success}</p>
        </div>
      </div>

      <EventLogTerminal logs={logs} />
    </div>
  );
}
