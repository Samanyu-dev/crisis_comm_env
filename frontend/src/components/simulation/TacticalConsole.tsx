import { AlertTriangle, BrainCircuit, Siren, Timer } from "lucide-react";

import { IncidentNetwork } from "@/components/visuals/IncidentNetwork";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Progress } from "@/components/ui/progress";
import { StatusPill } from "@/components/common/StatusPill";
import { formatLabel } from "@/lib/utils";
import { CrisisObservation, LogEntry, StateSnapshot, ThreatSeverity } from "@/types/api";

interface TacticalConsoleProps {
  state: StateSnapshot | null;
  observation: CrisisObservation | null;
  logs: LogEntry[];
  lastReward: number;
  severity: ThreatSeverity;
}

function severityTone(severity: ThreatSeverity): "good" | "warning" | "danger" | "neutral" {
  if (severity === "critical" || severity === "high") return "danger";
  if (severity === "medium") return "warning";
  if (severity === "low") return "good";
  return "neutral";
}

export function TacticalConsole({ state, observation, logs, lastReward, severity }: TacticalConsoleProps) {
  const latestLogs = logs.slice(-8).reverse();
  const pendingCount = Object.keys(state?.pending_deadlines ?? {}).length;

  return (
    <div className="grid gap-4 xl:grid-cols-[1fr_1.3fr_1fr]">
      <section className="glass-panel rounded-xl border border-slate-700/60 p-4">
        <h3 className="text-sm font-semibold text-slate-100">Crisis Details</h3>
        <div className="mt-2 space-y-3 text-xs text-slate-300">
          <p>{observation?.scenario_description ?? state?.task_summary.description ?? "No scenario loaded."}</p>
          <div className="flex flex-wrap gap-2">
            <StatusPill label={`Threat ${severity}`} tone={severityTone(severity)} />
            <StatusPill label={`Turn ${state?.turn ?? 0}/${state?.max_turns ?? 0}`} tone="neutral" />
          </div>

          <div className="rounded-lg border border-slate-700/50 bg-slate-900/50 p-3">
            <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.08em] text-slate-300">Objectives</p>
            <ul className="space-y-1">
              {(observation?.required_disclosures ?? state?.task_summary.required_disclosures ?? []).slice(0, 5).map((item) => (
                <li key={item} className="text-[11px] text-slate-400">• {item}</li>
              ))}
            </ul>
          </div>

          <div className="rounded-lg border border-slate-700/50 bg-slate-900/50 p-3">
            <p className="mb-2 text-[11px] font-semibold uppercase tracking-[0.08em] text-slate-300">Timeline</p>
            <div className="space-y-2">
              {(observation?.events ?? []).length === 0 ? <p className="text-[11px] text-slate-400">Awaiting incoming events...</p> : null}
              {(observation?.events ?? []).map((event) => (
                <div key={`${event.turn}-${event.content}`} className="rounded border border-slate-700/40 bg-slate-950/50 p-2">
                  <p className="text-[11px] text-cyan-100">T{event.turn} • {formatLabel(event.event_type)}</p>
                  <p className="mt-1 text-[11px] text-slate-300">{event.content}</p>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="glass-panel rounded-xl border border-slate-700/60 p-4">
        <h3 className="mb-3 text-sm font-semibold text-slate-100">Live Simulation Visualization</h3>
        <IncidentNetwork severity={severity} />

        <div className="mt-4 grid gap-3 md:grid-cols-2">
          <div className="rounded-lg border border-slate-700/50 bg-slate-900/55 p-3">
            <p className="mb-1 flex items-center gap-1 text-[11px] text-slate-300">
              <BrainCircuit className="h-3.5 w-3.5 text-cyan-300" /> AI Recommendations
            </p>
            <ul className="space-y-1 text-[11px] text-slate-400">
              <li>• Prioritize fact consistency across audiences.</li>
              <li>• Notify regulators before press expansion.</li>
              <li>• Log rationale for every escalation decision.</li>
            </ul>
          </div>

          <div className="rounded-lg border border-slate-700/50 bg-slate-900/55 p-3">
            <p className="mb-1 flex items-center gap-1 text-[11px] text-slate-300">
              <Timer className="h-3.5 w-3.5 text-cyan-300" /> Incident Propagation
            </p>
            <div className="space-y-2 text-[11px] text-slate-300">
              <p>Pending deadlines: {pendingCount}</p>
              <Progress value={Math.min(100, pendingCount * 23 + 12)} />
              <p className="text-slate-400">Higher pressure means wider propagation risk.</p>
            </div>
          </div>
        </div>
      </section>

      <section className="glass-panel rounded-xl border border-slate-700/60 p-4">
        <h3 className="mb-3 text-sm font-semibold text-slate-100">Intel Feed</h3>
        <div className="grid gap-3 text-xs text-slate-300">
          <div className="rounded-lg border border-slate-700/50 bg-slate-900/55 p-3">
            <p className="mb-1 text-[11px] font-semibold uppercase tracking-[0.08em] text-slate-300">Metrics</p>
            <p>Reward: {(lastReward * 100).toFixed(1)}%</p>
            <p>Stakeholders notified: {state?.notified_audiences.length ?? 0}</p>
            <p>Operational risk: {severity}</p>
          </div>

          <div className="rounded-lg border border-slate-700/50 bg-slate-900/55 p-3">
            <p className="mb-1 flex items-center gap-1 text-[11px] font-semibold uppercase tracking-[0.08em] text-slate-300">
              <Siren className="h-3.5 w-3.5 text-rose-300" /> Alerts
            </p>
            <ul className="space-y-1 text-[11px] text-slate-400">
              {(observation?.forbidden_statements ?? state?.task_summary.forbidden_statements ?? [])
                .slice(0, 3)
                .map((item) => (
                  <li key={item}>• Avoid: {item}</li>
                ))}
            </ul>
          </div>

          <div className="rounded-lg border border-slate-700/50 bg-slate-900/55 p-3">
            <p className="mb-1 flex items-center gap-1 text-[11px] font-semibold uppercase tracking-[0.08em] text-slate-300">
              <AlertTriangle className="h-3.5 w-3.5 text-amber-300" /> Log Snapshot
            </p>
            <ScrollArea className="h-32">
              <div className="space-y-1 text-[11px]">
                {latestLogs.map((log) => (
                  <p key={log.id} className="text-slate-400">
                    [{log.timestamp}] {log.level} {log.context ? `(${log.context})` : ""}: {log.message}
                  </p>
                ))}
              </div>
            </ScrollArea>
          </div>
        </div>
      </section>
    </div>
  );
}
