import { BrainCircuit } from "lucide-react";

import { Progress } from "@/components/ui/progress";
import { cn } from "@/lib/utils";
import { AgentNode } from "@/types/api";

interface MultiAgentPanelProps {
  agents: AgentNode[];
}

const statusStyles: Record<AgentNode["status"], string> = {
  idle: "text-slate-300 border-slate-500/40 bg-slate-700/30",
  working: "text-cyan-100 border-cyan-400/35 bg-cyan-500/15",
  warning: "text-amber-100 border-amber-400/35 bg-amber-500/15",
  locked: "text-rose-100 border-rose-400/35 bg-rose-500/15"
};

export function MultiAgentPanel({ agents }: MultiAgentPanelProps) {
  return (
    <div className="glass-panel rounded-xl border border-slate-700/60 p-4">
      <h3 className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-100">
        <BrainCircuit className="h-4 w-4 text-cyan-300" /> Multi-Agent Visualization
      </h3>
      <div className="space-y-3">
        {agents.map((agent) => (
          <div key={agent.id} className="rounded-lg border border-slate-700/50 bg-slate-900/50 p-3">
            <div className="mb-1 flex items-center justify-between">
              <p className="text-xs font-semibold text-slate-100">{agent.label}</p>
              <span
                className={cn(
                  "rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-wider",
                  statusStyles[agent.status]
                )}
              >
                {agent.status}
              </span>
            </div>
            <p className="mb-2 text-[11px] text-slate-400">{agent.focus}</p>
            <Progress value={agent.confidence} />
            <p className="mt-1 text-[10px] text-slate-400">Confidence {Math.round(agent.confidence)}%</p>
          </div>
        ))}
      </div>
    </div>
  );
}
