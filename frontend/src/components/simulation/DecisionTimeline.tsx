import { History } from "lucide-react";

import { ScrollArea } from "@/components/ui/scroll-area";
import { formatPercent } from "@/lib/utils";
import { TimelineEntry } from "@/types/api";

interface DecisionTimelineProps {
  timeline: TimelineEntry[];
}

export function DecisionTimeline({ timeline }: DecisionTimelineProps) {
  return (
    <div className="glass-panel rounded-xl border border-slate-700/60 p-4">
      <h3 className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-100">
        <History className="h-4 w-4 text-cyan-300" /> Decision Timeline
      </h3>
      <ScrollArea className="h-[210px] pr-2">
        <div className="space-y-3">
          {timeline.length === 0 ? (
            <p className="text-xs text-slate-400">No decisions yet. Dispatch an action to begin mission timeline tracking.</p>
          ) : null}
          {timeline.map((entry) => (
            <div key={entry.id} className="rounded-lg border border-slate-700/50 bg-slate-900/45 p-3">
              <div className="flex items-center justify-between text-xs">
                <p className="font-semibold uppercase tracking-wide text-cyan-100">{entry.action.replace("_", " ")}</p>
                <p className="text-slate-400">{entry.timestamp}</p>
              </div>
              <p className="mt-1 text-[11px] text-slate-300">Turn {entry.turn}</p>
              <p className="text-[11px] text-slate-400">{entry.note}</p>
              <p className="mt-1 text-[11px] text-emerald-200">Reward {formatPercent(entry.reward * 100)}</p>
            </div>
          ))}
        </div>
      </ScrollArea>
    </div>
  );
}
