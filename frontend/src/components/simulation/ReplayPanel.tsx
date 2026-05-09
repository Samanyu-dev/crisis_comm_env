import { useEffect, useState } from "react";
import { Pause, Play, RotateCcw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Progress } from "@/components/ui/progress";
import { TimelineEntry } from "@/types/api";

interface ReplayPanelProps {
  timeline: TimelineEntry[];
}

export function ReplayPanel({ timeline }: ReplayPanelProps) {
  const [playing, setPlaying] = useState(false);
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (!playing || timeline.length === 0) {
      return;
    }

    const timer = window.setInterval(() => {
      setIndex((current) => {
        if (current >= timeline.length - 1) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 1100);

    return () => window.clearInterval(timer);
  }, [playing, timeline.length]);

  useEffect(() => {
    if (index > timeline.length - 1) {
      setIndex(Math.max(0, timeline.length - 1));
    }
  }, [timeline.length, index]);

  const current = timeline[index];
  const progress = timeline.length > 1 ? (index / (timeline.length - 1)) * 100 : 0;

  return (
    <div className="glass-panel rounded-xl border border-slate-700/60 p-4">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="text-sm font-medium text-slate-100">Simulation Replay</h3>
        <div className="flex items-center gap-1">
          <Button size="icon" variant="ghost" onClick={() => setPlaying((value) => !value)}>
            {playing ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
          </Button>
          <Button
            size="icon"
            variant="ghost"
            onClick={() => {
              setPlaying(false);
              setIndex(0);
            }}
          >
            <RotateCcw className="h-4 w-4" />
          </Button>
        </div>
      </div>

      <Progress value={progress} />
      <p className="mt-2 text-xs text-slate-400">Frame {timeline.length === 0 ? 0 : index + 1} of {timeline.length}</p>

      <div className="mt-3 rounded-lg border border-slate-700/50 bg-slate-900/45 p-3 text-xs">
        {current ? (
          <>
            <p className="font-semibold uppercase text-cyan-100">{current.action.replace("_", " ")}</p>
            <p className="mt-1 text-slate-300">{current.note}</p>
            <p className="mt-1 text-slate-400">Turn {current.turn} • {current.timestamp}</p>
          </>
        ) : (
          <p className="text-slate-400">Replay will populate after mission actions are recorded.</p>
        )}
      </div>
    </div>
  );
}
