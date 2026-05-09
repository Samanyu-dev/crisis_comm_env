import { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Terminal } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { LogEntry, LogLevel } from "@/types/api";
import { cn } from "@/lib/utils";

interface EventLogTerminalProps {
  logs: LogEntry[];
}

const levelStyles: Record<LogLevel, string> = {
  INFO: "text-cyan-200",
  WARNING: "text-amber-200",
  CRITICAL: "text-rose-200",
  SUCCESS: "text-emerald-200"
};

const levels: Array<LogLevel | "ALL"> = ["ALL", "INFO", "WARNING", "CRITICAL", "SUCCESS"];

export function EventLogTerminal({ logs }: EventLogTerminalProps) {
  const [filter, setFilter] = useState<LogLevel | "ALL">("ALL");
  const viewportRef = useRef<HTMLDivElement>(null);

  const filteredLogs = useMemo(() => {
    if (filter === "ALL") {
      return logs;
    }
    return logs.filter((log) => log.level === filter);
  }, [logs, filter]);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) {
      return;
    }
    viewport.scrollTop = viewport.scrollHeight;
  }, [filteredLogs.length]);

  return (
    <div className="glass-panel rounded-xl border border-slate-700/60 p-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <h3 className="flex items-center gap-2 text-sm font-medium text-slate-100">
          <Terminal className="h-4 w-4 text-cyan-300" /> Event Log Stream
        </h3>
        <div className="flex flex-wrap gap-1">
          {levels.map((level) => (
            <Button
              key={level}
              variant={filter === level ? "default" : "ghost"}
              size="sm"
              className="h-7"
              onClick={() => setFilter(level)}
            >
              {level}
            </Button>
          ))}
        </div>
      </div>

      <ScrollArea className="log-scanline h-[420px] rounded-lg border border-slate-700/50 bg-slate-950/80 p-3">
        <div ref={viewportRef} className="space-y-1.5 font-mono text-xs">
          <AnimatePresence initial={false}>
            {filteredLogs.map((log) => (
              <motion.div
                key={log.id}
                initial={{ opacity: 0, x: -8 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0 }}
                className={cn("flex gap-2", levelStyles[log.level])}
              >
                <span className="text-slate-500">[{log.timestamp}]</span>
                <span className="min-w-[74px]">{log.level}</span>
                {log.context ? <span className="text-slate-400">[{log.context}]</span> : null}
                <span className="text-slate-100">{log.message}</span>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </ScrollArea>
    </div>
  );
}
