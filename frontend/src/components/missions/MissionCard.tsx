import { motion } from "framer-motion";
import { Clock3, PlayCircle, Signal, Target } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { formatLabel } from "@/lib/utils";
import { TaskSummary, ThreatSeverity } from "@/types/api";

interface MissionCardProps {
  task: TaskSummary;
  progress: number;
  status: "ready" | "active" | "completed";
  onStart: (taskName: string) => void;
}

const severityByDifficulty: Record<TaskSummary["difficulty"], ThreatSeverity> = {
  easy: "low",
  medium: "medium",
  hard: "high",
  challenge: "critical"
};

const severityTone: Record<ThreatSeverity, string> = {
  low: "success",
  medium: "warning",
  high: "danger",
  critical: "danger"
};

const severityColor: Record<ThreatSeverity, string> = {
  low: "#10b981",
  medium: "#facc15",
  high: "#fb923c",
  critical: "#ef4444"
};

export function MissionCard({ task, progress, status, onStart }: MissionCardProps) {
  const severity = severityByDifficulty[task.difficulty];
  const radius = 34;
  const circumference = 2 * Math.PI * radius;
  const strokeOffset = circumference - (progress / 100) * circumference;

  return (
    <motion.div
      whileHover={{ y: -4, rotateX: 4, rotateY: -4 }}
      transition={{ type: "spring", stiffness: 200, damping: 20 }}
      style={{ transformStyle: "preserve-3d" }}
    >
      <Card className="glass-panel neon-border border-slate-700/55">
        <CardHeader className="pb-2">
          <div className="flex items-center justify-between">
            <CardTitle className="text-lg text-slate-50">{formatLabel(task.name)}</CardTitle>
            <Badge variant={severityTone[severity] as "success" | "warning" | "danger"}>{severity}</Badge>
          </div>
          <p className="text-xs text-slate-400">{task.description}</p>
        </CardHeader>

        <CardContent className="space-y-4">
          <div className="grid grid-cols-[80px_1fr] items-center gap-4">
            <div className="relative mx-auto h-20 w-20">
              <svg viewBox="0 0 80 80" className="h-full w-full -rotate-90">
                <circle cx="40" cy="40" r={radius} stroke="#1e293b" strokeWidth="8" fill="transparent" />
                <circle
                  cx="40"
                  cy="40"
                  r={radius}
                  stroke={severityColor[severity]}
                  strokeWidth="8"
                  fill="transparent"
                  strokeDasharray={circumference}
                  strokeDashoffset={strokeOffset}
                  strokeLinecap="round"
                />
              </svg>
              <span className="absolute inset-0 flex items-center justify-center text-sm font-semibold text-slate-100">
                {Math.round(progress)}%
              </span>
            </div>

            <div className="space-y-2 text-xs text-slate-300">
              <p className="flex items-center gap-2">
                <Signal className="h-3.5 w-3.5 text-cyan-300" /> Complexity: {task.difficulty}
              </p>
              <p className="flex items-center gap-2">
                <Clock3 className="h-3.5 w-3.5 text-cyan-300" /> ETA: {task.max_turns * 6} mins
              </p>
              <p className="flex items-center gap-2">
                <Target className="h-3.5 w-3.5 text-cyan-300" /> Status: {status}
              </p>
            </div>
          </div>

          <div className="space-y-2">
            <Progress value={progress} />
            <Button className="w-full" onClick={() => onStart(task.name)}>
              <PlayCircle className="mr-2 h-4 w-4" />
              {status === "active" ? "Resume Simulation" : "Start Simulation"}
            </Button>
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
