import { motion } from "framer-motion";

import { ThreatSeverity } from "@/types/api";

interface IncidentNetworkProps {
  severity: ThreatSeverity;
}

const nodes = [
  { id: "core", label: "Core Incident", x: 50, y: 48, priority: 1 },
  { id: "legal", label: "Legal", x: 22, y: 28, priority: 2 },
  { id: "pr", label: "PR", x: 79, y: 28, priority: 2 },
  { id: "ops", label: "Ops", x: 76, y: 70, priority: 2 },
  { id: "cyber", label: "Cyber", x: 23, y: 72, priority: 2 },
  { id: "public", label: "Public", x: 50, y: 16, priority: 3 }
];

const links = [
  ["core", "legal"],
  ["core", "pr"],
  ["core", "ops"],
  ["core", "cyber"],
  ["pr", "public"],
  ["legal", "public"]
];

const glowBySeverity: Record<ThreatSeverity, string> = {
  low: "rgba(16, 185, 129, 0.9)",
  medium: "rgba(250, 204, 21, 0.9)",
  high: "rgba(251, 146, 60, 0.9)",
  critical: "rgba(239, 68, 68, 0.95)"
};

export function IncidentNetwork({ severity }: IncidentNetworkProps) {
  const glow = glowBySeverity[severity];

  return (
    <div className="relative h-64 overflow-hidden rounded-xl border border-slate-700/50 bg-slate-950/60">
      <svg viewBox="0 0 100 100" className="absolute inset-0 h-full w-full">
        {links.map(([from, to]) => {
          const start = nodes.find((node) => node.id === from);
          const end = nodes.find((node) => node.id === to);
          if (!start || !end) {
            return null;
          }
          return (
            <line
              key={`${from}-${to}`}
              x1={start.x}
              y1={start.y}
              x2={end.x}
              y2={end.y}
              stroke="rgba(56, 189, 248, 0.35)"
              strokeWidth="0.5"
              strokeDasharray="2 1"
            />
          );
        })}
      </svg>

      {nodes.map((node, index) => (
        <motion.div
          key={node.id}
          className="absolute -translate-x-1/2 -translate-y-1/2"
          style={{ left: `${node.x}%`, top: `${node.y}%` }}
          animate={{ scale: [1, node.priority === 1 ? 1.2 : 1.08, 1] }}
          transition={{ duration: 2.6 - node.priority * 0.2, repeat: Infinity, delay: index * 0.1 }}
        >
          <div
            className="h-3 w-3 rounded-full"
            style={{
              background: glow,
              boxShadow: `0 0 14px ${glow}`
            }}
          />
          <p className="mt-1 -translate-x-1/2 whitespace-nowrap text-[10px] text-slate-300">{node.label}</p>
        </motion.div>
      ))}
    </div>
  );
}
