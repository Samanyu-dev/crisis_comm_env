import { motion } from "framer-motion";

import { ThreatSeverity } from "@/types/api";

interface ThreatMapProps {
  severity: ThreatSeverity;
}

const regions = [
  { id: "na", label: "North America", x: 18, y: 35 },
  { id: "eu", label: "Europe", x: 47, y: 28 },
  { id: "mea", label: "Middle East", x: 58, y: 42 },
  { id: "apac", label: "APAC", x: 74, y: 47 },
  { id: "latam", label: "Latin America", x: 30, y: 62 }
];

const severityScale: Record<ThreatSeverity, number> = {
  low: 0.35,
  medium: 0.55,
  high: 0.75,
  critical: 1
};

export function ThreatMap({ severity }: ThreatMapProps) {
  const intensity = severityScale[severity];

  return (
    <div className="glass-panel rounded-xl border border-slate-700/60 p-4">
      <h3 className="mb-3 text-sm font-medium text-slate-100">Threat Map</h3>
      <div className="relative h-[210px] overflow-hidden rounded-lg border border-slate-700/50 bg-slate-950/65">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_20%,rgba(56,189,248,0.2),transparent_40%),radial-gradient(circle_at_80%_80%,rgba(244,63,94,0.22),transparent_45%)]" />
        <svg viewBox="0 0 100 70" className="absolute inset-0 h-full w-full opacity-45">
          <path d="M8 18 L24 14 L31 24 L26 29 L14 27 Z" fill="#1f2937" />
          <path d="M34 18 L46 14 L56 20 L52 30 L40 29 Z" fill="#1f2937" />
          <path d="M58 22 L74 20 L83 26 L78 38 L64 36 Z" fill="#1f2937" />
          <path d="M22 40 L31 46 L30 58 L21 60 L16 51 Z" fill="#1f2937" />
          <path d="M68 44 L79 46 L85 55 L75 62 L66 58 Z" fill="#1f2937" />
        </svg>

        {regions.map((region, index) => (
          <motion.div
            key={region.id}
            className="absolute"
            style={{ left: `${region.x}%`, top: `${region.y}%` }}
            animate={{ scale: [1, 1 + intensity * 0.6, 1], opacity: [0.45, 0.95, 0.45] }}
            transition={{
              duration: 2.8 - intensity,
              repeat: Infinity,
              delay: index * 0.22
            }}
          >
            <span className="relative flex h-3 w-3">
              <span className="absolute inline-flex h-full w-full rounded-full bg-rose-400 opacity-60" />
              <span className="relative inline-flex h-3 w-3 rounded-full bg-cyan-300" />
            </span>
            <p className="mt-1 -translate-x-1/2 whitespace-nowrap text-[10px] text-slate-300">{region.label}</p>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
