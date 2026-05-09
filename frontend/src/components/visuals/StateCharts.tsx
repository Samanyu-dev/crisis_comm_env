import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

import { ChartPoint } from "@/types/api";

interface StateChartsProps {
  data: ChartPoint[];
}

const heatCells = [
  "#0ea5e9",
  "#38bdf8",
  "#22d3ee",
  "#34d399",
  "#facc15",
  "#fb923c",
  "#f97316",
  "#ef4444"
];

export function StateCharts({ data }: StateChartsProps) {
  const compact = data.slice(-16);

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <div className="glass-panel rounded-xl border border-slate-700/60 p-4">
        <h3 className="mb-3 text-sm font-medium text-slate-100">Threat Progression</h3>
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={compact}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.35} />
              <XAxis dataKey="tick" stroke="#94a3b8" fontSize={11} />
              <YAxis stroke="#94a3b8" fontSize={11} />
              <Tooltip
                contentStyle={{
                  background: "rgba(2, 6, 23, 0.95)",
                  border: "1px solid rgba(56, 189, 248, 0.35)",
                  borderRadius: 12
                }}
              />
              <Line type="monotone" dataKey="threat" stroke="#38bdf8" strokeWidth={2.5} dot={false} />
              <Line type="monotone" dataKey="stability" stroke="#34d399" strokeWidth={2.2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="glass-panel rounded-xl border border-slate-700/60 p-4">
        <h3 className="mb-3 text-sm font-medium text-slate-100">Public Sentiment & Financial Impact</h3>
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={compact}>
              <defs>
                <linearGradient id="sentimentGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.45} />
                  <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.04} />
                </linearGradient>
                <linearGradient id="financialGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f97316" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#f97316" stopOpacity={0.03} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.35} />
              <XAxis dataKey="tick" stroke="#94a3b8" fontSize={11} />
              <YAxis stroke="#94a3b8" fontSize={11} />
              <Tooltip
                contentStyle={{
                  background: "rgba(2, 6, 23, 0.95)",
                  border: "1px solid rgba(56, 189, 248, 0.35)",
                  borderRadius: 12
                }}
              />
              <Area type="monotone" dataKey="sentiment" stroke="#22d3ee" fill="url(#sentimentGradient)" />
              <Area type="monotone" dataKey="financial" stroke="#f97316" fill="url(#financialGradient)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="glass-panel rounded-xl border border-slate-700/60 p-4">
        <h3 className="mb-3 text-sm font-medium text-slate-100">Containment Efficiency</h3>
        <div className="h-56">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={compact.slice(-8)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.35} />
              <XAxis dataKey="tick" stroke="#94a3b8" fontSize={11} />
              <YAxis stroke="#94a3b8" fontSize={11} />
              <Tooltip
                contentStyle={{
                  background: "rgba(2, 6, 23, 0.95)",
                  border: "1px solid rgba(56, 189, 248, 0.35)",
                  borderRadius: 12
                }}
              />
              <Bar dataKey="containment" radius={[8, 8, 0, 0]}>
                {compact.slice(-8).map((_, index) => (
                  <Cell key={`cell-${index}`} fill={heatCells[index % heatCells.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="glass-panel rounded-xl border border-slate-700/60 p-4">
        <h3 className="mb-3 text-sm font-medium text-slate-100">Operational Heatmap</h3>
        <div className="grid h-56 grid-cols-8 gap-1 rounded-lg border border-slate-700/50 bg-slate-950/55 p-2">
          {Array.from({ length: 64 }).map((_, index) => {
            const tone = heatCells[(index + compact.length) % heatCells.length];
            const opacity = 0.28 + ((index * 7) % 8) * 0.08;
            return (
              <div
                key={index}
                className="rounded-sm"
                style={{ backgroundColor: tone, opacity }}
              />
            );
          })}
        </div>
      </div>
    </div>
  );
}
