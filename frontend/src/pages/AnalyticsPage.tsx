import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

import { SectionHeader } from "@/components/common/SectionHeader";
import { ThreatMap } from "@/components/visuals/ThreatMap";
import { useSimulationStore } from "@/store/simulationStore";

export function AnalyticsPage() {
  const { chartSeries, timeline, threatSeverity } = useSimulationStore((state) => ({
    chartSeries: state.chartSeries,
    timeline: state.timeline,
    threatSeverity: state.threatSeverity
  }));

  const rewardData = timeline.map((entry, index) => ({
    step: index + 1,
    reward: Number((entry.reward * 100).toFixed(1))
  }));

  return (
    <div className="space-y-4">
      <SectionHeader
        title="Analytics"
        subtitle="Mission performance tracking, reward trends, and cross-scenario pressure patterns"
      />

      <div className="grid gap-4 xl:grid-cols-[1.2fr_1fr]">
        <div className="glass-panel rounded-xl border border-slate-700/60 p-4">
          <h3 className="mb-3 text-sm font-medium text-slate-100">Reward Trajectory</h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={rewardData.length ? rewardData : [{ step: 0, reward: 0 }]}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.35} />
                <XAxis dataKey="step" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} />
                <Tooltip
                  contentStyle={{
                    background: "rgba(2, 6, 23, 0.95)",
                    border: "1px solid rgba(56, 189, 248, 0.35)",
                    borderRadius: 12
                  }}
                />
                <Line type="monotone" dataKey="reward" stroke="#34d399" strokeWidth={2.5} dot={{ r: 4 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <ThreatMap severity={threatSeverity} />
      </div>

      <div className="glass-panel rounded-xl border border-slate-700/60 p-4">
        <h3 className="mb-3 text-sm font-medium text-slate-100">Operational Stability Sequence</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartSeries.slice(-20)}>
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
              <Line type="monotone" dataKey="stability" stroke="#22d3ee" strokeWidth={2.4} dot={false} />
              <Line type="monotone" dataKey="containment" stroke="#a78bfa" strokeWidth={2.2} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}
