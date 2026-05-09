import { memo } from "react";
import { Area, AreaChart, ResponsiveContainer } from "recharts";

interface TrendMiniChartProps {
  values: number[];
  tone?: "neutral" | "good" | "warning" | "danger";
}

const gradients = {
  neutral: ["#38bdf8", "#60a5fa"],
  good: ["#34d399", "#10b981"],
  warning: ["#fbbf24", "#fb923c"],
  danger: ["#f87171", "#ef4444"]
};

export const TrendMiniChart = memo(function TrendMiniChart({ values, tone = "neutral" }: TrendMiniChartProps) {
  const data = values.map((value, index) => ({ index, value }));
  const colors = gradients[tone];

  return (
    <div className="h-14 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data}>
          <defs>
            <linearGradient id={`trend-${tone}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor={colors[0]} stopOpacity={0.5} />
              <stop offset="95%" stopColor={colors[1]} stopOpacity={0.1} />
            </linearGradient>
          </defs>
          <Area
            type="monotone"
            dataKey="value"
            stroke={colors[0]}
            strokeWidth={2}
            fill={`url(#trend-${tone})`}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
});
