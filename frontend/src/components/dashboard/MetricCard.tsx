import { motion } from "framer-motion";
import { LucideIcon } from "lucide-react";

import { TrendMiniChart } from "@/components/common/TrendMiniChart";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useCountUp } from "@/hooks/useCountUp";
import { MetricCardData } from "@/types/api";

interface MetricCardProps {
  metric: MetricCardData;
  icon: LucideIcon;
}

export function MetricCard({ metric, icon: Icon }: MetricCardProps) {
  const animatedValue = useCountUp(metric.value);

  return (
    <motion.div whileHover={{ y: -4, scale: 1.01 }} transition={{ type: "spring", stiffness: 320, damping: 24 }}>
      <Card className="glass-panel neon-border h-full border-slate-700/55">
        <CardHeader className="pb-2">
          <CardTitle className="flex items-center justify-between text-sm font-medium text-slate-300">
            <span>{metric.label}</span>
            <Icon className="h-4 w-4 text-cyan-300" />
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-2xl font-semibold text-white">
            {animatedValue.toFixed(metric.suffix ? 0 : 1)}
            {metric.suffix}
          </p>
          <div className="mt-3">
            <TrendMiniChart values={metric.trend} tone={metric.tone} />
          </div>
        </CardContent>
      </Card>
    </motion.div>
  );
}
