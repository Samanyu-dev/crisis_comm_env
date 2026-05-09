import { LucideIcon } from "lucide-react";

import { MetricCard } from "@/components/dashboard/MetricCard";
import { MetricCardData } from "@/types/api";

interface MetricsOverviewProps {
  items: Array<{ metric: MetricCardData; icon: LucideIcon }>;
}

export function MetricsOverview({ items }: MetricsOverviewProps) {
  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-6">
      {items.map((item) => (
        <MetricCard key={item.metric.id} metric={item.metric} icon={item.icon} />
      ))}
    </div>
  );
}
