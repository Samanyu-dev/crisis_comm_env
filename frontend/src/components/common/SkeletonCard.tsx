import { cn } from "@/lib/utils";

interface SkeletonCardProps {
  className?: string;
}

export function SkeletonCard({ className }: SkeletonCardProps) {
  return (
    <div className={cn("glass-panel overflow-hidden rounded-xl border border-slate-700/50 p-5", className)}>
      <div className="shimmer h-4 w-1/2 rounded" />
      <div className="mt-4 shimmer h-7 w-1/3 rounded" />
      <div className="mt-4 shimmer h-20 w-full rounded" />
    </div>
  );
}
