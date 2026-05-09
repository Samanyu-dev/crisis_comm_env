import { cn } from "@/lib/utils";

interface StatusPillProps {
  label: string;
  tone?: "good" | "warning" | "danger" | "neutral";
}

const tones: Record<NonNullable<StatusPillProps["tone"]>, string> = {
  good: "border-emerald-500/40 bg-emerald-500/15 text-emerald-200",
  warning: "border-amber-500/40 bg-amber-500/15 text-amber-200",
  danger: "border-rose-500/40 bg-rose-500/15 text-rose-200",
  neutral: "border-slate-500/50 bg-slate-700/45 text-slate-200"
};

export function StatusPill({ label, tone = "neutral" }: StatusPillProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-2 py-0.5 text-[11px] font-semibold uppercase tracking-[0.08em]",
        tones[tone]
      )}
    >
      {label}
    </span>
  );
}
