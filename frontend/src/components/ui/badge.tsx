import * as React from "react";
import { cva, type VariantProps } from "class-variance-authority";

import { cn } from "@/lib/utils";

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors",
  {
    variants: {
      variant: {
        default: "border-neon/50 bg-neon/15 text-neon",
        secondary: "border-slate-500/50 bg-slate-700/40 text-slate-100",
        success: "border-emerald-500/50 bg-emerald-500/15 text-emerald-200",
        warning: "border-amber-500/50 bg-amber-500/15 text-amber-200",
        danger: "border-rose-500/50 bg-rose-500/15 text-rose-200"
      }
    },
    defaultVariants: {
      variant: "default"
    }
  }
);

export interface BadgeProps
  extends React.HTMLAttributes<HTMLDivElement>,
    VariantProps<typeof badgeVariants> {}

function Badge({ className, variant, ...props }: BadgeProps) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />;
}

export { Badge, badgeVariants };
