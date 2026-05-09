import * as React from "react";

import { cn } from "@/lib/utils";

interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value?: number;
}

const Progress = React.forwardRef<HTMLDivElement, ProgressProps>(
  ({ className, value = 0, ...props }, ref) => {
    const safeValue = Math.max(0, Math.min(100, value));

    return (
      <div
        ref={ref}
        className={cn("relative h-2 w-full overflow-hidden rounded-full bg-slate-700/45", className)}
        {...props}
      >
        <div
          className="h-full bg-gradient-to-r from-neon via-sky-400 to-indigo-400 transition-all duration-300"
          style={{ width: `${safeValue}%` }}
        />
      </div>
    );
  }
);
Progress.displayName = "Progress";

export { Progress };
