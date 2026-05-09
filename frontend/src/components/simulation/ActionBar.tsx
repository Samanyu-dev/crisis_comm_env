import { Activity, Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ACTION_BUTTONS } from "@/lib/constants";
import { StepActionType } from "@/types/api";

interface ActionBarProps {
  onAction: (action: StepActionType) => void;
  loading?: boolean;
}

export function ActionBar({ onAction, loading = false }: ActionBarProps) {
  return (
    <div className="glass-panel mt-4 rounded-xl border border-slate-700/60 p-3">
      <div className="mb-2 flex items-center gap-2 text-xs text-slate-300">
        <Activity className="h-3.5 w-3.5 text-cyan-300" />
        Decision Controls
      </div>
      <div className="grid grid-cols-2 gap-2 md:grid-cols-4 xl:grid-cols-7">
        {ACTION_BUTTONS.map((button) => (
          <Button
            key={button.id}
            variant="secondary"
            disabled={loading}
            className={`justify-center ${button.tone}`}
            onClick={() => onAction(button.id)}
          >
            {loading ? <Loader2 className="mr-1 h-3.5 w-3.5 animate-spin" /> : null}
            {button.label}
          </Button>
        ))}
      </div>
    </div>
  );
}
