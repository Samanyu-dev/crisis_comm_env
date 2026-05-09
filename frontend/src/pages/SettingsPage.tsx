import { MonitorCog, Music2, RefreshCcw, Sparkle } from "lucide-react";

import { SectionHeader } from "@/components/common/SectionHeader";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/useToast";
import { useSimulationStore } from "@/store/simulationStore";

export function SettingsPage() {
  const { toast } = useToast();
  const {
    soundEnabled,
    matrixEnabled,
    toggleSound,
    toggleMatrix,
    initialize
  } = useSimulationStore((state) => ({
    soundEnabled: state.soundEnabled,
    matrixEnabled: state.matrixEnabled,
    toggleSound: state.toggleSound,
    toggleMatrix: state.toggleMatrix,
    initialize: state.initialize
  }));

  return (
    <div className="space-y-4">
      <SectionHeader
        title="Settings"
        subtitle="Command center presentation, UX effects, and session controls"
      />

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="glass-panel rounded-xl border border-slate-700/60 p-4">
          <h3 className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-100">
            <Music2 className="h-4 w-4 text-cyan-300" /> Command Center Sounds
          </h3>
          <p className="text-sm text-slate-400">Toggle synthetic command tones for action dispatch events.</p>
          <Button
            className="mt-3"
            variant={soundEnabled ? "default" : "secondary"}
            onClick={() => {
              toggleSound();
              toast({
                title: `Audio ${soundEnabled ? "disabled" : "enabled"}`,
                tone: "default"
              });
            }}
          >
            {soundEnabled ? "Disable Audio" : "Enable Audio"}
          </Button>
        </div>

        <div className="glass-panel rounded-xl border border-slate-700/60 p-4">
          <h3 className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-100">
            <Sparkle className="h-4 w-4 text-cyan-300" /> Matrix Rain Overlay
          </h3>
          <p className="text-sm text-slate-400">Enable optional matrix rain background effect for cinematic mode.</p>
          <Button
            className="mt-3"
            variant={matrixEnabled ? "default" : "secondary"}
            onClick={() => {
              toggleMatrix();
              toast({
                title: `Matrix overlay ${matrixEnabled ? "disabled" : "enabled"}`,
                tone: "default"
              });
            }}
          >
            {matrixEnabled ? "Disable Matrix" : "Enable Matrix"}
          </Button>
        </div>

        <div className="glass-panel rounded-xl border border-slate-700/60 p-4">
          <h3 className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-100">
            <RefreshCcw className="h-4 w-4 text-cyan-300" /> Session Sync
          </h3>
          <p className="text-sm text-slate-400">Force refresh API state, task catalog, and health indicators.</p>
          <Button
            className="mt-3"
            onClick={async () => {
              await initialize();
              toast({
                title: "Session synchronized",
                description: "Telemetry and mission catalog refreshed.",
                tone: "success"
              });
            }}
          >
            Reload Session
          </Button>
        </div>

        <div className="glass-panel rounded-xl border border-slate-700/60 p-4">
          <h3 className="mb-3 flex items-center gap-2 text-sm font-medium text-slate-100">
            <MonitorCog className="h-4 w-4 text-cyan-300" /> Runtime
          </h3>
          <p className="text-sm text-slate-400">API Base URL: {import.meta.env.VITE_API_BASE_URL || "same-origin"}</p>
          <p className="mt-2 text-xs text-slate-500">
            Deploy frontend-only builds by setting VITE_API_BASE_URL to your FastAPI endpoint.
          </p>
        </div>
      </div>
    </div>
  );
}
