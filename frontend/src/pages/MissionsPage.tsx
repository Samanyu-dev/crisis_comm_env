import { useMemo } from "react";
import { Loader2 } from "lucide-react";
import { useNavigate } from "react-router-dom";

import { SectionHeader } from "@/components/common/SectionHeader";
import { SkeletonCard } from "@/components/common/SkeletonCard";
import { MissionCard } from "@/components/missions/MissionCard";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/useToast";
import { useSimulationStore } from "@/store/simulationStore";

export function MissionsPage() {
  const navigate = useNavigate();
  const { toast } = useToast();
  const { tasks, currentState, selectedTask, resetLoading, startScenario } = useSimulationStore((state) => ({
    tasks: state.tasks,
    currentState: state.currentState,
    selectedTask: state.selectedTask,
    resetLoading: state.resetLoading,
    startScenario: state.startScenario
  }));

  const scenarios = useMemo(() => {
    return tasks.map((task) => {
      const active = currentState?.scenario_name === task.name || selectedTask === task.name;
      const progress =
        active && currentState
          ? Math.min(100, (currentState.turn / currentState.max_turns) * 100)
          : 0;

      const status: "ready" | "active" | "completed" = currentState?.done && active ? "completed" : active ? "active" : "ready";
      return { task, progress, status };
    });
  }, [tasks, currentState, selectedTask]);

  return (
    <div className="space-y-4">
      <SectionHeader
        title="Active Missions"
        subtitle="Select a crisis scenario to launch tactical simulation controls."
        actions={
          <Button
            variant="secondary"
            onClick={() => navigate("/simulation")}
            disabled={!currentState}
          >
            Open Tactical View
          </Button>
        }
      />

      {tasks.length === 0 ? (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          <SkeletonCard />
          <SkeletonCard />
          <SkeletonCard />
        </div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
          {scenarios.map((item) => (
            <MissionCard
              key={item.task.name}
              task={item.task}
              progress={item.progress}
              status={item.status}
              onStart={async (taskName) => {
                await startScenario(taskName);
                toast({
                  title: "Scenario loaded",
                  description: `${taskName} is ready for tactical response controls.`,
                  tone: "success"
                });
                navigate("/simulation");
              }}
            />
          ))}
        </div>
      )}

      {resetLoading ? (
        <div className="inline-flex items-center gap-2 rounded-lg border border-cyan-500/30 bg-cyan-500/10 px-3 py-2 text-xs text-cyan-100">
          <Loader2 className="h-3.5 w-3.5 animate-spin" /> Initializing scenario...
        </div>
      ) : null}
    </div>
  );
}
