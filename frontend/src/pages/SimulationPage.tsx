import { useMemo } from "react";
import { ArrowRightCircle, Loader2 } from "lucide-react";
import { Link } from "react-router-dom";
import { useShallow } from "zustand/shallow";

import { SectionHeader } from "@/components/common/SectionHeader";
import { ActionBar } from "@/components/simulation/ActionBar";
import { AICommandAssistant } from "@/components/simulation/AICommandAssistant";
import { DecisionTimeline } from "@/components/simulation/DecisionTimeline";
import { MissionCompleteOverlay } from "@/components/simulation/MissionCompleteOverlay";
import { MultiAgentPanel } from "@/components/simulation/MultiAgentPanel";
import { ReplayPanel } from "@/components/simulation/ReplayPanel";
import { ResetConfirmDialog } from "@/components/simulation/ResetConfirmDialog";
import { TacticalConsole } from "@/components/simulation/TacticalConsole";
import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/useToast";
import { useSimulationStore } from "@/store/simulationStore";

export function SimulationPage() {
  const { toast } = useToast();
  const {
    currentState,
    currentObservation,
    logs,
    lastReward,
    threatSeverity,
    stepLoading,
    resetLoading,
    completionOverlayVisible,
    agents,
    timeline,
    executeAction,
    resetCurrentScenario,
    dismissCompletionOverlay,
    assistantMessages,
    pushAssistantPrompt
  } = useSimulationStore(
    useShallow((state) => ({
      currentState: state.currentState,
      currentObservation: state.currentObservation,
      logs: state.logs,
      lastReward: state.lastReward,
      threatSeverity: state.threatSeverity,
      stepLoading: state.stepLoading,
      resetLoading: state.resetLoading,
      completionOverlayVisible: state.completionOverlayVisible,
      agents: state.agents,
      timeline: state.timeline,
      executeAction: state.executeAction,
      resetCurrentScenario: state.resetCurrentScenario,
      dismissCompletionOverlay: state.dismissCompletionOverlay,
      assistantMessages: state.assistantMessages,
      pushAssistantPrompt: state.pushAssistantPrompt
    }))
  );

  const missionTitle = useMemo(
    () => (currentState?.scenario_name ? currentState.scenario_name.replace(/-/g, " ") : "No mission active"),
    [currentState?.scenario_name]
  );

  if (!currentState) {
    return (
      <div className="glass-panel rounded-xl border border-slate-700/55 p-8 text-center">
        <p className="text-sm text-slate-300">No active simulation mission yet.</p>
        <Button asChild className="mt-4">
          <Link to="/missions">
            <ArrowRightCircle className="mr-2 h-4 w-4" /> Open Missions
          </Link>
        </Button>
      </div>
    );
  }

  return (
    <>
      <div className="space-y-4">
        <SectionHeader
          title="Tactical Simulation"
          subtitle={`Mission: ${missionTitle} • Turn ${currentState.turn}/${currentState.max_turns}`}
          actions={
            <ResetConfirmDialog
              onConfirm={async () => {
                await resetCurrentScenario();
                toast({
                  title: "Simulation reset complete",
                  description: "Mission state returned to turn one.",
                  tone: "success"
                });
              }}
              loading={resetLoading}
            />
          }
        />

        <TacticalConsole
          state={currentState}
          observation={currentObservation}
          logs={logs}
          lastReward={lastReward}
          severity={threatSeverity}
        />

        <ActionBar
          loading={stepLoading}
          onAction={async (action) => {
            await executeAction(action);
            toast({
              title: `Action executed: ${action.replace("_", " ")}`,
              description: "Simulation state and charts updated.",
              tone: "default"
            });
          }}
        />

        {stepLoading ? (
          <div className="inline-flex items-center gap-2 rounded-lg border border-cyan-500/30 bg-cyan-500/10 px-3 py-2 text-xs text-cyan-100">
            <Loader2 className="h-3.5 w-3.5 animate-spin" /> Updating simulation state...
          </div>
        ) : null}

        <div className="grid gap-4 xl:grid-cols-3">
          <DecisionTimeline timeline={timeline} />
          <ReplayPanel timeline={timeline} />
          <MultiAgentPanel agents={agents} />
        </div>

        <div className="grid gap-4 xl:grid-cols-2">
          <AICommandAssistant messages={assistantMessages} onSubmit={pushAssistantPrompt} />
          <div className="glass-panel rounded-xl border border-slate-700/60 p-4">
            <p className="mb-2 text-sm font-medium text-slate-100">AI Reasoning Highlights</p>
            <div className="space-y-2">
              {assistantMessages.slice(-3).map((message) => (
                <div
                  key={message.id}
                  className={`rounded-lg border p-3 text-xs ${
                    message.role === "assistant"
                      ? "border-cyan-400/30 bg-cyan-500/10 text-cyan-100"
                      : "border-slate-600/50 bg-slate-800/40 text-slate-100"
                  }`}
                >
                  <p>{message.text}</p>
                  <p className="mt-1 text-[10px] opacity-70">{message.timestamp}</p>
                </div>
              ))}
            </div>
            <Button
              variant="secondary"
              className="mt-3"
              onClick={() => pushAssistantPrompt("Predict escalation")}
            >
              Predict escalation
            </Button>
          </div>
        </div>
      </div>

      <MissionCompleteOverlay open={completionOverlayVisible} onClose={dismissCompletionOverlay} />
    </>
  );
}
