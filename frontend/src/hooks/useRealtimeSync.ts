import { useEffect, useRef } from "react";

import { useSimulationStore } from "@/store/simulationStore";

export function useRealtimeSync() {
  const bootstrapped = useSimulationStore((state) => state.bootstrapped);
  const launched = useSimulationStore((state) => state.launched);
  const initialize = useSimulationStore((state) => state.initialize);
  const refreshSnapshot = useSimulationStore((state) => state.refreshSnapshot);
  const streamTick = useSimulationStore((state) => state.streamTick);

  const initializeRef = useRef(initialize);
  const refreshSnapshotRef = useRef(refreshSnapshot);
  const streamTickRef = useRef(streamTick);

  useEffect(() => {
    initializeRef.current = initialize;
  }, [initialize]);

  useEffect(() => {
    refreshSnapshotRef.current = refreshSnapshot;
  }, [refreshSnapshot]);

  useEffect(() => {
    streamTickRef.current = streamTick;
  }, [streamTick]);

  useEffect(() => {
    if (bootstrapped) {
      return;
    }

    void initializeRef.current();
  }, [bootstrapped]);

  useEffect(() => {
    if (!launched) {
      return;
    }

    const poll = window.setInterval(() => {
      void refreshSnapshotRef.current();
    }, 9000);

    const stream = window.setInterval(() => {
      streamTickRef.current();
    }, 3200);

    return () => {
      window.clearInterval(poll);
      window.clearInterval(stream);
    };
  }, [launched]);
}
