import { useEffect } from "react";

import { useSimulationStore } from "@/store/simulationStore";

export function useRealtimeSync() {
  const {
    bootstrapped,
    initialize,
    launched,
    refreshSnapshot,
    streamTick
  } = useSimulationStore((state) => ({
    bootstrapped: state.bootstrapped,
    initialize: state.initialize,
    launched: state.launched,
    refreshSnapshot: state.refreshSnapshot,
    streamTick: state.streamTick
  }));

  useEffect(() => {
    if (!bootstrapped) {
      void initialize();
    }
  }, [bootstrapped, initialize]);

  useEffect(() => {
    if (!launched) {
      return;
    }

    const poll = window.setInterval(() => {
      void refreshSnapshot();
    }, 9000);

    const stream = window.setInterval(() => {
      streamTick();
    }, 3200);

    return () => {
      window.clearInterval(poll);
      window.clearInterval(stream);
    };
  }, [launched, refreshSnapshot, streamTick]);
}
