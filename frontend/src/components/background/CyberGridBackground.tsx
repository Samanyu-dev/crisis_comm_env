import { motion } from "framer-motion";

import { MatrixRain } from "@/components/background/MatrixRain";

interface CyberGridBackgroundProps {
  matrixEnabled?: boolean;
}

export function CyberGridBackground({ matrixEnabled = false }: CyberGridBackgroundProps) {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      <div className="cyber-grid absolute inset-0 opacity-50" />
      <motion.div
        className="absolute -left-16 top-12 h-72 w-72 rounded-full bg-cyan-500/20 blur-[90px]"
        animate={{ scale: [1, 1.1, 1], opacity: [0.4, 0.7, 0.4] }}
        transition={{ duration: 7, repeat: Infinity }}
      />
      <motion.div
        className="absolute -right-10 bottom-4 h-80 w-80 rounded-full bg-fuchsia-500/18 blur-[100px]"
        animate={{ scale: [1, 1.08, 1], opacity: [0.45, 0.72, 0.45] }}
        transition={{ duration: 9, repeat: Infinity, delay: 0.8 }}
      />
      <motion.div
        className="absolute left-1/2 top-1/2 h-[520px] w-[520px] -translate-x-1/2 -translate-y-1/2 rounded-full border border-cyan-400/10"
        animate={{ rotate: 360 }}
        transition={{ duration: 42, ease: "linear", repeat: Infinity }}
      />
      {matrixEnabled ? <MatrixRain /> : null}
    </div>
  );
}
