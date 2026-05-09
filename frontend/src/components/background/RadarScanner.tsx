import { motion } from "framer-motion";

export function RadarScanner() {
  return (
    <div className="relative aspect-square w-full max-w-[260px] overflow-hidden rounded-full border border-cyan-400/25 bg-slate-950/70">
      <div className="absolute inset-4 radar-ring" />
      <div className="absolute inset-10 radar-ring" />
      <div className="absolute inset-16 radar-ring" />
      <motion.div
        className="absolute inset-0 origin-bottom bg-gradient-to-t from-cyan-400/30 via-cyan-300/8 to-transparent"
        animate={{ rotate: 360 }}
        transition={{ duration: 4.2, ease: "linear", repeat: Infinity }}
      />
      <div className="absolute left-1/2 top-1/2 h-3 w-3 -translate-x-1/2 -translate-y-1/2 rounded-full bg-cyan-300 shadow-neon" />
    </div>
  );
}
