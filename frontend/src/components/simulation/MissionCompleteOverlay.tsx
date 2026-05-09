import { motion, AnimatePresence } from "framer-motion";
import { CheckCircle2, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";

interface MissionCompleteOverlayProps {
  open: boolean;
  onClose: () => void;
}

export function MissionCompleteOverlay({ open, onClose }: MissionCompleteOverlayProps) {
  return (
    <AnimatePresence>
      {open ? (
        <motion.div
          className="fixed inset-0 z-[90] flex items-center justify-center bg-black/70 backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
        >
          <motion.div
            initial={{ scale: 0.9, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            exit={{ scale: 0.95, opacity: 0 }}
            className="glass-panel neon-border mx-4 w-full max-w-xl rounded-2xl border border-cyan-400/35 p-8 text-center"
          >
            <motion.div
              animate={{ rotate: [0, 4, -4, 0] }}
              transition={{ duration: 1.6, repeat: Infinity }}
              className="mx-auto mb-4 inline-flex rounded-full border border-emerald-400/40 bg-emerald-500/15 p-4"
            >
              <CheckCircle2 className="h-8 w-8 text-emerald-300" />
            </motion.div>
            <h3 className="text-2xl font-semibold text-white">Mission Complete</h3>
            <p className="mt-2 text-sm text-slate-300">
              Scenario sequence resolved. Debrief and replay timeline are now available.
            </p>
            <Button className="mt-6" onClick={onClose}>
              <Sparkles className="mr-2 h-4 w-4" /> Continue to Debrief
            </Button>
          </motion.div>
        </motion.div>
      ) : null}
    </AnimatePresence>
  );
}
