import { AnimatePresence, motion } from "framer-motion";
import { AlertTriangle, CheckCircle2, Info, XCircle } from "lucide-react";
import { useShallow } from "zustand/shallow";

import { cn } from "@/lib/utils";
import { useToastStore } from "@/hooks/useToast";

const toneStyles = {
  default: "border-slate-500/40 bg-slate-900/85 text-slate-100",
  success: "border-emerald-500/40 bg-emerald-950/70 text-emerald-50",
  warning: "border-amber-400/40 bg-amber-950/65 text-amber-100",
  danger: "border-rose-500/40 bg-rose-950/60 text-rose-50"
} as const;

function ToneIcon({ tone = "default" }: { tone?: "default" | "success" | "warning" | "danger" }) {
  if (tone === "success") return <CheckCircle2 className="h-4 w-4" />;
  if (tone === "warning") return <AlertTriangle className="h-4 w-4" />;
  if (tone === "danger") return <XCircle className="h-4 w-4" />;
  return <Info className="h-4 w-4" />;
}

export function Toaster() {
  const { toasts, remove } = useToastStore(
    useShallow((state) => ({
      toasts: state.toasts,
      remove: state.remove
    }))
  );

  return (
    <div className="pointer-events-none fixed right-3 top-3 z-[120] flex w-[min(380px,95vw)] flex-col gap-2">
      <AnimatePresence>
        {toasts.map((toast) => (
          <motion.button
            key={toast.id}
            type="button"
            onClick={() => remove(toast.id)}
            initial={{ opacity: 0, x: 20, y: -8 }}
            animate={{ opacity: 1, x: 0, y: 0 }}
            exit={{ opacity: 0, x: 15 }}
            className={cn(
              "pointer-events-auto rounded-xl border px-3 py-2 text-left shadow-neon",
              toneStyles[toast.tone ?? "default"]
            )}
          >
            <div className="flex items-start gap-2">
              <ToneIcon tone={toast.tone} />
              <div>
                <p className="text-sm font-semibold">{toast.title}</p>
                {toast.description ? <p className="mt-1 text-xs opacity-85">{toast.description}</p> : null}
              </div>
            </div>
          </motion.button>
        ))}
      </AnimatePresence>
    </div>
  );
}
