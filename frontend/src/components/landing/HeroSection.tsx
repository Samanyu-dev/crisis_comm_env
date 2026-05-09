import { motion } from "framer-motion";
import { Activity, GaugeCircle, ShieldAlert, Sparkles } from "lucide-react";

import { RadarScanner } from "@/components/background/RadarScanner";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { fadeInUp, staggerChildren } from "@/animations/motionPresets";

interface HeroSectionProps {
  activeScenarios: number;
  apiStatus: string;
  accuracy: number;
  onLaunch: () => void;
}

export function HeroSection({ activeScenarios, apiStatus, accuracy, onLaunch }: HeroSectionProps) {
  return (
    <motion.section
      variants={staggerChildren}
      initial="hidden"
      animate="visible"
      className="relative overflow-hidden rounded-2xl border border-slate-700/55 bg-slate-950/55 p-6 md:p-10"
    >
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_right,rgba(34,211,238,0.20),transparent_45%),radial-gradient(circle_at_bottom_left,rgba(168,85,247,0.22),transparent_40%)]" />
      <div className="relative z-10 grid items-center gap-8 lg:grid-cols-[1.2fr_0.8fr]">
        <div>
          <motion.div variants={fadeInUp} className="inline-flex rounded-full border border-cyan-400/35 bg-cyan-500/10 px-3 py-1 text-xs text-cyan-100">
            Emergency Operations Command Stack
          </motion.div>
          <motion.h1 variants={fadeInUp} className="mt-4 text-3xl font-semibold tracking-tight text-white md:text-5xl">
            Crisis Command Environment
          </motion.h1>
          <motion.p variants={fadeInUp} className="mt-3 max-w-2xl text-sm text-slate-300 md:text-base">
            AI-powered multi-scenario crisis response simulator
          </motion.p>

          <motion.div variants={fadeInUp} className="mt-6 flex flex-wrap gap-2">
            <Badge variant={apiStatus === "ok" ? "success" : "danger"}>
              <Activity className="mr-1 h-3 w-3" /> API {apiStatus}
            </Badge>
            <Badge variant="secondary">
              <ShieldAlert className="mr-1 h-3 w-3" /> Active Scenarios {activeScenarios}
            </Badge>
            <Badge variant="default">
              <GaugeCircle className="mr-1 h-3 w-3" /> Response Accuracy {Math.round(accuracy * 100)}%
            </Badge>
          </motion.div>

          <motion.div variants={fadeInUp} className="mt-8">
            <Button
              size="lg"
              className="relative overflow-hidden bg-gradient-to-r from-cyan-500/35 via-blue-500/35 to-fuchsia-500/35 text-white"
              onClick={onLaunch}
            >
              <span className="absolute inset-0 animate-pulse bg-white/10" />
              <span className="relative inline-flex items-center gap-2">
                <Sparkles className="h-4 w-4" /> Launch Simulation
              </span>
            </Button>
          </motion.div>
        </div>

        <motion.div variants={fadeInUp} className="grid justify-items-center gap-4">
          <RadarScanner />
          <div className="grid w-full grid-cols-3 gap-2 text-center text-xs">
            <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/10 p-2">
              <p className="font-semibold text-cyan-100">{activeScenarios}</p>
              <p className="text-slate-400">Scenarios</p>
            </div>
            <div className="rounded-xl border border-fuchsia-500/20 bg-fuchsia-500/10 p-2">
              <p className="font-semibold text-fuchsia-100">{Math.round(accuracy * 100)}%</p>
              <p className="text-slate-400">AI Score</p>
            </div>
            <div className="rounded-xl border border-emerald-500/20 bg-emerald-500/10 p-2">
              <p className="font-semibold text-emerald-100">LIVE</p>
              <p className="text-slate-400">Pulse</p>
            </div>
          </div>
        </motion.div>
      </div>
    </motion.section>
  );
}
