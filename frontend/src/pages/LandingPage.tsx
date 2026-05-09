import { motion } from "framer-motion";
import { BarChart3, ShieldCheck, Timer } from "lucide-react";
import { useNavigate } from "react-router-dom";
import { useShallow } from "zustand/shallow";

import { HeroSection } from "@/components/landing/HeroSection";
import { useSimulationStore } from "@/store/simulationStore";
import { fadeInUp, staggerChildren } from "@/animations/motionPresets";

export function LandingPage() {
  const navigate = useNavigate();
  const { tasks, health, lastReward, launchSimulation } = useSimulationStore(
    useShallow((state) => ({
      tasks: state.tasks,
      health: state.health,
      lastReward: state.lastReward,
      launchSimulation: state.launchSimulation
    }))
  );

  return (
    <div className="relative mx-auto max-w-7xl space-y-6 px-4 py-6 md:px-8 md:py-10">
      <HeroSection
        activeScenarios={tasks.length}
        apiStatus={health?.status ?? "initializing"}
        accuracy={lastReward || 0.71}
        onLaunch={() => {
          launchSimulation();
          navigate("/dashboard");
        }}
      />

      <motion.section
        variants={staggerChildren}
        initial="hidden"
        animate="visible"
        className="grid gap-4 md:grid-cols-3"
      >
        {[
          {
            icon: ShieldCheck,
            title: "Operational Integrity",
            description: "Cross-audience consistency tracking with deadline monitoring and legal safeguards."
          },
          {
            icon: Timer,
            title: "Realtime Simulation Pulse",
            description: "Turn-based escalation stream with mission progression telemetry and synthetic stress tests."
          },
          {
            icon: BarChart3,
            title: "Decision Analytics",
            description: "Threat progression, sentiment impact, containment efficiency, and replayable action timelines."
          }
        ].map((item) => (
          <motion.div key={item.title} variants={fadeInUp} className="glass-panel rounded-xl border border-slate-700/55 p-5">
            <item.icon className="h-5 w-5 text-cyan-300" />
            <h3 className="mt-3 text-lg font-semibold text-slate-100">{item.title}</h3>
            <p className="mt-2 text-sm text-slate-400">{item.description}</p>
          </motion.div>
        ))}
      </motion.section>
    </div>
  );
}
