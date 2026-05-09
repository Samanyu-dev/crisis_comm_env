import { Suspense, lazy, useEffect, type ReactNode } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { Navigate, Route, Routes, useLocation } from "react-router-dom";

import { CommandLayout } from "@/components/layout/CommandLayout";
import { Toaster } from "@/components/ui/toaster";
import { useRealtimeSync } from "@/hooks/useRealtimeSync";
import { useToast } from "@/hooks/useToast";
import { pageTransition } from "@/animations/motionPresets";
import { useSimulationStore } from "@/store/simulationStore";

const LandingPage = lazy(() => import("@/pages/LandingPage").then((module) => ({ default: module.LandingPage })));
const DashboardPage = lazy(() => import("@/pages/DashboardPage").then((module) => ({ default: module.DashboardPage })));
const MissionsPage = lazy(() => import("@/pages/MissionsPage").then((module) => ({ default: module.MissionsPage })));
const SimulationPage = lazy(() => import("@/pages/SimulationPage").then((module) => ({ default: module.SimulationPage })));
const StatePage = lazy(() => import("@/pages/StatePage").then((module) => ({ default: module.StatePage })));
const LogsPage = lazy(() => import("@/pages/LogsPage").then((module) => ({ default: module.LogsPage })));
const AnalyticsPage = lazy(() => import("@/pages/AnalyticsPage").then((module) => ({ default: module.AnalyticsPage })));
const SettingsPage = lazy(() => import("@/pages/SettingsPage").then((module) => ({ default: module.SettingsPage })));
const NotFoundPage = lazy(() => import("@/pages/NotFoundPage").then((module) => ({ default: module.NotFoundPage })));

function PageLoader() {
  return (
    <div className="flex min-h-[55vh] items-center justify-center">
      <div className="glass-panel rounded-xl border border-slate-700/50 px-4 py-3 text-sm text-slate-300">
        Synchronizing command interface...
      </div>
    </div>
  );
}

function AnimatedPage({ children }: { children: ReactNode }) {
  return (
    <motion.div variants={pageTransition} initial="initial" animate="animate" exit="exit">
      {children}
    </motion.div>
  );
}

export default function App() {
  const location = useLocation();
  const { toast } = useToast();
  const { error, clearError } = useSimulationStore((state) => ({
    error: state.error,
    clearError: state.clearError
  }));

  useRealtimeSync();

  useEffect(() => {
    if (!error) {
      return;
    }

    toast({
      title: "Command error",
      description: error,
      tone: "danger"
    });

    const timer = window.setTimeout(() => {
      clearError();
    }, 2400);

    return () => window.clearTimeout(timer);
  }, [error, toast, clearError]);

  return (
    <>
      <AnimatePresence mode="wait">
        <Suspense fallback={<PageLoader />}>
          <Routes location={location} key={location.pathname}>
            <Route
              path="/"
              element={
                <AnimatedPage>
                  <LandingPage />
                </AnimatedPage>
              }
            />

            <Route
              path="/dashboard"
              element={
                <CommandLayout>
                  <AnimatedPage>
                    <DashboardPage />
                  </AnimatedPage>
                </CommandLayout>
              }
            />
            <Route
              path="/missions"
              element={
                <CommandLayout>
                  <AnimatedPage>
                    <MissionsPage />
                  </AnimatedPage>
                </CommandLayout>
              }
            />
            <Route
              path="/simulation"
              element={
                <CommandLayout>
                  <AnimatedPage>
                    <SimulationPage />
                  </AnimatedPage>
                </CommandLayout>
              }
            />
            <Route path="/scenarios" element={<Navigate to="/simulation" replace />} />
            <Route
              path="/state"
              element={
                <CommandLayout>
                  <AnimatedPage>
                    <StatePage />
                  </AnimatedPage>
                </CommandLayout>
              }
            />
            <Route
              path="/logs"
              element={
                <CommandLayout>
                  <AnimatedPage>
                    <LogsPage />
                  </AnimatedPage>
                </CommandLayout>
              }
            />
            <Route
              path="/analytics"
              element={
                <CommandLayout>
                  <AnimatedPage>
                    <AnalyticsPage />
                  </AnimatedPage>
                </CommandLayout>
              }
            />
            <Route
              path="/settings"
              element={
                <CommandLayout>
                  <AnimatedPage>
                    <SettingsPage />
                  </AnimatedPage>
                </CommandLayout>
              }
            />

            <Route
              path="*"
              element={
                <CommandLayout>
                  <AnimatedPage>
                    <NotFoundPage />
                  </AnimatedPage>
                </CommandLayout>
              }
            />
          </Routes>
        </Suspense>
      </AnimatePresence>

      <Toaster />
    </>
  );
}
