import { useState, type ReactNode } from "react";

import { CyberGridBackground } from "@/components/background/CyberGridBackground";
import { MobileNav } from "@/components/layout/MobileNav";
import { Sidebar } from "@/components/layout/Sidebar";
import { TopBar } from "@/components/layout/TopBar";
import { useSimulationStore } from "@/store/simulationStore";

interface CommandLayoutProps {
  children: ReactNode;
}

export function CommandLayout({ children }: CommandLayoutProps) {
  const [collapsed, setCollapsed] = useState(false);
  const matrixEnabled = useSimulationStore((state) => state.matrixEnabled);

  return (
    <div className="relative flex min-h-screen overflow-hidden">
      <CyberGridBackground matrixEnabled={matrixEnabled} />
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed((value) => !value)} />

      <main className="relative z-10 flex min-h-screen w-full flex-col px-3 pb-24 md:px-5 md:pb-5">
        <TopBar />
        <div className="flex-1">{children}</div>
      </main>

      <MobileNav />
    </div>
  );
}
