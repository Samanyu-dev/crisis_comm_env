import { motion } from "framer-motion";
import { ChevronLeft, ChevronRight, Shield } from "lucide-react";
import { NavLink } from "react-router-dom";

import { Button } from "@/components/ui/button";
import { NAV_ITEMS } from "@/lib/constants";
import { cn } from "@/lib/utils";

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

export function Sidebar({ collapsed, onToggle }: SidebarProps) {
  return (
    <motion.aside
      animate={{ width: collapsed ? 82 : 254 }}
      transition={{ type: "spring", stiffness: 240, damping: 28 }}
      className="glass-panel relative hidden h-screen shrink-0 border-r border-slate-700/50 px-3 py-4 md:block"
    >
      <div className="flex items-center justify-between px-2">
        <div className="flex items-center gap-2">
          <span className="rounded-lg border border-cyan-400/40 bg-cyan-500/10 p-2 text-cyan-300">
            <Shield className="h-4 w-4" />
          </span>
          {!collapsed ? (
            <div>
              <p className="text-sm font-semibold tracking-wide">Crisis Command</p>
              <p className="text-[11px] text-slate-400">Environment OS</p>
            </div>
          ) : null}
        </div>
        <Button variant="ghost" size="icon" onClick={onToggle}>
          {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
        </Button>
      </div>

      <nav className="mt-8 space-y-2 px-1">
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) =>
              cn(
                "group relative flex items-center gap-3 rounded-xl border border-transparent px-3 py-2 text-sm text-slate-300 transition-all",
                "hover:border-cyan-400/30 hover:bg-cyan-500/10 hover:text-cyan-100",
                isActive && "border-cyan-400/45 bg-cyan-400/15 text-cyan-100 shadow-neon"
              )
            }
          >
            <item.icon className="h-4 w-4 shrink-0" />
            {!collapsed ? <span>{item.label}</span> : null}
          </NavLink>
        ))}
      </nav>

      {!collapsed ? (
        <div className="absolute bottom-4 left-3 right-3 rounded-xl border border-cyan-400/20 bg-cyan-500/10 p-3 text-xs text-cyan-100">
          <p className="font-semibold">Node Sync Active</p>
          <p className="mt-1 text-cyan-100/80">Realtime mission telemetry connected.</p>
        </div>
      ) : null}
    </motion.aside>
  );
}
