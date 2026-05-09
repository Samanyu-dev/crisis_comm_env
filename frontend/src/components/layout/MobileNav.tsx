import { NavLink } from "react-router-dom";

import { MOBILE_NAV_ITEMS } from "@/lib/constants";
import { cn } from "@/lib/utils";

export function MobileNav() {
  return (
    <nav className="glass-panel fixed bottom-2 left-2 right-2 z-50 flex items-center justify-around rounded-2xl border border-slate-700/60 p-2 md:hidden">
      {MOBILE_NAV_ITEMS.map((item) => (
        <NavLink
          key={item.path}
          to={item.path}
          className={({ isActive }) =>
            cn(
              "flex min-w-[64px] flex-col items-center gap-1 rounded-xl px-2 py-1 text-[11px] font-medium text-slate-400",
              isActive && "bg-cyan-500/20 text-cyan-100"
            )
          }
        >
          <item.icon className="h-4 w-4" />
          <span>{item.label}</span>
        </NavLink>
      ))}
    </nav>
  );
}
