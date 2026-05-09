import {
  Activity,
  BarChart3,
  Command,
  FileWarning,
  Gauge,
  LayoutDashboard,
  ListChecks,
  Settings,
  ShieldAlert,
  type LucideIcon
} from "lucide-react";

import { StepActionType } from "@/types/api";

export interface NavSection {
  label: string;
  path: string;
  icon: LucideIcon;
}

export const NAV_ITEMS: NavSection[] = [
  { label: "Dashboard", path: "/dashboard", icon: LayoutDashboard },
  { label: "Active Missions", path: "/missions", icon: ListChecks },
  { label: "Crisis Scenarios", path: "/simulation", icon: ShieldAlert },
  { label: "Simulation State", path: "/state", icon: Gauge },
  { label: "Event Logs", path: "/logs", icon: FileWarning },
  { label: "Analytics", path: "/analytics", icon: BarChart3 },
  { label: "Settings", path: "/settings", icon: Settings }
];

export const MOBILE_NAV_ITEMS: NavSection[] = [
  { label: "Dashboard", path: "/dashboard", icon: LayoutDashboard },
  { label: "Missions", path: "/missions", icon: ListChecks },
  { label: "Sim", path: "/simulation", icon: Command },
  { label: "State", path: "/state", icon: Activity }
];

export const ACTION_BUTTONS: Array<{ id: StepActionType; label: string; tone: string }> = [
  { id: "respond", label: "Respond", tone: "bg-sky-500/20 text-sky-200" },
  { id: "escalate", label: "Escalate", tone: "bg-rose-500/20 text-rose-200" },
  { id: "contain", label: "Contain", tone: "bg-emerald-500/20 text-emerald-200" },
  { id: "notify_public", label: "Notify Public", tone: "bg-indigo-500/20 text-indigo-200" },
  { id: "investigate", label: "Investigate", tone: "bg-cyan-500/20 text-cyan-200" },
  { id: "rollback", label: "Rollback", tone: "bg-orange-500/20 text-orange-200" },
  { id: "deploy_team", label: "Deploy Team", tone: "bg-fuchsia-500/20 text-fuchsia-200" }
];
