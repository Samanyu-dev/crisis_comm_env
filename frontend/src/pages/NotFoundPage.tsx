import { Link } from "react-router-dom";

import { Button } from "@/components/ui/button";

export function NotFoundPage() {
  return (
    <div className="flex min-h-[60vh] flex-col items-center justify-center rounded-xl border border-slate-700/50 bg-slate-950/45 p-8 text-center">
      <p className="text-sm text-slate-400">Command route not found.</p>
      <Button asChild className="mt-4">
        <Link to="/dashboard">Return to Dashboard</Link>
      </Button>
    </div>
  );
}
