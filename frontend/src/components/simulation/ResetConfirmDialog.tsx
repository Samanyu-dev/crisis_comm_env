import { Loader2, RotateCcw } from "lucide-react";

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

interface ResetConfirmDialogProps {
  onConfirm: () => void;
  loading?: boolean;
}

export function ResetConfirmDialog({ onConfirm, loading = false }: ResetConfirmDialogProps) {
  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button variant="destructive">
          <RotateCcw className="mr-2 h-4 w-4" /> Reset Simulation
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Reset mission state?</DialogTitle>
          <DialogDescription>
            This will clear turn progression and restart the current scenario from turn 1.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button variant="destructive" onClick={onConfirm} disabled={loading}>
            {loading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : null}
            Confirm Reset
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
