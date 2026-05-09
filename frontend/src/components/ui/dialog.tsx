import * as DialogPrimitive from "@radix-ui/react-dialog";
import type { HTMLAttributes } from "react";
import { X } from "lucide-react";

import { cn } from "@/lib/utils";

const Dialog = DialogPrimitive.Root;
const DialogTrigger = DialogPrimitive.Trigger;
const DialogPortal = DialogPrimitive.Portal;
const DialogClose = DialogPrimitive.Close;

const DialogOverlay = ({ className, ...props }: DialogPrimitive.DialogOverlayProps) => (
  <DialogPrimitive.Overlay
    className={cn("fixed inset-0 z-50 bg-black/70 backdrop-blur-sm", className)}
    {...props}
  />
);

const DialogContent = ({ className, children, ...props }: DialogPrimitive.DialogContentProps) => (
  <DialogPortal>
    <DialogOverlay />
    <DialogPrimitive.Content
      className={cn(
        "fixed left-1/2 top-1/2 z-50 w-[95vw] max-w-lg -translate-x-1/2 -translate-y-1/2 rounded-xl border border-slate-700/50 bg-slate-950/95 p-6 shadow-neon",
        className
      )}
      {...props}
    >
      {children}
      <DialogClose className="absolute right-3 top-3 rounded-sm text-slate-400 transition-opacity hover:text-slate-100 focus:outline-none">
        <X className="h-4 w-4" />
      </DialogClose>
    </DialogPrimitive.Content>
  </DialogPortal>
);

const DialogHeader = ({ className, ...props }: HTMLAttributes<HTMLDivElement>) => (
  <div className={cn("flex flex-col space-y-2 text-left", className)} {...props} />
);

const DialogFooter = ({ className, ...props }: HTMLAttributes<HTMLDivElement>) => (
  <div className={cn("mt-6 flex flex-col-reverse gap-2 sm:flex-row sm:justify-end", className)} {...props} />
);

const DialogTitle = ({ className, ...props }: DialogPrimitive.DialogTitleProps) => (
  <DialogPrimitive.Title className={cn("text-lg font-semibold", className)} {...props} />
);

const DialogDescription = ({ className, ...props }: DialogPrimitive.DialogDescriptionProps) => (
  <DialogPrimitive.Description className={cn("text-sm text-slate-400", className)} {...props} />
);

export {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger
};
