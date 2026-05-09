import { create } from "zustand";

export interface ToastItem {
  id: string;
  title: string;
  description?: string;
  tone?: "default" | "success" | "warning" | "danger";
}

interface ToastState {
  toasts: ToastItem[];
  push: (toast: Omit<ToastItem, "id">) => void;
  remove: (id: string) => void;
}

export const useToastStore = create<ToastState>((set, get) => ({
  toasts: [],
  push: (toast) => {
    const id = crypto.randomUUID();
    set((state) => ({
      toasts: [...state.toasts, { id, ...toast }]
    }));

    window.setTimeout(() => {
      get().remove(id);
    }, 3400);
  },
  remove: (id) => {
    set((state) => ({
      toasts: state.toasts.filter((toast) => toast.id !== id)
    }));
  }
}));

export function useToast() {
  const toast = useToastStore((state) => state.push);
  const removeToast = useToastStore((state) => state.remove);
  return { toast, removeToast };
}
