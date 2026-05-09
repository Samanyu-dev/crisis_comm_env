import React, { Component, ReactNode } from "react";

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  constructor(props: Props) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo): void {
    console.error("ErrorBoundary caught error:", error);
    console.error("Error info:", errorInfo);
  }

  render(): ReactNode {
    if (this.state.hasError) {
      return (
        <div className="flex min-h-screen items-center justify-center bg-slate-950 p-4">
          <div className="max-w-md rounded-lg border border-red-500/30 bg-slate-900/50 p-6 backdrop-blur-sm">
            <div className="mb-4">
              <h1 className="text-xl font-bold text-red-400">Command System Error</h1>
              <p className="mt-2 text-sm text-slate-400">
                The interface encountered an unexpected error. Please refresh the page.
              </p>
            </div>
            {this.state.error && (
              <div className="mb-4 rounded bg-slate-950 p-3 font-mono text-xs text-red-300">
                {this.state.error.message}
              </div>
            )}
            <button
              onClick={() => window.location.reload()}
              className="w-full rounded bg-slate-700 px-4 py-2 text-sm font-medium text-slate-100 hover:bg-slate-600"
            >
              Refresh Page
            </button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}
