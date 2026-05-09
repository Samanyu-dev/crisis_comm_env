import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#02040a",
        foreground: "#e6f1ff",
        panel: "#0b1120",
        "panel-muted": "#111827",
        neon: "#32d3ff",
        "neon-soft": "#2a9dff",
        "risk-low": "#10b981",
        "risk-mid": "#facc15",
        "risk-high": "#fb923c",
        "risk-critical": "#ef4444"
      },
      fontFamily: {
        display: ["Space Grotesk", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"]
      },
      boxShadow: {
        neon: "0 0 30px rgba(50, 211, 255, 0.25)",
        alert: "0 0 24px rgba(239, 68, 68, 0.35)"
      },
      keyframes: {
        pulseGlow: {
          "0%, 100%": { opacity: "0.4", transform: "scale(1)" },
          "50%": { opacity: "1", transform: "scale(1.05)" }
        },
        scan: {
          "0%": { transform: "rotate(0deg)" },
          "100%": { transform: "rotate(360deg)" }
        },
        shimmer: {
          "0%": { backgroundPosition: "-400px 0" },
          "100%": { backgroundPosition: "400px 0" }
        },
        matrixDrift: {
          "0%": { transform: "translateY(-100%)" },
          "100%": { transform: "translateY(100%)" }
        }
      },
      animation: {
        pulseGlow: "pulseGlow 2.2s ease-in-out infinite",
        scan: "scan 7s linear infinite",
        shimmer: "shimmer 2.2s linear infinite",
        matrixDrift: "matrixDrift 12s linear infinite"
      },
      backgroundImage: {
        "cyber-grid":
          "linear-gradient(rgba(50, 211, 255, 0.08) 1px, transparent 1px), linear-gradient(90deg, rgba(50, 211, 255, 0.08) 1px, transparent 1px)"
      }
    }
  },
  plugins: []
};

export default config;
