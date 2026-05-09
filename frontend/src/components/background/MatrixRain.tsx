import { memo } from "react";

const columns = Array.from({ length: 20 }, (_, index) => ({
  id: index,
  left: `${(index / 20) * 100}%`,
  duration: `${8 + Math.random() * 9}s`,
  delay: `${Math.random() * 6}s`,
  text: Array.from({ length: 28 }, () => (Math.random() > 0.5 ? "1" : "0")).join("")
}));

export const MatrixRain = memo(function MatrixRain() {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden">
      {columns.map((column) => (
        <div
          key={column.id}
          className="matrix-column"
          style={{
            left: column.left,
            animationDuration: column.duration,
            animationDelay: column.delay
          }}
        >
          {column.text}
        </div>
      ))}
    </div>
  );
});
