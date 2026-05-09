import { useEffect, useState } from "react";

export function useCountUp(target: number, duration = 650) {
  const [value, setValue] = useState(target);

  useEffect(() => {
    const startValue = value;
    const delta = target - startValue;
    if (Math.abs(delta) < 0.01) {
      return;
    }

    const start = performance.now();
    let raf = 0;

    const tick = (now: number) => {
      const progress = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - progress, 3);
      setValue(startValue + delta * eased);

      if (progress < 1) {
        raf = requestAnimationFrame(tick);
      }
    };

    raf = requestAnimationFrame(tick);

    return () => {
      cancelAnimationFrame(raf);
    };
  }, [target]);

  return value;
}
