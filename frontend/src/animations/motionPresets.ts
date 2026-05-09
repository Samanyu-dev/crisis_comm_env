import { Variants } from "framer-motion";

export const fadeInUp: Variants = {
  hidden: { opacity: 0, y: 18 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.45, ease: "easeOut" }
  }
};

export const staggerChildren: Variants = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.08,
      delayChildren: 0.08
    }
  }
};

export const pageTransition: Variants = {
  initial: { opacity: 0, scale: 0.985 },
  animate: {
    opacity: 1,
    scale: 1,
    transition: {
      duration: 0.38,
      ease: "easeOut"
    }
  },
  exit: {
    opacity: 0,
    scale: 0.99,
    transition: { duration: 0.2 }
  }
};

export const pulse = {
  animate: {
    scale: [1, 1.04, 1],
    opacity: [0.5, 1, 0.5],
    transition: {
      duration: 2.2,
      repeat: Infinity,
      ease: "easeInOut"
    }
  }
};
