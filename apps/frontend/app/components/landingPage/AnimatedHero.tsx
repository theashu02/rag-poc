"use client";

import { Button } from "@/components/ui/button";
import * as React from "react";
import { useRouter } from "next/navigation";

type Vec2 = { x: number; y: number };
type Blob = {
  pos: Vec2;
  vel: Vec2;
  target: Vec2;
  color: string;
  radius: number;
  stiffness: number;
  damping: number;
};

function clampDPR(dpr: number) {
  return Math.min(1.75, Math.max(1, dpr || 1));
}

export function InteractiveHero({
  title = "INTELLIGENCE&nbsp;ANALYTICS,&nbsp;FINALLY.",
  subtitle = "A lightweight, accessible glow-field that springs to your pointer and pauses when you leave—clean, performant, and production-ready." 
}: { title?: string; subtitle?: string; }) {
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const containerRef = React.useRef<HTMLDivElement | null>(null);
  const rafRef = React.useRef<number | null>(null);
  const runningRef = React.useRef(false);
  const reducedRef = React.useRef(false);
  const pointerRef = React.useRef<Vec2 | null>(null);
  const blobsRef = React.useRef<Blob[]>([]);
  const sizeRef = React.useRef({ w: 0, h: 0, dpr: 1 });
  const hoverRef = React.useRef(false);
  const router = useRouter();

  // Color system (3-5): primary blue, teal accent, neutrals from theme
  const COLORS = ["#0EA5E9", "#22D3EE", "#7DD3FC"];

  const ensureBlobs = React.useCallback(() => {
    const { w, h } = sizeRef.current;
    if (!w || !h) return;
    const center = { x: w / 2, y: h / 2 };
    const makeBlob = (i: number): Blob => ({
      pos: { x: center.x, y: center.y },
      vel: { x: 0, y: 0 },
      target: { x: center.x, y: center.y },
      color: COLORS[i % COLORS.length],
      radius: 180 - i * 35,
      stiffness: 0.06 - i * 0.007,
      damping: 0.85 - i * 0.03,
    });
    blobsRef.current = [0, 1, 2, 3, 4].map(makeBlob);
  }, []);

  const resize = React.useCallback(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const rect = container.getBoundingClientRect();
    const dpr = clampDPR(window.devicePixelRatio || 1);
    sizeRef.current = {
      w: Math.max(1, Math.floor(rect.width)),
      h: Math.max(1, Math.floor(rect.height)),
      dpr,
    };
    canvas.width = Math.floor(sizeRef.current.w * dpr);
    canvas.height = Math.floor(sizeRef.current.h * dpr);
    canvas.style.width = `${sizeRef.current.w}px`;
    canvas.style.height = `${sizeRef.current.h}px`;
    ensureBlobs();
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.save();
      ctx.scale(dpr, dpr);
      ctx.clearRect(0, 0, sizeRef.current.w, sizeRef.current.h);
      ctx.restore();
    }
  }, [ensureBlobs]);

  const draw = React.useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const { w, h, dpr } = sizeRef.current;

    ctx.save();
    ctx.scale(dpr, dpr);

    // subtle trail
    ctx.globalCompositeOperation = "source-over";
    ctx.fillStyle = "rgba(2, 6, 23, 0.06)";
    ctx.fillRect(0, 0, w, h);

    // additive glow
    ctx.globalCompositeOperation = "lighter";

    const pointer = pointerRef.current;
    const center = { x: w / 2, y: h / 2 };
    const activeTarget = hoverRef.current && pointer ? pointer : center;

    for (const b of blobsRef.current) {
      b.target.x = activeTarget.x;
      b.target.y = activeTarget.y;

      const ax = (b.target.x - b.pos.x) * b.stiffness;
      const ay = (b.target.y - b.pos.y) * b.stiffness;
      b.vel.x = b.vel.x * b.damping + ax;
      b.vel.y = b.vel.y * b.damping + ay;
      b.pos.x += b.vel.x;
      b.pos.y += b.vel.y;

      const grad = ctx.createRadialGradient(
        b.pos.x,
        b.pos.y,
        0,
        b.pos.x,
        b.pos.y,
        b.radius
      );
      grad.addColorStop(0, b.color);
      grad.addColorStop(0.35, `${b.color}CC`);
      grad.addColorStop(1, "rgba(0,0,0,0)");
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(b.pos.x, b.pos.y, b.radius, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.restore();
  }, []);

  const loop = React.useCallback(() => {
    if (!runningRef.current) return;
    draw();
    rafRef.current = requestAnimationFrame(loop);
  }, [draw]);

  const start = React.useCallback(() => {
    if (reducedRef.current) return;
    if (runningRef.current) return;
    runningRef.current = true;
    rafRef.current = requestAnimationFrame(loop);
  }, [loop]);

  const stop = React.useCallback(() => {
    runningRef.current = false;
    if (rafRef.current) {
      cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
    }
  }, []);

  const onPointerMove = React.useCallback(
    (e: React.MouseEvent | React.TouchEvent) => {
      const container = containerRef.current;
      if (!container) return;
      const rect = container.getBoundingClientRect();
      let clientX = 0;
      let clientY = 0;
      if ("touches" in e && e.touches.length) {
        clientX = e.touches[0].clientX;
        clientY = e.touches[0].clientY;
      } else if ("clientX" in e) {
        clientX = (e as React.MouseEvent).clientX;
        clientY = (e as React.MouseEvent).clientY;
      }
      pointerRef.current = { x: clientX - rect.left, y: clientY - rect.top };
    },
    []
  );

  const onEnter = React.useCallback(() => {
    hoverRef.current = true;
    start();
  }, [start]);

  const onLeave = React.useCallback(() => {
    hoverRef.current = false;
    const t = setTimeout(() => {
      if (!hoverRef.current) stop();
    }, 500);
    return () => clearTimeout(t);
  }, [stop]);

  React.useEffect(() => {
    reducedRef.current = window.matchMedia(
      "(prefers-reduced-motion: reduce)"
    ).matches;
    resize();
    if (!reducedRef.current) draw();
    const ro = new ResizeObserver(() => resize());
    if (containerRef.current) ro.observe(containerRef.current);
    return () => {
      ro.disconnect();
      stop();
    };
  }, [draw, resize, stop]);

  const handleClick = () => {
    router.push("/auth");
  }

  return (
    <section
      ref={containerRef}
      className="relative isolate h-screen w-screen overflow-hidden border border-white/10 bg-[rgb(2,6,23)]/95"
      aria-label="Interactive hero"
      onMouseEnter={onEnter}
      onMouseLeave={onLeave}
      onMouseMove={onPointerMove as any}
      onTouchStart={onEnter}
      onTouchEnd={onLeave as any}
      onTouchMove={onPointerMove as any}
    >
      <canvas
        ref={canvasRef}
        aria-hidden="true"
        className="pointer-events-none absolute inset-0 h-full w-full"
      />

      <div className="relative z-10 flex h-full w-full items-center justify-center px-6 py-20 md:px-10 md:py-28">
        <div className="flex flex-col max-w-3xl text-center">
          <h1 className="text-balance text-3xl font-semibold tracking-tight text-white md:text-5xl">
            {title}
          </h1>
          <p className="mt-4 text-pretty text-sm leading-relaxed text-white/80 md:text-base">
            {subtitle}
          </p>

          <div className="mt-8 flex items-center justify-center">
            <Button size="lg" onClick={handleClick} className="mt-6 cursor-pointer bg-amber-200 p-7 text-2xl">
                Try it out
            </Button>
          </div>
        </div>
      </div>

      {/* Subtle vignette */}
      <div
        className="pointer-events-none absolute inset-0 bg-[radial-gradient(1200px_400px_at_50%_120%,rgba(2,6,23,0)_0%,rgba(2,6,23,0.35)_60%,rgba(2,6,23,0.9)_100%)]"
        aria-hidden="true"
      />
    </section>
  );
}
