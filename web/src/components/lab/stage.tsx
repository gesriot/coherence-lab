import { useEffect, useRef } from "react";
import type { ViewState } from "@/lib/lab/types";
import { WORLD } from "@/lib/lab/math";

interface StageProps {
  view: ViewState;
  showTruth: boolean;
  showHypotheses: boolean;
  onPick: (x: number, y: number) => void;
}

function token(name: string): string {
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value;
}

export function Stage({ view, showTruth, showHypotheses, onPick }: StageProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const mapRef = useRef({ ox: 0, oy: 0, scale: 1 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const parent = canvas.parentElement;
    if (!parent) return;

    const draw = () => {
      const width = parent.clientWidth;
      const height = parent.clientHeight;
      if (width < 2 || height < 2) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      const ctx = canvas.getContext("2d");
      if (!ctx) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const bg = token("--color-bg") || "#101410";
      const line = token("--color-line") || "#314038";
      const fg = token("--color-fg") || "#ebe6da";
      const accent = token("--color-accent") || "#e0a15a";
      const muted = token("--color-muted") || "#9aa396";

      ctx.fillStyle = bg;
      ctx.fillRect(0, 0, width, height);

      const pad = 18;
      const scale = Math.min((width - pad * 2) / WORLD, (height - pad * 2) / WORLD);
      const ox = (width - WORLD * scale) / 2;
      const oy = (height - WORLD * scale) / 2;
      mapRef.current = { ox, oy, scale };
      const X = (x: number) => ox + x * scale;
      const Y = (y: number) => oy + y * scale;

      ctx.strokeStyle = line;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.7;
      for (let g = 0; g <= WORLD; g += 20) {
        ctx.beginPath();
        ctx.moveTo(X(g), Y(0));
        ctx.lineTo(X(g), Y(WORLD));
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(X(0), Y(g));
        ctx.lineTo(X(WORLD), Y(g));
        ctx.stroke();
      }
      ctx.globalAlpha = 1;

      const occ = view.occluder;
      ctx.save();
      ctx.beginPath();
      ctx.rect(X(occ.x), Y(occ.y), occ.w * scale, occ.h * scale);
      ctx.clip();
      ctx.fillStyle = muted;
      ctx.globalAlpha = 0.28;
      ctx.fillRect(X(occ.x), Y(occ.y), occ.w * scale, occ.h * scale);
      ctx.globalAlpha = 0.55;
      ctx.strokeStyle = fg;
      ctx.beginPath();
      const span = occ.w * scale + occ.h * scale;
      for (let i = -span; i < span; i += 8) {
        ctx.moveTo(X(occ.x) + i, Y(occ.y));
        ctx.lineTo(X(occ.x) + i + occ.h * scale, Y(occ.y) + occ.h * scale);
      }
      ctx.stroke();
      ctx.restore();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = accent;
      ctx.lineWidth = 1;
      ctx.strokeRect(X(occ.x) + 0.5, Y(occ.y) + 0.5, occ.w * scale, occ.h * scale);

      if (showTruth) {
        for (const object of view.objects) {
          const hidden = view.evaluator.objects.find((item) => item.id === object.id)?.occluded;
          ctx.save();
          ctx.globalAlpha = hidden ? 0.35 : 1;
          ctx.fillStyle = `hsl(${object.hue} 68% 52%)`;
          ctx.strokeStyle = fg;
          ctx.lineWidth = 1.25;
          const cx = X(object.x);
          const cy = Y(object.y);
          const r = object.r * scale;
          ctx.beginPath();
          if (object.shape === "disk") {
            ctx.arc(cx, cy, r, 0, Math.PI * 2);
          } else if (object.shape === "square") {
            ctx.rect(cx - r, cy - r, r * 2, r * 2);
          } else {
            ctx.moveTo(cx, cy - r);
            ctx.lineTo(cx + r, cy + r * 0.75);
            ctx.lineTo(cx - r, cy + r * 0.75);
            ctx.closePath();
          }
          ctx.fill();
          ctx.stroke();
          ctx.globalAlpha = 1;
          ctx.fillStyle = fg;
          ctx.font = "12px Outfit, sans-serif";
          ctx.fillText(object.id, cx + r + 4, cy - 2);
          ctx.restore();
        }
      } else {
        const gaze = view.obsGaze;
        const n = view.grid;
        const cell = (gaze.aperture * 2) / n;
        for (let row = 0; row < n; row++) {
          for (let col = 0; col < n; col++) {
            const sample = view.cells[row * n + col];
            if (!sample || sample.kind === "empty") continue;
            const wx = gaze.x - gaze.aperture + col * cell;
            const wy = gaze.y - gaze.aperture + row * cell;
            ctx.fillStyle =
              sample.kind === "occluder" ? "#3a342c" : `hsl(${sample.hue} 70% 50%)`;
            ctx.fillRect(X(wx), Y(wy), Math.ceil(cell * scale), Math.ceil(cell * scale));
          }
        }
      }

      if (showHypotheses) {
        for (const track of view.tracks) {
          ctx.save();
          ctx.beginPath();
          ctx.setLineDash([4, 3]);
          ctx.lineWidth = track.inWorkspace ? 1.75 : 1;
          ctx.strokeStyle = `hsl(${track.hue} 70% 62%)`;
          ctx.globalAlpha = 0.35 + track.confidence * 0.65;
          ctx.arc(X(track.x), Y(track.y), 5.2 * scale, 0, Math.PI * 2);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.font = "11px 'IBM Plex Mono', monospace";
          ctx.fillStyle = fg;
          ctx.fillText(`T${track.id}`, X(track.x) + 8, Y(track.y) - 6);
          ctx.restore();
        }
      }

      const gaze = view.obsGaze;
      ctx.strokeStyle = accent;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(
        X(gaze.x - gaze.aperture),
        Y(gaze.y - gaze.aperture),
        gaze.aperture * 2 * scale,
        gaze.aperture * 2 * scale,
      );
      ctx.beginPath();
      ctx.moveTo(X(gaze.x) - 6, Y(gaze.y));
      ctx.lineTo(X(gaze.x) + 6, Y(gaze.y));
      ctx.moveTo(X(gaze.x), Y(gaze.y) - 6);
      ctx.lineTo(X(gaze.x), Y(gaze.y) + 6);
      ctx.stroke();

      const next = view.nextGaze;
      ctx.strokeStyle = muted;
      ctx.setLineDash([2, 3]);
      ctx.beginPath();
      ctx.arc(X(next.x), Y(next.y), 3.5, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(parent);
    return () => observer.disconnect();
  }, [view, showTruth, showHypotheses]);

  return (
    <canvas
      ref={canvasRef}
      className="h-full w-full touch-manipulation"
      aria-label="Поле мира: истина для оценщика и гипотезы агента"
      onClick={(event) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const rect = canvas.getBoundingClientRect();
        const { ox, oy, scale } = mapRef.current;
        if (scale <= 0) return;
        const x = (event.clientX - rect.left - ox) / scale;
        const y = (event.clientY - rect.top - oy) / scale;
        if (x < 0 || y < 0 || x > WORLD || y > WORLD) return;
        onPick(x, y);
      }}
    />
  );
}
