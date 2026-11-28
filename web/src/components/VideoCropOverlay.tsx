import { useEffect, useMemo, useRef, useState } from "react";
import type { RefObject, SyntheticEvent } from "react";

export interface NormalizedCropRect {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface Props {
  videoRef: RefObject<HTMLVideoElement | null>;
  mediaSrc: string;
  playMode: "source" | "rendered";
  sourceWidth: number;
  sourceHeight: number;
  effectiveCrop: NormalizedCropRect | null;
  editing: boolean;
  draftRect: NormalizedCropRect;
  previewCroppedOnly: boolean;
  onDraftChange: (rect: NormalizedCropRect) => void;
  onTimeUpdate: () => void;
  onLoadedMetadata?: (e: SyntheticEvent<HTMLVideoElement>) => void;
}

type DragKind = "draw" | "move" | "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";

interface DragState {
  kind: DragKind;
  startClientX: number;
  startClientY: number;
  startNormX: number;
  startNormY: number;
  startRect: NormalizedCropRect;
  boxW: number;
  boxH: number;
}

const MIN_RATIO = 0.05;

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function round4(v: number): number {
  return Math.round(v * 10000) / 10000;
}

export function sanitizeRect(r: NormalizedCropRect): NormalizedCropRect {
  const x = clamp(r.x, 0, 1 - MIN_RATIO);
  const y = clamp(r.y, 0, 1 - MIN_RATIO);
  const w = clamp(r.w, MIN_RATIO, 1 - x);
  const h = clamp(r.h, MIN_RATIO, 1 - y);
  return { x: round4(x), y: round4(y), w: round4(w), h: round4(h) };
}

export function computeEvenOutputDims(
  rect: NormalizedCropRect,
  srcW: number,
  srcH: number
): { px: number; py: number; pw: number; ph: number } {
  const vw = Math.max(16, Math.floor(srcW) & ~1);
  const vh = Math.max(16, Math.floor(srcH) & ~1);
  const sr = sanitizeRect(rect);
  const px = Math.max(0, Math.min(vw - 16, Math.round(sr.x * vw) & ~1));
  const py = Math.max(0, Math.min(vh - 16, Math.round(sr.y * vh) & ~1));
  const pw = Math.max(16, Math.min(vw - px, (Math.round(sr.w * vw) + 1) & ~1));
  const ph = Math.max(16, Math.min(vh - py, (Math.round(sr.h * vh) + 1) & ~1));
  return { px, py, pw, ph };
}

function fmtScrubberTime(sec: number): string {
  if (!Number.isFinite(sec) || sec < 0) return "0:00";
  const s = Math.floor(sec);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const rem = s % 60;
  if (h > 0) return `${h}:${String(m).padStart(2, "0")}:${String(rem).padStart(2, "0")}`;
  return `${m}:${String(rem).padStart(2, "0")}`;
}

export default function VideoCropOverlay({
  videoRef,
  mediaSrc,
  playMode,
  sourceWidth,
  sourceHeight,
  effectiveCrop,
  editing,
  draftRect,
  previewCroppedOnly,
  onDraftChange,
  onTimeUpdate,
  onLoadedMetadata,
}: Props) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const overlayRef = useRef<HTMLDivElement | null>(null);
  const dragRef = useRef<DragState | null>(null);

  const [stageSize, setStageSize] = useState({ w: 640, h: 360 });
  const [natSize, setNatSize] = useState({
    w: sourceWidth > 0 ? sourceWidth : 1920,
    h: sourceHeight > 0 ? sourceHeight : 1080,
  });
  const [paused, setPaused] = useState(true);
  const [muted, setMuted] = useState(false);
  const [curTime, setCurTime] = useState(0);
  const [duration, setDuration] = useState(0);

  useEffect(() => {
    setNatSize({
      w: sourceWidth > 0 ? sourceWidth : 1920,
      h: sourceHeight > 0 ? sourceHeight : 1080,
    });
  }, [sourceWidth, sourceHeight]);

  useEffect(() => {
    const el = stageRef.current;
    if (!el) return;
    const update = () => {
      const r = el.getBoundingClientRect();
      if (r.width > 16 && r.height > 16) {
        setStageSize({ w: r.width, h: r.height });
      }
    };
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const isSource = playMode === "source";
  const showEditMode = isSource && editing;
  const showCroppedPreview = isSource && !editing && effectiveCrop !== null && previewCroppedOnly;
  const showFullWithOutline = isSource && !editing && effectiveCrop !== null && !previewCroppedOnly;
  const hideNativeControls = showEditMode || showCroppedPreview;

  const fullFit = useMemo(() => {
    const nw = Math.max(16, natSize.w);
    const nh = Math.max(16, natSize.h);
    const scale = Math.min(stageSize.w / nw, stageSize.h / nh);
    return {
      w: Math.max(32, Math.floor(nw * scale)),
      h: Math.max(32, Math.floor(nh * scale)),
    };
  }, [stageSize.w, stageSize.h, natSize.w, natSize.h]);

  const croppedLayout = useMemo(() => {
    if (!effectiveCrop) return null;
    const nw = Math.max(16, natSize.w);
    const nh = Math.max(16, natSize.h);
    const cw = Math.max(16, effectiveCrop.w * nw);
    const ch = Math.max(16, effectiveCrop.h * nh);
    const scale = Math.min(stageSize.w / cw, stageSize.h / ch);
    return {
      boxW: Math.max(32, Math.round(cw * scale)),
      boxH: Math.max(32, Math.round(ch * scale)),
      videoW: Math.round(nw * scale),
      videoH: Math.round(nh * scale),
      offsetX: -Math.round(effectiveCrop.x * nw * scale),
      offsetY: -Math.round(effectiveCrop.y * nh * scale),
    };
  }, [effectiveCrop, stageSize.w, stageSize.h, natSize.w, natSize.h]);

  const activeBoxRect = showEditMode ? sanitizeRect(draftRect) : effectiveCrop;
  const liveOutputDims = useMemo(() => {
    if (!activeBoxRect) return null;
    return computeEvenOutputDims(activeBoxRect, natSize.w, natSize.h);
  }, [activeBoxRect, natSize.w, natSize.h]);

  function startPointerDrag(kind: DragKind, e: React.PointerEvent<HTMLElement>) {
    if (!showEditMode || !overlayRef.current) return;
    e.preventDefault();
    e.stopPropagation();
    const bounds = overlayRef.current.getBoundingClientRect();
    if (bounds.width < 10 || bounds.height < 10) return;
    const normX = clamp((e.clientX - bounds.left) / bounds.width, 0, 1);
    const normY = clamp((e.clientY - bounds.top) / bounds.height, 0, 1);
    dragRef.current = {
      kind,
      startClientX: e.clientX,
      startClientY: e.clientY,
      startNormX: normX,
      startNormY: normY,
      startRect: sanitizeRect(draftRect),
      boxW: bounds.width,
      boxH: bounds.height,
    };
    overlayRef.current.setPointerCapture(e.pointerId);
  }

  function handleOverlayPointerMove(e: React.PointerEvent<HTMLDivElement>) {
    const d = dragRef.current;
    if (!d) return;
    const dx = (e.clientX - d.startClientX) / d.boxW;
    const dy = (e.clientY - d.startClientY) / d.boxH;
    const sr = d.startRect;

    if (d.kind === "draw") {
      const curX = clamp(d.startNormX + dx, 0, 1);
      const curY = clamp(d.startNormY + dy, 0, 1);
      const x0 = Math.min(d.startNormX, curX);
      const y0 = Math.min(d.startNormY, curY);
      const x1 = Math.max(d.startNormX, curX);
      const y1 = Math.max(d.startNormY, curY);
      if (x1 - x0 >= 0.015 && y1 - y0 >= 0.015) {
        onDraftChange(
          sanitizeRect({
            x: x0,
            y: y0,
            w: Math.max(MIN_RATIO, x1 - x0),
            h: Math.max(MIN_RATIO, y1 - y0),
          })
        );
      }
      return;
    }

    if (d.kind === "move") {
      const nx = clamp(sr.x + dx, 0, 1 - sr.w);
      const ny = clamp(sr.y + dy, 0, 1 - sr.h);
      onDraftChange({ x: round4(nx), y: round4(ny), w: sr.w, h: sr.h });
      return;
    }

    let left = sr.x;
    let top = sr.y;
    let right = sr.x + sr.w;
    let bottom = sr.y + sr.h;

    if (d.kind.includes("w")) {
      left = clamp(sr.x + dx, 0, right - MIN_RATIO);
    }
    if (d.kind.includes("e")) {
      right = clamp(sr.x + sr.w + dx, left + MIN_RATIO, 1);
    }
    if (d.kind.includes("n")) {
      top = clamp(sr.y + dy, 0, bottom - MIN_RATIO);
    }
    if (d.kind.includes("s")) {
      bottom = clamp(sr.y + sr.h + dy, top + MIN_RATIO, 1);
    }

    onDraftChange(sanitizeRect({ x: left, y: top, w: right - left, h: bottom - top }));
  }

  function handleOverlayPointerUp(e: React.PointerEvent<HTMLDivElement>) {
    if (!dragRef.current) return;
    dragRef.current = null;
    if (overlayRef.current?.hasPointerCapture(e.pointerId)) {
      overlayRef.current.releasePointerCapture(e.pointerId);
    }
  }

  function togglePlayPause() {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) {
      v.play().catch(() => {});
    } else {
      v.pause();
    }
  }

  function toggleMute() {
    const v = videoRef.current;
    if (!v) return;
    v.muted = !v.muted;
    setMuted(v.muted);
  }

  function handleSeek(nextSec: number) {
    const v = videoRef.current;
    if (!v) return;
    v.currentTime = clamp(nextSec, 0, duration || 0);
    setCurTime(v.currentTime);
  }

  const videoNode = (
    <video
      ref={videoRef}
      key={mediaSrc}
      className={showCroppedPreview || showEditMode || showFullWithOutline ? "video-el video-el-fill" : "video-el"}
      src={mediaSrc}
      controls={!hideNativeControls}
      style={
        showCroppedPreview && croppedLayout
          ? {
              position: "absolute",
              left: croppedLayout.offsetX,
              top: croppedLayout.offsetY,
              width: croppedLayout.videoW,
              height: croppedLayout.videoH,
              maxWidth: "none",
              maxHeight: "none",
              borderRadius: 0,
            }
          : undefined
      }
      onLoadedMetadata={(e) => {
        const v = e.currentTarget;
        if (v.videoWidth > 0 && v.videoHeight > 0) {
          setNatSize({ w: v.videoWidth, h: v.videoHeight });
        }
        setDuration(Number.isFinite(v.duration) ? v.duration : 0);
        setCurTime(v.currentTime || 0);
        setPaused(v.paused);
        setMuted(v.muted);
        onLoadedMetadata?.(e);
      }}
      onPlay={() => setPaused(false)}
      onPause={() => setPaused(true)}
      onVolumeChange={(e) => setMuted(e.currentTarget.muted)}
      onTimeUpdate={(e) => {
        setCurTime(e.currentTarget.currentTime);
        onTimeUpdate();
      }}
    />
  );

  const handles: DragKind[] = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];

  return (
    <>
      <div ref={stageRef} className="video-stage">
        {showCroppedPreview && croppedLayout ? (
          <div
            className="crop-preview-viewport"
            style={{ width: croppedLayout.boxW, height: croppedLayout.boxH }}
            onClick={togglePlayPause}
            title="点击播放 / 暂停（当前仅显示裁剪框内画面）"
          >
            {videoNode}
          </div>
        ) : showEditMode || showFullWithOutline ? (
          <div className="crop-stage-box" style={{ width: fullFit.w, height: fullFit.h }}>
            {videoNode}
            {showEditMode && activeBoxRect && (
              <div
                ref={overlayRef}
                className="crop-overlay"
                onPointerDown={(e) => startPointerDrag("draw", e)}
                onPointerMove={handleOverlayPointerMove}
                onPointerUp={handleOverlayPointerUp}
                onPointerCancel={handleOverlayPointerUp}
              >
                <div
                  className="crop-mask-strip"
                  style={{ left: 0, top: 0, width: "100%", height: `${activeBoxRect.y * 100}%` }}
                />
                <div
                  className="crop-mask-strip"
                  style={{
                    left: 0,
                    top: `${(activeBoxRect.y + activeBoxRect.h) * 100}%`,
                    width: "100%",
                    height: `${Math.max(0, 1 - activeBoxRect.y - activeBoxRect.h) * 100}%`,
                  }}
                />
                <div
                  className="crop-mask-strip"
                  style={{
                    left: 0,
                    top: `${activeBoxRect.y * 100}%`,
                    width: `${activeBoxRect.x * 100}%`,
                    height: `${activeBoxRect.h * 100}%`,
                  }}
                />
                <div
                  className="crop-mask-strip"
                  style={{
                    left: `${(activeBoxRect.x + activeBoxRect.w) * 100}%`,
                    top: `${activeBoxRect.y * 100}%`,
                    width: `${Math.max(0, 1 - activeBoxRect.x - activeBoxRect.w) * 100}%`,
                    height: `${activeBoxRect.h * 100}%`,
                  }}
                />
                <div
                  className="crop-selection"
                  style={{
                    left: `${activeBoxRect.x * 100}%`,
                    top: `${activeBoxRect.y * 100}%`,
                    width: `${activeBoxRect.w * 100}%`,
                    height: `${activeBoxRect.h * 100}%`,
                  }}
                  onPointerDown={(e) => startPointerDrag("move", e)}
                >
                  <div className="crop-grid-h" style={{ top: "33.33%" }} />
                  <div className="crop-grid-h" style={{ top: "66.66%" }} />
                  <div className="crop-grid-v" style={{ left: "33.33%" }} />
                  <div className="crop-grid-v" style={{ left: "66.66%" }} />
                  {liveOutputDims && (
                    <div className="crop-badge-dims">
                      {liveOutputDims.pw} × {liveOutputDims.ph} px
                    </div>
                  )}
                  {handles.map((hk) => (
                    <div
                      key={hk}
                      className={`crop-handle crop-handle-${hk}`}
                      onPointerDown={(e) => startPointerDrag(hk, e)}
                    />
                  ))}
                </div>
              </div>
            )}
            {showFullWithOutline && activeBoxRect && (
              <div className="crop-outline-readonly">
                <div
                  className="crop-outline-box"
                  style={{
                    left: `${activeBoxRect.x * 100}%`,
                    top: `${activeBoxRect.y * 100}%`,
                    width: `${activeBoxRect.w * 100}%`,
                    height: `${activeBoxRect.h * 100}%`,
                  }}
                />
              </div>
            )}
          </div>
        ) : (
          videoNode
        )}
      </div>
      {hideNativeControls && (
        <div className="crop-scrubber">
          <button type="button" className="sm" onClick={togglePlayPause}>
            {paused ? "播放" : "暂停"}
          </button>
          <input
            type="range"
            className="crop-scrubber-range"
            min={0}
            max={duration > 0 ? duration : 1}
            step={0.05}
            value={Math.min(curTime, duration || 1)}
            onChange={(e) => handleSeek(Number(e.target.value))}
          />
          <span className="mono crop-scrubber-time">
            {fmtScrubberTime(curTime)} / {fmtScrubberTime(duration)}
          </span>
          <button type="button" className="sm" onClick={toggleMute}>
            {muted ? "取消静音" : "静音"}
          </button>
        </div>
      )}
    </>
  );
}
