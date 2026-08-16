import {
  memo,
  useCallback,
  useEffect,
  useMemo,
  type MouseEvent as ReactMouseEvent,
  type WheelEvent as ReactWheelEvent,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import {
  Activity,
  Cpu,
  Download,
  Gauge,
  Gamepad2,
  Grid3x3,
  Layers,
  MonitorCog,
  MoonStar,
  Palette,
  RefreshCcw,
  RotateCcw,
  Shuffle,
  SkipForward,
  Sparkles,
  Sun,
  Timer,
  WandSparkles,
} from "lucide-react";
import { toast, Toaster } from "sonner";

import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Textarea } from "@/components/ui/textarea";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import SweepPanel from "@/components/SweepPanel";
import { cn } from "@/lib/utils";
import {
  fetchBootstrap,
  fetchEnvironments,
  fetchInteractiveRender,
  fetchInteractiveReset,
  fetchInteractiveStart,
  fetchInteractiveStep,
  fetchInteractiveStop,
  fetchRandomize,
  fetchRender,
  getErrorMessage,
  isAbortError,
} from "@/lib/dashboard-api";
import { applyPresetToStore, mergePresetIntoSpecs } from "@/lib/config-apply";
import type {
  EffectEntry,
  EffectSpecsByStage,
  EffectsStore,
  EnvironmentOption,
  HistoryEntry,
  InteractiveBindings,
  JSONValue,
  ParameterSpec,
  RendererStore,
  StageName,
  StatePayload,
  SweepPreset,
} from "@/lib/types";

const STAGE_ORDER: StageName[] = ["PRE_RENDER", "OBJECT_RENDER", "POST_RENDER"];

const STAGE_LABELS: Record<StageName, string> = {
  PRE_RENDER: "Pre-render",
  OBJECT_RENDER: "Object Render",
  POST_RENDER: "Post-render",
};

const HISTORY_LIMIT = 12;
const RENDER_DEBOUNCE_MS = 90;
const THEME_STORAGE_KEY = "spar-datadash-theme";

type ThemePreference = "system" | "light" | "dark";

interface DirectionalActionBindings {
  up?: number;
  down?: number;
  left?: number;
  right?: number;
  noop?: number;
}

interface WheelAxisBindings {
  negative?: number;
  positive?: number;
}

interface PointerGestureState {
  pointerId: number;
  startX: number;
  startY: number;
}

function loadInitialThemePreference(): ThemePreference {
  if (typeof window === "undefined") {
    return "system";
  }
  const raw = window.localStorage.getItem(THEME_STORAGE_KEY);
  return raw === "light" || raw === "dark" || raw === "system" ? raw : "system";
}

function stableStringify(value: unknown): string {
  if (value === null || typeof value !== "object") {
    return JSON.stringify(value);
  }
  if (Array.isArray(value)) {
    return `[${value.map((entry) => stableStringify(entry)).join(",")}]`;
  }
  const typed = value as Record<string, unknown>;
  const keys = Object.keys(typed).sort();
  const body = keys.map((key) => `${JSON.stringify(key)}:${stableStringify(typed[key])}`).join(",");
  return `{${body}}`;
}

function asJsonString(value: JSONValue | undefined): string {
  if (value === undefined) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return "";
  }
}

function numericOrFallback(value: JSONValue | undefined, fallback: number): number {
  if (typeof value === "number" && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return fallback;
}

type RendererControlKind = "number" | "switch" | "json" | "text";

function getRendererControlKind(value: JSONValue): RendererControlKind {
  if (typeof value === "number") {
    return "number";
  }
  if (typeof value === "boolean") {
    return "switch";
  }
  if (Array.isArray(value)) {
    return "json";
  }
  if (value !== null && typeof value === "object") {
    return "json";
  }
  return "text";
}

function parseRendererJsonInput(raw: string): JSONValue {
  const trimmed = raw.trim();
  if (trimmed.length === 0) {
    return "";
  }
  try {
    return JSON.parse(trimmed) as JSONValue;
  } catch {
    return raw;
  }
}

function isTextFieldTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) {
    return false;
  }
  const tag = target.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || target.isContentEditable;
}

function makeHistoryId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

function createInitialRendererValue(value: JSONValue | undefined): string {
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number") {
    return value.toString();
  }
  return "";
}

function normalizeBindingKey(raw: string): string {
  const key = raw.trim().toLowerCase();
  if (key === " " || key === "spacebar") {
    return "space";
  }
  return key;
}

function normalizeActionMap(raw: Record<string, number> | undefined): Record<string, number> {
  if (!raw) {
    return {};
  }
  const normalized: Record<string, number> = {};
  for (const [key, action] of Object.entries(raw)) {
    if (!Number.isInteger(action) || action < 0) {
      continue;
    }
    const normalizedKey = normalizeBindingKey(key);
    if (!normalizedKey) {
      continue;
    }
    normalized[normalizedKey] = action;
  }
  return normalized;
}

function normalizeDirectionalBindings(
  raw: Partial<Record<"up" | "down" | "left" | "right" | "noop", number>> | undefined,
): DirectionalActionBindings {
  if (!raw) {
    return {};
  }
  const next: DirectionalActionBindings = {};
  for (const direction of ["up", "down", "left", "right", "noop"] as const) {
    const action = raw[direction];
    if (typeof action === "number" && Number.isInteger(action) && action >= 0) {
      next[direction] = action;
    }
  }
  return next;
}

function normalizeWheelAxisBindings(raw: Partial<Record<"negative" | "positive", number>> | undefined): WheelAxisBindings {
  if (!raw) {
    return {};
  }
  const next: WheelAxisBindings = {};
  const negative = raw.negative;
  const positive = raw.positive;
  if (typeof negative === "number" && Number.isInteger(negative) && negative >= 0) {
    next.negative = negative;
  }
  if (typeof positive === "number" && Number.isInteger(positive) && positive >= 0) {
    next.positive = positive;
  }
  return next;
}

function normalizeEventSet(raw: string[] | undefined): Set<string> {
  if (!raw) {
    return new Set<string>();
  }
  return new Set(
    raw
      .map((entry) => normalizeBindingKey(entry))
      .filter((entry) => entry.length > 0),
  );
}

function resolvePointerButtonAction(
  eventType: string,
  button: number,
  eventMap: Record<string, number>,
  buttonMap: Record<string, number>,
  directional: DirectionalActionBindings,
  allowNoopFallback = true,
): number | undefined {
  const byEventAndButton = eventMap[`${eventType}:${button}`];
  if (byEventAndButton !== undefined) {
    return byEventAndButton;
  }
  const byEvent = eventMap[eventType];
  if (byEvent !== undefined) {
    return byEvent;
  }
  const byButton = buttonMap[String(button)];
  if (byButton !== undefined) {
    return byButton;
  }
  if (!allowNoopFallback) {
    return undefined;
  }
  return directional.noop;
}

function resolveSwipeAction(
  deltaX: number,
  deltaY: number,
  directional: DirectionalActionBindings,
  threshold: number,
): number | undefined {
  const absX = Math.abs(deltaX);
  const absY = Math.abs(deltaY);
  if (absX < threshold && absY < threshold) {
    return directional.noop;
  }
  if (absX >= absY) {
    return deltaX >= 0 ? directional.right : directional.left;
  }
  return deltaY >= 0 ? directional.down : directional.up;
}

function resolveWheelAction(deltaX: number, deltaY: number, vertical: WheelAxisBindings, horizontal: WheelAxisBindings): number | undefined {
  if (Math.abs(deltaY) >= Math.abs(deltaX)) {
    if (deltaY > 0) {
      return vertical.positive;
    }
    if (deltaY < 0) {
      return vertical.negative;
    }
    return undefined;
  }
  if (deltaX > 0) {
    return horizontal.positive;
  }
  if (deltaX < 0) {
    return horizontal.negative;
  }
  return undefined;
}

function buildEffectsStoreFromSpecs(specs: EffectSpecsByStage): EffectsStore {
  const nextStore: EffectsStore = {};
  for (const [stage, effects] of Object.entries(specs)) {
    const stageStore: Record<string, EffectEntry> = {};
    for (const effect of effects) {
      const params: Record<string, JSONValue> = {};
      for (const parameter of effect.parameters ?? []) {
        params[parameter.name] = parameter.default !== undefined ? parameter.default : null;
      }
      stageStore[effect.name] = {
        enabled: false,
        params,
      };
    }
    nextStore[stage] = stageStore;
  }
  return nextStore;
}

interface MetricPillProps {
  label: string;
  value: string;
  icon: typeof Gauge;
}

const MetricPill = memo(function MetricPill({ label, value, icon: Icon }: MetricPillProps) {
  return (
    <div className="metric-pill">
      <div className="metric-icon-wrap">
        <Icon className="h-4 w-4" />
      </div>
      <div>
        <p className="metric-label">{label}</p>
        <p className="metric-value">{value}</p>
      </div>
    </div>
  );
});

interface ParameterFieldProps {
  stage: string;
  effectName: string;
  param: ParameterSpec;
  value: JSONValue | undefined;
  disabled: boolean;
  onParamChange: (stage: string, effectName: string, paramName: string, nextValue: JSONValue) => void;
}

const ParameterField = memo(function ParameterField({
  stage,
  effectName,
  param,
  value,
  disabled,
  onParamChange,
}: ParameterFieldProps) {
  const kind = param.kind ?? "text";
  const paramValue = value ?? param.default;

  const onValueChange = useCallback(
    (nextValue: JSONValue) => {
      onParamChange(stage, effectName, param.name, nextValue);
    },
    [effectName, onParamChange, param.name, stage],
  );

  const optionEntries = useMemo(() => {
    const options = param.options ?? [];
    return options.map((entry) => ({
      label: entry.label,
      raw: entry.value,
      token: stableStringify(entry.value),
    }));
  }, [param.options]);

  const selectedToken = useMemo(() => {
    if (kind !== "select") {
      return "";
    }
    const asToken = stableStringify(paramValue ?? "");
    const match = optionEntries.find((entry) => entry.token === asToken);
    return match?.token ?? optionEntries[0]?.token ?? "";
  }, [kind, optionEntries, paramValue]);

  const hasBounds =
    kind === "number" &&
    typeof param.min === "number" &&
    typeof param.max === "number" &&
    Number.isFinite(param.min) &&
    Number.isFinite(param.max) &&
    param.max > param.min;

  const sliderStep = param.step ?? 0.05;
  const numericValue = typeof paramValue === "number" ? paramValue : Number(paramValue);
  const sliderValue =
    hasBounds && Number.isFinite(numericValue)
      ? Math.min(param.max as number, Math.max(param.min as number, numericValue))
      : (param.min as number);

  return (
    <div className="space-y-2 rounded-xl border border-border/60 bg-card/40 p-3">
      <div className="flex items-center justify-between gap-2">
        <Label className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">{param.label}</Label>
        {param.annotation ? <Badge variant="secondary">{param.annotation}</Badge> : null}
      </div>

      {kind === "number" ? (
        hasBounds ? (
          <div className="space-y-2">
            <Slider
              disabled={disabled}
              min={param.min}
              max={param.max}
              step={sliderStep}
              value={[sliderValue]}
              onValueChange={(next) => {
                onValueChange(next[0]);
              }}
            />
            <div className="flex items-center justify-between gap-2 text-xs text-muted-foreground">
              <span>{param.min}</span>
              <Input
                type="number"
                className="h-7 w-24 text-right"
                disabled={disabled}
                step={sliderStep}
                min={param.min}
                max={param.max}
                value={createInitialRendererValue(paramValue)}
                onChange={(event) => {
                  const raw = event.target.value;
                  onValueChange(raw === "" ? "" : Number(raw));
                }}
              />
              <span>{param.max}</span>
            </div>
          </div>
        ) : (
          <Input
            type="number"
            disabled={disabled}
            step={param.step ?? 0.1}
            value={createInitialRendererValue(paramValue)}
            onChange={(event) => {
              const raw = event.target.value;
              onValueChange(raw === "" ? "" : Number(raw));
            }}
          />
        )
      ) : null}

      {kind === "color" ? (
        <div className="grid grid-cols-[56px_1fr] gap-2">
          <Input
            type="color"
            disabled={disabled}
            value={typeof paramValue === "string" && /^#([A-Fa-f0-9]{6}|[A-Fa-f0-9]{3})$/.test(paramValue) ? paramValue : "#8899aa"}
            onChange={(event) => {
              onValueChange(event.target.value);
            }}
          />
          <Input
            type="text"
            disabled={disabled}
            placeholder="#RRGGBB"
            value={createInitialRendererValue(paramValue)}
            onChange={(event) => {
              onValueChange(event.target.value);
            }}
          />
        </div>
      ) : null}

      {kind === "select" ? (
        <Select
          disabled={disabled || optionEntries.length === 0}
          value={selectedToken}
          onValueChange={(token) => {
            const selected = optionEntries.find((entry) => entry.token === token);
            onValueChange(selected ? selected.raw : token);
          }}
        >
          <SelectTrigger>
            <SelectValue placeholder="Select option" />
          </SelectTrigger>
          <SelectContent>
            {optionEntries.map((entry) => (
              <SelectItem key={`${param.name}-${entry.token}`} value={entry.token}>
                {entry.label}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      ) : null}

      {kind === "switch" ? (
        <div className="flex items-center justify-between rounded-lg border border-border/60 bg-muted/30 px-3 py-2">
          <span className="text-sm text-foreground/90">Enabled</span>
          <Switch
            disabled={disabled}
            checked={Boolean(paramValue)}
            onCheckedChange={(next) => {
              onValueChange(next);
            }}
          />
        </div>
      ) : null}

      {kind === "json" ? (
        <Textarea
          disabled={disabled}
          rows={4}
          placeholder={param.placeholder ?? "Enter JSON"}
          value={asJsonString(paramValue)}
          onChange={(event) => {
            onValueChange(event.target.value);
          }}
        />
      ) : null}

      {(kind === "text" || (!kind && kind !== "json")) && (
        <Input
          type="text"
          disabled={disabled}
          placeholder={param.placeholder ?? "Value"}
          value={createInitialRendererValue(paramValue)}
          onChange={(event) => {
            onValueChange(event.target.value);
          }}
        />
      )}
    </div>
  );
});

interface HistoryCardProps {
  entry: HistoryEntry;
  onRestore: (entry: HistoryEntry) => void;
}

const HistoryCard = memo(function HistoryCard({ entry, onRestore }: HistoryCardProps) {
  return (
    <Card className="overflow-hidden border-border/70 bg-card/70">
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between gap-2">
          <CardTitle className="text-sm uppercase tracking-[0.12em] text-muted-foreground">{entry.env}</CardTitle>
          <Badge variant={entry.cached ? "secondary" : "outline"}>{entry.cached ? "Cached" : `${entry.renderMs.toFixed(1)} ms`}</Badge>
        </div>
        <CardDescription>{new Date(entry.createdAt).toLocaleTimeString()}</CardDescription>
      </CardHeader>
      <CardContent className="space-y-3">
        <div className="history-preview-wrap">
          <img src={entry.image} alt={`${entry.env} history preview`} className="history-preview" loading="lazy" />
        </div>
        <Button
          type="button"
          variant="secondary"
          className="w-full"
          onClick={() => {
            onRestore(entry);
          }}
        >
          Restore Snapshot
        </Button>
      </CardContent>
    </Card>
  );
});

function App() {
  const [environmentOptions, setEnvironmentOptions] = useState<EnvironmentOption[]>([]);
  const [environment, setEnvironment] = useState<string>("");
  const [statePayload, setStatePayload] = useState<StatePayload | null>(null);
  const [effectSpecs, setEffectSpecs] = useState<EffectSpecsByStage>({});
  const [effectsStore, setEffectsStore] = useState<EffectsStore>({});
  const [rendererStore, setRendererStore] = useState<RendererStore>({});

  const [imageSrc, setImageSrc] = useState<string>("");
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [isBootstrapping, setIsBootstrapping] = useState<boolean>(true);
  const [isRendering, setIsRendering] = useState<boolean>(false);
  const [renderMs, setRenderMs] = useState<number>(0);
  const [wasCached, setWasCached] = useState<boolean>(false);
  const [lastError, setLastError] = useState<string>("");
  const [interactiveMode, setInteractiveMode] = useState<boolean>(false);
  const [interactiveSessionId, setInteractiveSessionId] = useState<string | null>(null);
  const [interactiveActionLabels, setInteractiveActionLabels] = useState<string[]>([]);
  const [interactiveBindings, setInteractiveBindings] = useState<InteractiveBindings>({});
  const [interactiveSelectedAction, setInteractiveSelectedAction] = useState<number>(0);
  const [isInteractiveBusy, setIsInteractiveBusy] = useState<boolean>(false);
  const [themePreference, setThemePreference] = useState<ThemePreference>(loadInitialThemePreference);
  const [systemPrefersDark, setSystemPrefersDark] = useState<boolean>(() => {
    if (typeof window === "undefined" || !window.matchMedia) {
      return false;
    }
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  });

  const renderAbortRef = useRef<AbortController | null>(null);
  const bootstrapAbortRef = useRef<AbortController | null>(null);
  const interactiveAbortRef = useRef<AbortController | null>(null);
  const interactiveSessionRef = useRef<string | null>(null);
  const previewFrameRef = useRef<HTMLDivElement | null>(null);
  const pointerGestureRef = useRef<PointerGestureState | null>(null);
  const suppressNextClickRef = useRef<boolean>(false);
  const queuedInteractiveActionsRef = useRef<number[]>([]);
  const lastRenderKeyRef = useRef<string>("");
  const interactiveRenderKeyRef = useRef<string>("");
  const bootstrapSequenceRef = useRef<number>(0);

  const canRender = Boolean(environment && statePayload);
  const interactiveActionCount = interactiveActionLabels.length;
  const resolvedTheme = themePreference === "system" ? (systemPrefersDark ? "dark" : "light") : themePreference;
  const interactiveKeyToAction = useMemo(
    () => normalizeActionMap(interactiveBindings.keyboard?.key_to_action),
    [interactiveBindings.keyboard?.key_to_action],
  );
  const interactiveKeyboardEvents = useMemo(() => {
    const configured = normalizeEventSet(interactiveBindings.keyboard?.events);
    return configured.size > 0 ? configured : new Set<string>(["keydown"]);
  }, [interactiveBindings.keyboard?.events]);
  const interactivePointerEvents = useMemo(() => {
    const configured = normalizeEventSet(interactiveBindings.pointer?.events);
    return configured.size > 0
      ? configured
      : new Set<string>(["pointerdown", "pointerup", "click", "auxclick", "dblclick", "contextmenu"]);
  }, [interactiveBindings.pointer?.events]);
  const interactivePointerDirectional = useMemo(
    () => normalizeDirectionalBindings(interactiveBindings.pointer?.directional),
    [interactiveBindings.pointer?.directional],
  );
  const interactivePointerButtonMap = useMemo(
    () => normalizeActionMap(interactiveBindings.pointer?.button_to_action),
    [interactiveBindings.pointer?.button_to_action],
  );
  const interactivePointerEventMap = useMemo(
    () => normalizeActionMap(interactiveBindings.pointer?.event_to_action),
    [interactiveBindings.pointer?.event_to_action],
  );
  const interactiveSwipeThreshold = useMemo(() => {
    const configured = interactiveBindings.pointer?.swipe_threshold;
    if (typeof configured === "number" && Number.isFinite(configured) && configured >= 0) {
      return configured;
    }
    return 18;
  }, [interactiveBindings.pointer?.swipe_threshold]);
  const interactiveWheelEvents = useMemo(() => {
    const configured = normalizeEventSet(interactiveBindings.wheel?.events);
    return configured.size > 0 ? configured : new Set<string>(["wheel"]);
  }, [interactiveBindings.wheel?.events]);
  const interactiveWheelVertical = useMemo(
    () => normalizeWheelAxisBindings(interactiveBindings.wheel?.vertical),
    [interactiveBindings.wheel?.vertical],
  );
  const interactiveWheelHorizontal = useMemo(
    () => normalizeWheelAxisBindings(interactiveBindings.wheel?.horizontal),
    [interactiveBindings.wheel?.horizontal],
  );

  const renderInputKey = useMemo(() => {
    if (!canRender || statePayload === null) {
      return "";
    }
    return stableStringify({
      env: environment,
      state: statePayload,
      effects: effectsStore,
      renderer: rendererStore,
    });
  }, [canRender, effectsStore, environment, rendererStore, statePayload]);

  const interactiveRenderInputKey = useMemo(() => {
    if (!interactiveMode || !interactiveSessionId) {
      return "";
    }
    return stableStringify({
      session: interactiveSessionId,
      effects: effectsStore,
      renderer: rendererStore,
    });
  }, [effectsStore, interactiveMode, interactiveSessionId, rendererStore]);

  const stagedEffects = useMemo(
    () =>
      STAGE_ORDER.map((stage) => ({
        stage,
        label: STAGE_LABELS[stage],
        effects: effectSpecs[stage] ?? [],
      })),
    [effectSpecs],
  );

  const activeEffectCount = useMemo(
    () =>
      Object.values(effectsStore).reduce((count, stageStore) => {
        return (
          count +
          Object.values(stageStore).reduce((stageCount, effectEntry) => {
            return stageCount + (effectEntry.enabled ? 1 : 0);
          }, 0)
        );
      }, 0),
    [effectsStore],
  );

  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) {
      return;
    }
    const mediaQuery = window.matchMedia("(prefers-color-scheme: dark)");
    const update = () => {
      setSystemPrefersDark(mediaQuery.matches);
    };
    update();
    mediaQuery.addEventListener("change", update);
    return () => {
      mediaQuery.removeEventListener("change", update);
    };
  }, []);

  useEffect(() => {
    if (typeof window !== "undefined") {
      window.localStorage.setItem(THEME_STORAGE_KEY, themePreference);
    }
    document.documentElement.setAttribute("data-theme", resolvedTheme);
    document.documentElement.style.colorScheme = resolvedTheme;
  }, [resolvedTheme, themePreference]);

  useEffect(() => {
    interactiveSessionRef.current = interactiveSessionId;
  }, [interactiveSessionId]);

  const releaseInteractiveSession = useCallback((sessionId: string | null) => {
    if (!sessionId) {
      return;
    }
    void fetchInteractiveStop(sessionId).catch(() => {
      // The session may already have ended, which needs no additional action.
    });
  }, []);

  const clearInteractiveSession = useCallback(
    (options?: { releaseRemote?: boolean }) => {
      const releaseRemote = options?.releaseRemote ?? true;
      if (interactiveAbortRef.current) {
        interactiveAbortRef.current.abort();
        interactiveAbortRef.current = null;
      }
      pointerGestureRef.current = null;
      suppressNextClickRef.current = false;
      queuedInteractiveActionsRef.current = [];
      const currentSessionId = interactiveSessionRef.current;
      if (releaseRemote && currentSessionId) {
        releaseInteractiveSession(currentSessionId);
      }
      interactiveSessionRef.current = null;
      setInteractiveSessionId((previous) => (previous === null ? previous : null));
      setInteractiveActionLabels((previous) => (previous.length === 0 ? previous : []));
      setInteractiveBindings({});
      setInteractiveSelectedAction(0);
      setIsInteractiveBusy(false);
      interactiveRenderKeyRef.current = "";
    },
    [releaseInteractiveSession],
  );

  const bootEnvironment = useCallback(async (nextEnvironment: string) => {
    bootstrapSequenceRef.current += 1;
    const currentSequence = bootstrapSequenceRef.current;
    clearInteractiveSession({ releaseRemote: true });

    if (bootstrapAbortRef.current) {
      bootstrapAbortRef.current.abort();
    }
    const abortController = new AbortController();
    bootstrapAbortRef.current = abortController;

    // Clear stale state immediately so no render/interactive request can run with a mismatched env/state pair.
    setIsBootstrapping(true);
    setLastError("");
    setEnvironment(nextEnvironment);
    setStatePayload(null);
    setImageSrc("");
    setRenderMs(0);
    setWasCached(false);
    setIsRendering(false);
    lastRenderKeyRef.current = "";

    try {
      const response = await fetchBootstrap(nextEnvironment, abortController.signal);
      if (currentSequence !== bootstrapSequenceRef.current) {
        return;
      }
      setEnvironment(response.env);
      setEffectSpecs(response.effect_specs);
      setEffectsStore(response.effects_store);
      setRendererStore(response.renderer);
      setStatePayload(response.state);
      setIsInteractiveBusy(false);
    } catch (error) {
      if (isAbortError(error)) {
        return;
      }
      const message = getErrorMessage(error);
      setLastError(message);
      toast.error(message);
    } finally {
      if (currentSequence === bootstrapSequenceRef.current) {
        setIsBootstrapping(false);
      }
    }
  }, [clearInteractiveSession]);

  const applyPresetToEffects = useCallback((preset: SweepPreset) => {
    setEffectSpecs((previous) => mergePresetIntoSpecs(previous, preset));
    setEffectsStore((previous) => applyPresetToStore(previous, preset));
    toast.success(`Applied "${preset.name}" to the effect stack`);
  }, []);

  useEffect(() => {
    const abortController = new AbortController();

    const initialize = async () => {
      setIsBootstrapping(true);
      try {
        const response = await fetchEnvironments(abortController.signal);
        setEnvironmentOptions(response.environments);
        const preferred = response.default_env || response.environments[0]?.value;
        if (preferred) {
          await bootEnvironment(preferred);
        } else {
          setIsBootstrapping(false);
        }
      } catch (error) {
        if (isAbortError(error)) {
          return;
        }
        const message = getErrorMessage(error);
        setLastError(message);
        setIsBootstrapping(false);
        toast.error(message);
      }
    };

    void initialize();

    return () => {
      abortController.abort();
      if (bootstrapAbortRef.current) {
        bootstrapAbortRef.current.abort();
      }
      if (renderAbortRef.current) {
        renderAbortRef.current.abort();
      }
      if (interactiveAbortRef.current) {
        interactiveAbortRef.current.abort();
      }
    };
  }, [bootEnvironment]);

  useEffect(() => {
    if (interactiveMode) {
      return;
    }
    clearInteractiveSession({ releaseRemote: true });
  }, [clearInteractiveSession, interactiveMode]);

  useEffect(() => {
    if (!interactiveMode || !interactiveSessionId) {
      return;
    }
    const timeoutId = window.setTimeout(() => {
      previewFrameRef.current?.focus({ preventScroll: true });
    }, 0);
    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [interactiveMode, interactiveSessionId]);

  useEffect(() => {
    if (!interactiveMode || isBootstrapping || !canRender || statePayload === null || interactiveSessionId) {
      return;
    }
    if (interactiveAbortRef.current) {
      interactiveAbortRef.current.abort();
    }
    const abortController = new AbortController();
    interactiveAbortRef.current = abortController;
    setIsInteractiveBusy(true);

    void fetchInteractiveStart(environment, statePayload, effectsStore, rendererStore, abortController.signal)
      .then((result) => {
        if (abortController.signal.aborted) {
          return;
        }
        interactiveSessionRef.current = result.session_id;
        setInteractiveSessionId(result.session_id);
        setInteractiveActionLabels(result.action_labels);
        setInteractiveBindings(result.interactive_bindings ?? {});
        setInteractiveSelectedAction((previous) => {
          if (result.action_count <= 0) {
            return 0;
          }
          return Math.min(previous, result.action_count - 1);
        });
        setStatePayload(result.state);
        setImageSrc(result.image);
        setRenderMs(result.render_ms);
        setWasCached(false);
        setLastError("");
        interactiveRenderKeyRef.current = stableStringify({
          session: result.session_id,
          effects: effectsStore,
          renderer: rendererStore,
        });
      })
      .catch((error) => {
        if (isAbortError(error)) {
          return;
        }
        const message = getErrorMessage(error);
        setLastError(message);
        toast.error(message);
        setInteractiveMode(false);
        clearInteractiveSession({ releaseRemote: true });
      })
      .finally(() => {
        if (!abortController.signal.aborted) {
          setIsInteractiveBusy(false);
        }
      });

    return () => {
      abortController.abort();
    };
  }, [canRender, clearInteractiveSession, effectsStore, environment, interactiveMode, interactiveSessionId, isBootstrapping, rendererStore, statePayload]);

  useEffect(() => {
    if (!interactiveMode || !interactiveSessionId || !interactiveRenderInputKey) {
      return;
    }
    if (interactiveRenderInputKey === interactiveRenderKeyRef.current) {
      return;
    }

    const timeoutId = window.setTimeout(() => {
      if (interactiveAbortRef.current) {
        interactiveAbortRef.current.abort();
      }
      const abortController = new AbortController();
      interactiveAbortRef.current = abortController;
      setIsInteractiveBusy(true);

      void fetchInteractiveRender(interactiveSessionId, effectsStore, rendererStore, abortController.signal)
        .then((result) => {
          if (abortController.signal.aborted) {
            return;
          }
          setStatePayload(result.state);
          setImageSrc(result.image);
          setRenderMs(result.render_ms);
          setWasCached(false);
          setInteractiveActionLabels(result.action_labels);
          setInteractiveBindings(result.interactive_bindings ?? {});
          setInteractiveSelectedAction((previous) => {
            if (result.action_count <= 0) {
              return 0;
            }
            return Math.min(previous, result.action_count - 1);
          });
          setLastError("");
          interactiveRenderKeyRef.current = interactiveRenderInputKey;
        })
        .catch((error) => {
          if (isAbortError(error)) {
            return;
          }
          const message = getErrorMessage(error);
          setLastError(message);
          toast.error(message);
        })
        .finally(() => {
          if (!abortController.signal.aborted) {
            setIsInteractiveBusy(false);
          }
        });
    }, RENDER_DEBOUNCE_MS);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [effectsStore, interactiveMode, interactiveRenderInputKey, interactiveSessionId, rendererStore]);

  const handleInteractiveStep = useCallback(
    async (action: number) => {
      if (!interactiveMode || !interactiveSessionId || isInteractiveBusy) {
        return;
      }
      if (!Number.isInteger(action) || action < 0) {
        return;
      }
      if (interactiveAbortRef.current) {
        interactiveAbortRef.current.abort();
      }
      const abortController = new AbortController();
      interactiveAbortRef.current = abortController;
      setIsInteractiveBusy(true);
      try {
        const result = await fetchInteractiveStep(interactiveSessionId, action, effectsStore, rendererStore, abortController.signal);
        if (abortController.signal.aborted) {
          return;
        }
        setStatePayload(result.state);
        setImageSrc(result.image);
        setRenderMs(result.render_ms);
        setWasCached(false);
        setInteractiveActionLabels(result.action_labels);
        setInteractiveBindings(result.interactive_bindings ?? {});
        setInteractiveSelectedAction((previous) => {
          if (result.action_count <= 0) {
            return 0;
          }
          return Math.min(previous, result.action_count - 1);
        });
        setLastError("");
        interactiveRenderKeyRef.current = stableStringify({
          session: result.session_id,
          effects: effectsStore,
          renderer: rendererStore,
        });
      } catch (error) {
        if (isAbortError(error)) {
          return;
        }
        const message = getErrorMessage(error);
        setLastError(message);
        toast.error(message);
      } finally {
        if (!abortController.signal.aborted) {
          setIsInteractiveBusy(false);
        }
      }
    },
    [effectsStore, interactiveMode, interactiveSessionId, isInteractiveBusy, rendererStore],
  );

  const handleInteractiveReset = useCallback(async (): Promise<boolean> => {
    if (!interactiveMode || !interactiveSessionId || isInteractiveBusy) {
      return false;
    }
    if (interactiveAbortRef.current) {
      interactiveAbortRef.current.abort();
    }
    const abortController = new AbortController();
    interactiveAbortRef.current = abortController;
    setIsInteractiveBusy(true);
    try {
      const result = await fetchInteractiveReset(interactiveSessionId, effectsStore, rendererStore, abortController.signal);
      if (abortController.signal.aborted) {
        return false;
      }
      setStatePayload(result.state);
      setImageSrc(result.image);
      setRenderMs(result.render_ms);
      setWasCached(false);
      setInteractiveActionLabels(result.action_labels);
      setInteractiveBindings(result.interactive_bindings ?? {});
      setInteractiveSelectedAction((previous) => {
        if (result.action_count <= 0) {
          return 0;
        }
        return Math.min(previous, result.action_count - 1);
      });
      setLastError("");
      interactiveRenderKeyRef.current = stableStringify({
        session: result.session_id,
        effects: effectsStore,
        renderer: rendererStore,
      });
      return true;
    } catch (error) {
      if (isAbortError(error)) {
        return false;
      }
      const message = getErrorMessage(error);
      setLastError(message);
      toast.error(message);
      return false;
    } finally {
      if (!abortController.signal.aborted) {
        setIsInteractiveBusy(false);
      }
    }
  }, [effectsStore, interactiveMode, interactiveSessionId, isInteractiveBusy, rendererStore]);

  const enqueueInteractiveAction = useCallback(
    (action: number) => {
      if (!interactiveMode || !interactiveSessionId) {
        return;
      }
      if (!Number.isInteger(action) || action < 0) {
        return;
      }
      queuedInteractiveActionsRef.current.push(action);
      if (!isInteractiveBusy && queuedInteractiveActionsRef.current.length === 1) {
        const nextAction = queuedInteractiveActionsRef.current.shift();
        if (nextAction !== undefined) {
          void handleInteractiveStep(nextAction);
        }
      }
    },
    [handleInteractiveStep, interactiveMode, interactiveSessionId, isInteractiveBusy],
  );

  useEffect(() => {
    if (!interactiveMode || !interactiveSessionId || isInteractiveBusy) {
      return;
    }
    const nextAction = queuedInteractiveActionsRef.current.shift();
    if (nextAction === undefined) {
      return;
    }
    void handleInteractiveStep(nextAction);
  }, [handleInteractiveStep, interactiveMode, interactiveSessionId, isInteractiveBusy]);

  const handlePreviewKeyDown = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      if (!interactiveMode || !interactiveSessionId) {
        return;
      }
      if (!interactiveKeyboardEvents.has("keydown")) {
        return;
      }
      if (event.altKey || event.metaKey || event.ctrlKey) {
        return;
      }
      const key = normalizeBindingKey(event.key);
      const code = normalizeBindingKey(event.code);
      const mappedAction = interactiveKeyToAction[key] ?? interactiveKeyToAction[code];
      if (mappedAction === undefined) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      enqueueInteractiveAction(mappedAction);
    },
    [enqueueInteractiveAction, interactiveKeyToAction, interactiveKeyboardEvents, interactiveMode, interactiveSessionId],
  );

  const handlePreviewKeyUp = useCallback(
    (event: ReactKeyboardEvent<HTMLDivElement>) => {
      if (!interactiveMode || !interactiveSessionId) {
        return;
      }
      if (!interactiveKeyboardEvents.has("keyup")) {
        return;
      }
      if (event.altKey || event.metaKey || event.ctrlKey) {
        return;
      }
      const key = normalizeBindingKey(event.key);
      const code = normalizeBindingKey(event.code);
      const mappedAction = interactiveKeyToAction[key] ?? interactiveKeyToAction[code];
      if (mappedAction === undefined) {
        return;
      }
      event.preventDefault();
      event.stopPropagation();
      enqueueInteractiveAction(mappedAction);
    },
    [enqueueInteractiveAction, interactiveKeyToAction, interactiveKeyboardEvents, interactiveMode, interactiveSessionId],
  );

  const handlePreviewPointerDown = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (!interactiveMode || !interactiveSessionId) {
        return;
      }
      suppressNextClickRef.current = false;
      pointerGestureRef.current = {
        pointerId: event.pointerId,
        startX: event.clientX,
        startY: event.clientY,
      };
      event.currentTarget.focus({ preventScroll: true });
      event.currentTarget.setPointerCapture(event.pointerId);
      if (!interactivePointerEvents.has("pointerdown")) {
        return;
      }
      const action = resolvePointerButtonAction(
        "pointerdown",
        event.button,
        interactivePointerEventMap,
        interactivePointerButtonMap,
        interactivePointerDirectional,
        false,
      );
      if (action === undefined) {
        return;
      }
      event.preventDefault();
      enqueueInteractiveAction(action);
    },
    [
      enqueueInteractiveAction,
      interactiveMode,
      interactivePointerButtonMap,
      interactivePointerDirectional,
      interactivePointerEventMap,
      interactivePointerEvents,
      interactiveSessionId,
    ],
  );

  const handlePreviewPointerCancel = useCallback((event: ReactPointerEvent<HTMLDivElement>) => {
    const gesture = pointerGestureRef.current;
    if (gesture && gesture.pointerId === event.pointerId) {
      pointerGestureRef.current = null;
    }
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }, []);

  const handlePreviewPointerUp = useCallback(
    (event: ReactPointerEvent<HTMLDivElement>) => {
      if (!interactiveMode || !interactiveSessionId) {
        return;
      }
      const gesture = pointerGestureRef.current;
      if (!gesture || gesture.pointerId !== event.pointerId) {
        return;
      }
      pointerGestureRef.current = null;
      if (event.currentTarget.hasPointerCapture(event.pointerId)) {
        event.currentTarget.releasePointerCapture(event.pointerId);
      }
      if (!interactivePointerEvents.has("pointerup")) {
        return;
      }
      const action =
        resolveSwipeAction(
          event.clientX - gesture.startX,
          event.clientY - gesture.startY,
          interactivePointerDirectional,
          interactiveSwipeThreshold,
        ) ??
        resolvePointerButtonAction(
          "pointerup",
          event.button,
          interactivePointerEventMap,
          interactivePointerButtonMap,
          interactivePointerDirectional,
        );
      if (action === undefined) {
        return;
      }
      suppressNextClickRef.current = true;
      event.preventDefault();
      enqueueInteractiveAction(action);
    },
    [
      enqueueInteractiveAction,
      interactiveMode,
      interactivePointerButtonMap,
      interactivePointerDirectional,
      interactivePointerEventMap,
      interactivePointerEvents,
      interactiveSessionId,
      interactiveSwipeThreshold,
    ],
  );

  const handlePreviewClick = useCallback(
    (event: ReactMouseEvent<HTMLDivElement>) => {
      if (!interactiveMode || !interactiveSessionId) {
        return;
      }
      if (!interactivePointerEvents.has("click")) {
        return;
      }
      if (suppressNextClickRef.current) {
        suppressNextClickRef.current = false;
        return;
      }
      const action = resolvePointerButtonAction(
        "click",
        event.button,
        interactivePointerEventMap,
        interactivePointerButtonMap,
        interactivePointerDirectional,
      );
      if (action === undefined) {
        return;
      }
      event.preventDefault();
      enqueueInteractiveAction(action);
    },
    [
      enqueueInteractiveAction,
      interactiveMode,
      interactivePointerButtonMap,
      interactivePointerDirectional,
      interactivePointerEventMap,
      interactivePointerEvents,
      interactiveSessionId,
    ],
  );

  const handlePreviewDoubleClick = useCallback(
    (event: ReactMouseEvent<HTMLDivElement>) => {
      if (!interactiveMode || !interactiveSessionId) {
        return;
      }
      if (!interactivePointerEvents.has("dblclick")) {
        return;
      }
      const action = resolvePointerButtonAction(
        "dblclick",
        event.button,
        interactivePointerEventMap,
        interactivePointerButtonMap,
        interactivePointerDirectional,
      );
      if (action === undefined) {
        return;
      }
      event.preventDefault();
      enqueueInteractiveAction(action);
    },
    [
      enqueueInteractiveAction,
      interactiveMode,
      interactivePointerButtonMap,
      interactivePointerDirectional,
      interactivePointerEventMap,
      interactivePointerEvents,
      interactiveSessionId,
    ],
  );

  const handlePreviewAuxClick = useCallback(
    (event: ReactMouseEvent<HTMLDivElement>) => {
      if (!interactiveMode || !interactiveSessionId) {
        return;
      }
      if (!interactivePointerEvents.has("auxclick")) {
        return;
      }
      const action = resolvePointerButtonAction(
        "auxclick",
        event.button,
        interactivePointerEventMap,
        interactivePointerButtonMap,
        interactivePointerDirectional,
      );
      if (action === undefined) {
        return;
      }
      event.preventDefault();
      enqueueInteractiveAction(action);
    },
    [
      enqueueInteractiveAction,
      interactiveMode,
      interactivePointerButtonMap,
      interactivePointerDirectional,
      interactivePointerEventMap,
      interactivePointerEvents,
      interactiveSessionId,
    ],
  );

  const handlePreviewContextMenu = useCallback(
    (event: ReactMouseEvent<HTMLDivElement>) => {
      if (!interactiveMode || !interactiveSessionId) {
        return;
      }
      event.preventDefault();
      if (!interactivePointerEvents.has("contextmenu")) {
        return;
      }
      const action = resolvePointerButtonAction(
        "contextmenu",
        event.button,
        interactivePointerEventMap,
        interactivePointerButtonMap,
        interactivePointerDirectional,
      );
      if (action === undefined) {
        return;
      }
      enqueueInteractiveAction(action);
    },
    [
      enqueueInteractiveAction,
      interactiveMode,
      interactivePointerButtonMap,
      interactivePointerDirectional,
      interactivePointerEventMap,
      interactivePointerEvents,
      interactiveSessionId,
    ],
  );

  const handlePreviewWheel = useCallback(
    (event: ReactWheelEvent<HTMLDivElement>) => {
      if (!interactiveMode || !interactiveSessionId) {
        return;
      }
      if (!interactiveWheelEvents.has("wheel")) {
        return;
      }
      const action = resolveWheelAction(event.deltaX, event.deltaY, interactiveWheelVertical, interactiveWheelHorizontal);
      if (action === undefined) {
        return;
      }
      event.preventDefault();
      enqueueInteractiveAction(action);
    },
    [
      enqueueInteractiveAction,
      interactiveMode,
      interactiveSessionId,
      interactiveWheelEvents,
      interactiveWheelHorizontal,
      interactiveWheelVertical,
    ],
  );

  useEffect(() => {
    return () => {
      if (interactiveAbortRef.current) {
        interactiveAbortRef.current.abort();
      }
      const sessionId = interactiveSessionRef.current;
      if (sessionId) {
        releaseInteractiveSession(sessionId);
      }
    };
  }, [releaseInteractiveSession]);

  useEffect(() => {
    if (interactiveMode) {
      return;
    }
    if (!canRender || statePayload === null || !renderInputKey) {
      return;
    }

    if (renderInputKey === lastRenderKeyRef.current) {
      return;
    }

    const timeoutId = window.setTimeout(() => {
      if (renderAbortRef.current) {
        renderAbortRef.current.abort();
      }
      const abortController = new AbortController();
      renderAbortRef.current = abortController;

      setIsRendering(true);

      const requestKey = renderInputKey;
      const currentState = statePayload;
      const currentEffects = effectsStore;
      const currentRenderer = rendererStore;
      const currentSpecs = effectSpecs;

      void fetchRender(environment, currentState, currentEffects, currentRenderer, abortController.signal)
        .then((result) => {
          if (abortController.signal.aborted) {
            return;
          }

          setImageSrc(result.image);
          setRenderMs(result.render_ms);
          setWasCached(result.cached);
          setLastError("");
          lastRenderKeyRef.current = requestKey;

          const historyEntry: HistoryEntry = {
            id: makeHistoryId(),
            requestKey,
            createdAt: new Date().toISOString(),
            env: environment,
            image: result.image,
            state: structuredClone(currentState),
            effects: structuredClone(currentEffects),
            renderer: structuredClone(currentRenderer),
            effectSpecs: structuredClone(currentSpecs),
            renderMs: result.render_ms,
            cached: result.cached,
          };

          setHistory((previous) => {
            if (previous[0]?.requestKey === requestKey) {
              return previous;
            }
            const next = [historyEntry, ...previous];
            return next.slice(0, HISTORY_LIMIT);
          });
        })
        .catch((error) => {
          if (isAbortError(error)) {
            return;
          }
          const message = getErrorMessage(error);
          setLastError(message);
          toast.error(message);
        })
        .finally(() => {
          if (!abortController.signal.aborted) {
            setIsRendering(false);
          }
        });
    }, RENDER_DEBOUNCE_MS);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [canRender, effectSpecs, effectsStore, environment, interactiveMode, renderInputKey, rendererStore, statePayload]);

  const handleEnvironmentChange = useCallback(
    (nextEnvironment: string) => {
      if (!nextEnvironment || nextEnvironment === environment) {
        return;
      }
      void bootEnvironment(nextEnvironment);
    },
    [bootEnvironment, environment],
  );

  const handleRandomize = useCallback(async () => {
    if (!environment) {
      return;
    }
    if (interactiveMode && interactiveSessionId) {
      const didReset = await handleInteractiveReset();
      if (didReset) {
        toast.success(`Reset ${environment}`);
      }
      return;
    }
    try {
      const response = await fetchRandomize(environment);
      setStatePayload(response.state);
      toast.success(`Randomized ${response.env}`);
    } catch (error) {
      if (isAbortError(error)) {
        return;
      }
      const message = getErrorMessage(error);
      setLastError(message);
      toast.error(message);
    }
  }, [environment, handleInteractiveReset, interactiveMode, interactiveSessionId]);

  const handleExport = useCallback(() => {
    if (!imageSrc) {
      toast.error("No image has been rendered yet");
      return;
    }
    const anchor = document.createElement("a");
    const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
    anchor.href = imageSrc;
    anchor.download = `spar-render-${environment}-${timestamp}.png`;
    anchor.click();
  }, [environment, imageSrc]);

  const handleResetEffects = useCallback(() => {
    if (Object.keys(effectSpecs).length === 0) {
      return;
    }
    setEffectsStore(buildEffectsStoreFromSpecs(effectSpecs));
    toast.success("Effects reset to defaults");
  }, [effectSpecs]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (isTextFieldTarget(event.target)) {
        return;
      }
      if (interactiveMode && previewFrameRef.current && document.activeElement === previewFrameRef.current) {
        return;
      }
      if (event.key.toLowerCase() === "r") {
        event.preventDefault();
        void handleRandomize();
      }
      if (event.key.toLowerCase() === "e") {
        event.preventDefault();
        handleExport();
      }
    };

    window.addEventListener("keydown", handler);
    return () => {
      window.removeEventListener("keydown", handler);
    };
  }, [handleExport, handleRandomize, interactiveMode]);

  const updateEffectEnabled = useCallback((stage: string, effectName: string, enabled: boolean) => {
    setEffectsStore((previous) => {
      const stageStore = previous[stage];
      if (!stageStore) {
        return previous;
      }
      const effectEntry = stageStore[effectName];
      if (!effectEntry || effectEntry.enabled === enabled) {
        return previous;
      }
      return {
        ...previous,
        [stage]: {
          ...stageStore,
          [effectName]: {
            ...effectEntry,
            enabled,
          },
        },
      };
    });
  }, []);

  const updateEffectParam = useCallback(
    (stage: string, effectName: string, paramName: string, nextValue: JSONValue) => {
      setEffectsStore((previous) => {
        const stageStore = previous[stage];
        if (!stageStore) {
          return previous;
        }
        const effectEntry = stageStore[effectName];
        if (!effectEntry) {
          return previous;
        }
        return {
          ...previous,
          [stage]: {
            ...stageStore,
            [effectName]: {
              ...effectEntry,
              params: {
                ...effectEntry.params,
                [paramName]: nextValue,
              },
            },
          },
        };
      });
    },
    [],
  );

  const updateRendererValue = useCallback((key: string, value: JSONValue) => {
    setRendererStore((previous) => ({
      ...previous,
      [key]: value,
    }));
  }, []);

  const restoreHistory = useCallback((entry: HistoryEntry) => {
    clearInteractiveSession({ releaseRemote: true });
    setEnvironment(entry.env);
    setStatePayload(structuredClone(entry.state));
    setEffectsStore(structuredClone(entry.effects));
    setRendererStore(structuredClone(entry.renderer));
    setEffectSpecs(structuredClone(entry.effectSpecs));
    setImageSrc(entry.image);
    setRenderMs(entry.renderMs);
    setWasCached(entry.cached);
    setLastError("");
    lastRenderKeyRef.current = "";
    toast.success(`Restored snapshot for ${entry.env}`);
  }, [clearInteractiveSession]);

  const rendererFields = useMemo(() => {
    const entries = Object.entries(rendererStore);
    entries.sort(([left], [right]) => left.localeCompare(right));
    return entries;
  }, [rendererStore]);

  return (
    <div className="dashboard-shell">
      <div className="ambient-layer" aria-hidden="true" />
      <div className="grid-overlay" aria-hidden="true" />
      <TooltipProvider delayDuration={120}>
        <div className="theme-toolbar" role="group" aria-label="Theme preference">
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                type="button"
                variant={themePreference === "system" ? "secondary" : "ghost"}
                size="icon"
                className={cn("theme-icon-button", themePreference === "system" && "theme-icon-button-active")}
                aria-label="Use system theme"
                aria-pressed={themePreference === "system"}
                onClick={() => {
                  setThemePreference("system");
                }}
              >
                <MonitorCog className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="bottom">System theme</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                type="button"
                variant={themePreference === "light" ? "secondary" : "ghost"}
                size="icon"
                className={cn("theme-icon-button", themePreference === "light" && "theme-icon-button-active")}
                aria-label="Use light theme"
                aria-pressed={themePreference === "light"}
                onClick={() => {
                  setThemePreference("light");
                }}
              >
                <Sun className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="bottom">Light theme</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                type="button"
                variant={themePreference === "dark" ? "secondary" : "ghost"}
                size="icon"
                className={cn("theme-icon-button", themePreference === "dark" && "theme-icon-button-active")}
                aria-label="Use dark theme"
                aria-pressed={themePreference === "dark"}
                onClick={() => {
                  setThemePreference("dark");
                }}
              >
                <MoonStar className="h-4 w-4" />
              </Button>
            </TooltipTrigger>
            <TooltipContent side="bottom">Dark theme</TooltipContent>
          </Tooltip>
        </div>
      </TooltipProvider>

      <header className="dashboard-header">
        <div className="title-cluster">
          <p className="eyebrow">SPAR Dashboard</p>
          <h1 className="headline">Environment Preview and Effects</h1>
          <p className="subtitle">Configure any available SPAR environment, apply effects, and inspect rendered output.</p>
        </div>

        <div className="header-controls">
          <div className="w-full min-w-[180px] sm:w-[260px]">
            <Label htmlFor="environment-picker" className="mb-2 block text-xs uppercase tracking-[0.12em] text-muted-foreground">
              Environment
            </Label>
            <Select value={environment} onValueChange={handleEnvironmentChange} disabled={isBootstrapping}>
              <SelectTrigger id="environment-picker" className="h-11 bg-card/80">
                <SelectValue placeholder="Select environment" />
              </SelectTrigger>
              <SelectContent>
                {environmentOptions.map((option) => (
                  <SelectItem key={option.value} value={option.value}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="interactive-toggle-card">
            <div className="space-y-1">
              <p className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">Interactive Preview</p>
              <p className="text-[11px] text-muted-foreground/90">
                {interactiveMode ? "Live environment instance" : "Static render mode"}
              </p>
            </div>
            <Switch
              checked={interactiveMode}
              onCheckedChange={(next) => {
                setInteractiveMode(next);
              }}
              disabled={!canRender || isBootstrapping}
            />
          </div>

          <Button type="button" variant="secondary" className="h-11 gap-2" onClick={() => void handleRandomize()} disabled={!environment || isBootstrapping}>
            <RefreshCcw className="h-4 w-4" />
            Randomize
          </Button>

          <Button
            type="button"
            variant="secondary"
            className="h-11 gap-2"
            onClick={handleResetEffects}
            disabled={isBootstrapping || Object.keys(effectSpecs).length === 0}
          >
            <RotateCcw className="h-4 w-4" />
            Reset Effects
          </Button>

          <Button type="button" className="h-11 gap-2" onClick={handleExport} disabled={!imageSrc}>
            <Download className="h-4 w-4" />
            Export PNG
          </Button>
        </div>
      </header>

      <main className="dashboard-main">
        <section className="left-column">
          <Card className="visual-card">
            <CardHeader className="space-y-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <CardTitle className="text-2xl font-semibold">Preview</CardTitle>
                  <CardDescription>Changes to state, effects, or renderer settings update the rendering.</CardDescription>
                </div>
                <Badge
                  variant={isRendering || isInteractiveBusy ? "default" : "secondary"}
                  className={cn("uppercase tracking-[0.08em]", (isRendering || isInteractiveBusy) && "animate-pulse")}
                >
                  {isRendering || isInteractiveBusy ? "Rendering" : "Ready"}
                </Badge>
              </div>

              <div className="metric-strip">
                <MetricPill
                  label="Latency"
                  value={`${renderMs.toFixed(1)} ms`}
                  icon={Timer}
                />
                <MetricPill label="Effects On" value={activeEffectCount.toString()} icon={WandSparkles} />
                <MetricPill label="Renderer Keys" value={rendererFields.length.toString()} icon={Cpu} />
                <MetricPill label="Path" value={wasCached ? "Cache hit" : "Live render"} icon={Activity} />
              </div>

              {interactiveMode ? (
                <div className="interactive-controls">
                  <div className="interactive-controls-head">
                    <div className="inline-flex items-center gap-2">
                      <Gamepad2 className="h-4 w-4" />
                      <span className="text-xs font-semibold uppercase tracking-[0.12em] text-muted-foreground">Interactive Controls</span>
                    </div>
                    <Badge variant="outline">
                      {interactiveActionCount > 0 ? `${interactiveActionCount} actions` : "No actions"}
                    </Badge>
                  </div>
                  <div className="interactive-controls-grid">
                    <div className="space-y-2">
                      <Label htmlFor="interactive-action" className="text-xs uppercase tracking-[0.12em] text-muted-foreground">
                        Action
                      </Label>
                      <Select
                        value={interactiveActionCount > 0 ? String(interactiveSelectedAction) : ""}
                        onValueChange={(value) => {
                          const parsed = Number(value);
                          if (Number.isFinite(parsed)) {
                            setInteractiveSelectedAction(parsed);
                          }
                        }}
                        disabled={interactiveActionCount <= 0 || isInteractiveBusy}
                      >
                        <SelectTrigger id="interactive-action" className="h-10 bg-card/80">
                          <SelectValue placeholder="Select action" />
                        </SelectTrigger>
                        <SelectContent>
                          {interactiveActionLabels.map((label, index) => (
                            <SelectItem key={`interactive-action-${index}`} value={String(index)}>
                              {index}: {label}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="interactive-buttons">
                      <Button
                        type="button"
                        variant="secondary"
                        className="h-10 gap-2"
                        onClick={() => {
                          enqueueInteractiveAction(interactiveSelectedAction);
                        }}
                        disabled={interactiveActionCount <= 0 || isInteractiveBusy}
                      >
                        <SkipForward className="h-4 w-4" />
                        Step
                      </Button>
                      <Button
                        type="button"
                        variant="secondary"
                        className="h-10 gap-2"
                        onClick={() => {
                          if (interactiveActionCount <= 0) {
                            return;
                          }
                          const action = Math.floor(Math.random() * interactiveActionCount);
                          setInteractiveSelectedAction(action);
                          enqueueInteractiveAction(action);
                        }}
                        disabled={interactiveActionCount <= 0 || isInteractiveBusy}
                      >
                        <Shuffle className="h-4 w-4" />
                        Random Action
                      </Button>
                      <Button
                        type="button"
                        variant="secondary"
                        className="h-10 gap-2"
                        onClick={() => void handleInteractiveReset()}
                        disabled={isInteractiveBusy || !interactiveSessionId}
                      >
                        <RotateCcw className="h-4 w-4" />
                        Reset Episode
                      </Button>
                    </div>
                  </div>
                </div>
              ) : null}
            </CardHeader>

            <CardContent className="visual-card-content">
              <div
                ref={previewFrameRef}
                className={cn("render-frame", interactiveMode && interactiveSessionId && "interactive-input-surface")}
                tabIndex={interactiveMode && interactiveSessionId ? 0 : -1}
                role={interactiveMode && interactiveSessionId ? "application" : undefined}
                aria-label={interactiveMode && interactiveSessionId ? "Interactive environment preview" : undefined}
                onKeyDown={handlePreviewKeyDown}
                onKeyUp={handlePreviewKeyUp}
                onPointerDown={handlePreviewPointerDown}
                onPointerUp={handlePreviewPointerUp}
                onPointerCancel={handlePreviewPointerCancel}
                onPointerLeave={handlePreviewPointerCancel}
                onClick={handlePreviewClick}
                onDoubleClick={handlePreviewDoubleClick}
                onAuxClick={handlePreviewAuxClick}
                onContextMenu={handlePreviewContextMenu}
                onWheel={handlePreviewWheel}
              >
                {imageSrc ? (
                  <>
                    <img src={imageSrc} alt="" aria-hidden="true" className="render-image-backdrop" />
                    <img src={imageSrc} alt="SPAR render output" className="render-image" />
                  </>
                ) : (
                  <div className="render-placeholder">
                    <Sparkles className="h-8 w-8" />
                    <p>{isBootstrapping ? "Initializing environment..." : "Waiting for first render"}</p>
                  </div>
                )}
              </div>
              <p className="hint-text">
                {interactiveMode && interactiveSessionId
                  ? "Interactive mode: click preview, then use the environment keyboard and mouse bindings."
                  : "Shortcuts: press "}
                {interactiveMode && interactiveSessionId ? null : (
                  <>
                    <kbd>R</kbd> to randomize and <kbd>E</kbd> to export.
                  </>
                )}
              </p>
            </CardContent>
          </Card>

          {lastError ? (
            <Card className="border-destructive/40 bg-destructive/5">
              <CardHeader>
                <CardTitle className="text-sm text-destructive">Render Error</CardTitle>
                <CardDescription>{lastError}</CardDescription>
              </CardHeader>
            </Card>
          ) : null}
        </section>

        <section className="right-column">
          <Tabs defaultValue="effects" className="flex h-full min-h-0 flex-col">
            <TabsList className="grid w-full shrink-0 grid-cols-4 bg-card/70">
              <TabsTrigger value="effects" className="gap-2">
                <WandSparkles className="h-4 w-4" /> Effects
              </TabsTrigger>
              <TabsTrigger value="sweep" className="gap-2">
                <Grid3x3 className="h-4 w-4" /> Sweep
              </TabsTrigger>
              <TabsTrigger value="renderer" className="gap-2">
                <Palette className="h-4 w-4" /> Renderer
              </TabsTrigger>
              <TabsTrigger value="history" className="gap-2">
                <Layers className="h-4 w-4" /> History
              </TabsTrigger>
            </TabsList>

            <TabsContent value="effects" className="mt-4 min-h-0 flex-1 data-[state=inactive]:hidden">
              <Card className="flex h-full min-h-0 flex-col border-border/70 bg-card/70">
                <CardHeader className="shrink-0">
                  <CardTitle className="text-xl">Effect Stack</CardTitle>
                  <CardDescription>Enable effects by stage and edit parameters from each effect signature.</CardDescription>
                </CardHeader>
                <CardContent className="min-h-0 flex-1">
                  <ScrollArea className="h-full pr-4">
                    <Accordion type="multiple" defaultValue={STAGE_ORDER} className="space-y-3">
                      {stagedEffects.map((stageInfo) => (
                        <AccordionItem value={stageInfo.stage} key={stageInfo.stage} className="rounded-2xl border border-border/60 bg-card/40 px-4">
                          <AccordionTrigger className="py-4 text-left">
                            <div className="flex items-center gap-2">
                              <Badge variant="outline">{stageInfo.effects.length}</Badge>
                              <span className="font-medium">{stageInfo.label}</span>
                            </div>
                          </AccordionTrigger>
                          <AccordionContent className="space-y-3 pb-4">
                            {stageInfo.effects.length === 0 ? (
                              <div className="rounded-xl border border-dashed border-border/80 p-4 text-sm text-muted-foreground">
                                No effects available for this stage.
                              </div>
                            ) : (
                              stageInfo.effects.map((effect) => {
                                const effectState: EffectEntry | undefined = effectsStore[stageInfo.stage]?.[effect.name];
                                const isEnabled = Boolean(effectState?.enabled);
                                return (
                                  <div key={`${stageInfo.stage}-${effect.name}`} className="rounded-xl border border-border/70 bg-card/80 p-4">
                                    <div className="mb-3 flex flex-wrap items-start justify-between gap-3">
                                      <div className="space-y-1">
                                        <h3 className="text-sm font-semibold uppercase tracking-[0.08em]">{effect.name}</h3>
                                        <p className="text-sm text-muted-foreground">{effect.description ?? "No description provided."}</p>
                                        <div className="flex flex-wrap gap-2">
                                          <Badge variant="secondary">{effect.category}</Badge>
                                          {effect.performance ? <Badge variant="outline">Perf {String(effect.performance)}</Badge> : null}
                                          {effect.requires_rng ? <Badge variant="outline">RNG</Badge> : null}
                                        </div>
                                      </div>
                                      <div className="flex items-center gap-2">
                                        <Label className="text-xs uppercase tracking-[0.1em] text-muted-foreground">Active</Label>
                                        <Switch
                                          checked={isEnabled}
                                          onCheckedChange={(next) => {
                                            updateEffectEnabled(stageInfo.stage, effect.name, next);
                                          }}
                                        />
                                      </div>
                                    </div>

                                    {(effect.parameters ?? []).length > 0 ? (
                                      <div className="grid gap-3">
                                        {(effect.parameters ?? []).map((parameter) => (
                                          <ParameterField
                                            key={`${stageInfo.stage}-${effect.name}-${parameter.name}`}
                                            stage={stageInfo.stage}
                                            effectName={effect.name}
                                            param={parameter}
                                            value={effectState?.params?.[parameter.name]}
                                            disabled={!isEnabled}
                                            onParamChange={updateEffectParam}
                                          />
                                        ))}
                                      </div>
                                    ) : (
                                      <p className="text-sm text-muted-foreground">This effect has no configurable parameters.</p>
                                    )}
                                  </div>
                                );
                              })
                            )}
                          </AccordionContent>
                        </AccordionItem>
                      ))}
                    </Accordion>
                  </ScrollArea>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="sweep" className="mt-4 min-h-0 flex-1 data-[state=inactive]:hidden">
              <SweepPanel
                environment={environment}
                statePayload={statePayload}
                effectsStore={effectsStore}
                rendererStore={rendererStore}
                disabled={isBootstrapping || interactiveMode}
                onApplyPreset={applyPresetToEffects}
              />
            </TabsContent>

            <TabsContent value="renderer" className="mt-4 min-h-0 flex-1 data-[state=inactive]:hidden">
              <Card className="flex h-full min-h-0 flex-col border-border/70 bg-card/70">
                <CardHeader className="shrink-0">
                  <CardTitle className="text-xl">Renderer Controls</CardTitle>
                  <CardDescription>Adjust renderer parameters using control types inferred from current values.</CardDescription>
                </CardHeader>
                <CardContent className="min-h-0 flex-1">
                  <ScrollArea className="h-full pr-4">
                    <div className="space-y-4">
                      {rendererFields.map(([key, rawValue]) => {
                        const prettyKey = key.replaceAll("_", " ");
                        const controlKind = getRendererControlKind(rawValue);
                        const isDpi = key === "dpi";
                        const isSize = key === "size";
                        const isSliderControl = controlKind === "number" && (isDpi || isSize);

                        if (isSliderControl) {
                          const bounds = isDpi
                            ? { min: 80, max: 400, step: 10, fallback: 150 }
                            : { min: 0.3, max: 3.5, step: 0.05, fallback: 1.5 };
                          const numericValue = numericOrFallback(rawValue, bounds.fallback);

                          return (
                            <div key={key} className="space-y-3 rounded-xl border border-border/70 bg-card/50 p-4">
                              <div className="flex items-center justify-between gap-2">
                                <Label className="text-xs uppercase tracking-[0.1em] text-muted-foreground">{prettyKey}</Label>
                                <Badge variant="secondary">{numericValue.toFixed(isDpi ? 0 : 2)}</Badge>
                              </div>
                              <Slider
                                min={bounds.min}
                                max={bounds.max}
                                step={bounds.step}
                                value={[numericValue]}
                                onValueChange={(values) => {
                                  updateRendererValue(key, values[0] ?? numericValue);
                                }}
                              />
                              <Input
                                type="number"
                                value={numericValue}
                                step={bounds.step}
                                min={bounds.min}
                                max={bounds.max}
                                onChange={(event) => {
                                  const asText = event.target.value;
                                  if (asText.trim() === "") {
                                    updateRendererValue(key, "");
                                    return;
                                  }
                                  const next = Number(event.target.value);
                                  if (!Number.isFinite(next)) {
                                    return;
                                  }
                                  updateRendererValue(key, next);
                                }}
                              />
                            </div>
                          );
                        }

                        if (controlKind === "number") {
                          const fallback = Number.isInteger(rawValue) ? 0 : 0.0;
                          const numericValue = numericOrFallback(rawValue, fallback);
                          const step = Number.isInteger(numericValue) ? 1 : 0.01;
                          return (
                            <div key={key} className="space-y-2 rounded-xl border border-border/70 bg-card/50 p-4">
                              <Label className="text-xs uppercase tracking-[0.1em] text-muted-foreground">{prettyKey}</Label>
                              <Input
                                type="number"
                                value={numericValue}
                                step={step}
                                onChange={(event) => {
                                  const asText = event.target.value;
                                  if (asText.trim() === "") {
                                    updateRendererValue(key, "");
                                    return;
                                  }
                                  const next = Number(asText);
                                  if (!Number.isFinite(next)) {
                                    return;
                                  }
                                  updateRendererValue(key, next);
                                }}
                              />
                            </div>
                          );
                        }

                        if (controlKind === "switch") {
                          return (
                            <div key={key} className="space-y-2 rounded-xl border border-border/70 bg-card/50 p-4">
                              <div className="flex items-center justify-between rounded-lg border border-border/60 bg-muted/30 px-3 py-2">
                                <Label className="text-xs uppercase tracking-[0.1em] text-muted-foreground">{prettyKey}</Label>
                                <Switch
                                  checked={Boolean(rawValue)}
                                  onCheckedChange={(next) => {
                                    updateRendererValue(key, next);
                                  }}
                                />
                              </div>
                            </div>
                          );
                        }

                        if (controlKind === "json") {
                          return (
                            <div key={key} className="space-y-2 rounded-xl border border-border/70 bg-card/50 p-4">
                              <Label className="text-xs uppercase tracking-[0.1em] text-muted-foreground">{prettyKey}</Label>
                              <Textarea
                                rows={4}
                                value={asJsonString(rawValue)}
                                onChange={(event) => {
                                  updateRendererValue(key, parseRendererJsonInput(event.target.value));
                                }}
                              />
                            </div>
                          );
                        }

                        return (
                          <div key={key} className="space-y-2 rounded-xl border border-border/70 bg-card/50 p-4">
                            <Label className="text-xs uppercase tracking-[0.1em] text-muted-foreground">{prettyKey}</Label>
                            <Input
                              type="text"
                              value={createInitialRendererValue(rawValue)}
                              onChange={(event) => {
                                const next = event.target.value;
                                const parsed = Number(next);
                                if (next !== "" && Number.isFinite(parsed)) {
                                  updateRendererValue(key, parsed);
                                  return;
                                }
                                updateRendererValue(key, next);
                              }}
                            />
                          </div>
                        );
                      })}

                      {rendererFields.length === 0 ? (
                        <div className="rounded-xl border border-dashed border-border/80 p-4 text-sm text-muted-foreground">
                          Renderer controls are not available for this environment.
                        </div>
                      ) : null}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="history" className="mt-4 min-h-0 flex-1 data-[state=inactive]:hidden">
              <Card className="flex h-full min-h-0 flex-col border-border/70 bg-card/70">
                <CardHeader className="shrink-0">
                  <div className="flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <CardTitle className="text-xl">History</CardTitle>
                      <CardDescription>Restore recent snapshots of state, effects, renderer settings, and image output.</CardDescription>
                    </div>
                    <Button
                      variant="ghost"
                      onClick={() => {
                        setHistory([]);
                      }}
                      disabled={history.length === 0}
                    >
                      Clear
                    </Button>
                  </div>
                </CardHeader>
                <CardContent className="min-h-0 flex-1">
                  <ScrollArea className="h-full pr-4">
                    <div className="space-y-4">
                      {history.length === 0 ? (
                        <div className="rounded-xl border border-dashed border-border/80 p-6 text-sm text-muted-foreground">
                          History is empty. Start adjusting controls to create snapshots.
                        </div>
                      ) : (
                        history.map((entry) => <HistoryCard key={entry.id} entry={entry} onRestore={restoreHistory} />)
                      )}
                    </div>
                  </ScrollArea>
                </CardContent>
              </Card>
            </TabsContent>
          </Tabs>
        </section>
      </main>

      <footer className="dashboard-footer">
        <Separator className="mb-4" />
        <div className="flex flex-wrap items-center justify-between gap-3 text-xs uppercase tracking-[0.08em] text-muted-foreground">
          <p>SPAR • Data Dashboard</p>
          <div className="flex items-center gap-2">
            <Gauge className="h-3.5 w-3.5" />
            <span>Render debounce {RENDER_DEBOUNCE_MS}ms</span>
          </div>
        </div>
      </footer>

      <Toaster position="top-right" richColors closeButton />
    </div>
  );
}

export default App;
