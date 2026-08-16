import { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ChangeEvent } from "react";
import { ClipboardCopy, Download, Grid3x3, Loader2, Sparkles, Upload, Wand2 } from "lucide-react";
import { toast } from "sonner";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip";
import {
  fetchConfigList,
  fetchConfigParse,
  fetchSweepPresets,
  fetchSweepRender,
  getErrorMessage,
  isAbortError,
} from "@/lib/dashboard-api";
import type {
  ConfigFileInfo,
  EffectEntry,
  EffectsStore,
  JSONValue,
  RendererStore,
  StatePayload,
  SweepCellRequest,
  SweepCellResult,
  SweepParamDescriptor,
  SweepPreset,
  SweepPresetEffect,
} from "@/lib/types";

const MAX_SWEEP_CELLS = 96;
const MIN_SAMPLES = 2;
const MAX_SAMPLES = 8;
const GEN_DATA_SOURCE = "gen_data";
const UPLOAD_SOURCE = "upload";

interface SweepPanelProps {
  environment: string;
  statePayload: StatePayload | null;
  effectsStore: EffectsStore;
  rendererStore: RendererStore;
  disabled: boolean;
  onApplyPreset: (preset: SweepPreset) => void;
}

type RangeBounds = Record<string, { min: number; max: number }>;

interface CellPlan {
  label: string;
  overrides: Record<string, JSONValue>;
}

function paramKey(effectName: string, paramName: string): string {
  return `${effectName}.${paramName}`;
}

function formatNumber(value: number): string {
  if (Number.isInteger(value)) {
    return value.toString();
  }
  return Number.parseFloat(value.toFixed(4)).toString();
}

function linspace(min: number, max: number, count: number): number[] {
  if (count <= 1 || min === max) {
    return [min];
  }
  const step = (max - min) / (count - 1);
  return Array.from({ length: count }, (_, index) => min + step * index);
}

function integerSamples(min: number, max: number, count: number): number[] {
  const low = Math.round(Math.min(min, max));
  const high = Math.round(Math.max(min, max));
  if (high <= low) {
    return [low];
  }
  const span = high - low;
  const limit = Math.min(count, span + 1);
  const seen = new Set<number>();
  const values: number[] = [];
  for (const raw of linspace(low, high, limit)) {
    const rounded = Math.round(raw);
    if (!seen.has(rounded)) {
      seen.add(rounded);
      values.push(rounded);
    }
  }
  return values;
}

function midpointValue(param: SweepParamDescriptor, bounds: RangeBounds, key: string): JSONValue {
  if (param.kind === "range") {
    const entry = bounds[key];
    if (entry) {
      return (entry.min + entry.max) / 2;
    }
    return ((param.low ?? 0) + (param.high ?? 0)) / 2;
  }
  if (param.kind === "irange") {
    const entry = bounds[key];
    const low = entry ? entry.min : (param.low ?? 0);
    const high = entry ? entry.max : (param.high ?? 0);
    return Math.round((low + high) / 2);
  }
  if (param.kind === "choice") {
    return param.options?.[0] ?? null;
  }
  return param.value ?? null;
}

function extremeValue(param: SweepParamDescriptor, bounds: RangeBounds, key: string, side: "min" | "max"): JSONValue {
  if (param.kind === "range" || param.kind === "irange") {
    const entry = bounds[key];
    const low = entry ? entry.min : (param.low ?? 0);
    const high = entry ? entry.max : (param.high ?? 0);
    const value = side === "min" ? Math.min(low, high) : Math.max(low, high);
    return param.kind === "irange" ? Math.round(value) : value;
  }
  if (param.kind === "choice") {
    const options = param.options ?? [];
    return side === "min" ? (options[0] ?? null) : (options[options.length - 1] ?? null);
  }
  return param.value ?? null;
}

function sweepValues(param: SweepParamDescriptor, bounds: RangeBounds, key: string, samples: number): JSONValue[] {
  if (param.kind === "range") {
    const entry = bounds[key];
    const min = entry ? entry.min : (param.low ?? 0);
    const max = entry ? entry.max : (param.high ?? 0);
    return linspace(min, max, samples);
  }
  if (param.kind === "irange") {
    const entry = bounds[key];
    const min = entry ? entry.min : (param.low ?? 0);
    const max = entry ? entry.max : (param.high ?? 0);
    return integerSamples(min, max, samples);
  }
  if (param.kind === "choice") {
    return (param.options ?? []).slice(0, MAX_SAMPLES);
  }
  return [param.value ?? null];
}

function sweepableParams(preset: SweepPreset): Array<{ effect: SweepPresetEffect; param: SweepParamDescriptor }> {
  const entries: Array<{ effect: SweepPresetEffect; param: SweepParamDescriptor }> = [];
  for (const effect of preset.effects) {
    for (const param of effect.params) {
      if (param.kind === "range" || param.kind === "irange" || param.kind === "choice") {
        entries.push({ effect, param });
      }
    }
  }
  return entries;
}

function overrideValueLabel(value: JSONValue): string {
  if (typeof value === "number") {
    return formatNumber(value);
  }
  if (value === null) {
    return "null";
  }
  return String(value);
}

interface SweepControlsProps {
  preset: SweepPreset;
  bounds: RangeBounds;
  onBoundChange: (key: string, edge: "min" | "max", value: number) => void;
}

const SweepControls = memo(function SweepControls({ preset, bounds, onBoundChange }: SweepControlsProps) {
  return (
    <div className="space-y-3">
      {preset.effects.map((effect) => (
        <div key={effect.name} className="rounded-xl border border-border/60 bg-card/50 p-3">
          <div className="mb-2 flex items-center gap-2">
            <Badge variant="secondary">{effect.stage.replace("_", " ").toLowerCase()}</Badge>
            <span className="text-sm font-semibold uppercase tracking-[0.08em]">{effect.name}</span>
          </div>
          <div className="grid gap-2">
            {effect.params.map((param) => {
              const key = paramKey(effect.name, param.name);
              if (param.kind === "range" || param.kind === "irange") {
                const entry = bounds[key];
                const min = entry ? entry.min : (param.low ?? 0);
                const max = entry ? entry.max : (param.high ?? 0);
                const step = param.kind === "irange" ? 1 : 0.01;
                return (
                  <div key={key} className="grid grid-cols-[1fr_auto_auto] items-center gap-2">
                    <Label className="text-xs text-muted-foreground">{param.label}</Label>
                    <Input
                      type="number"
                      className="h-8 w-24"
                      step={step}
                      value={min}
                      onChange={(event) => {
                        const next = Number(event.target.value);
                        if (Number.isFinite(next)) {
                          onBoundChange(key, "min", next);
                        }
                      }}
                    />
                    <Input
                      type="number"
                      className="h-8 w-24"
                      step={step}
                      value={max}
                      onChange={(event) => {
                        const next = Number(event.target.value);
                        if (Number.isFinite(next)) {
                          onBoundChange(key, "max", next);
                        }
                      }}
                    />
                  </div>
                );
              }
              return (
                <div key={key} className="flex items-center justify-between gap-2">
                  <Label className="text-xs text-muted-foreground">{param.label}</Label>
                  <Badge variant="outline">{(param.options ?? []).length} options</Badge>
                </div>
              );
            })}
            {effect.params.every((param) => param.kind === "fixed") ? (
              <p className="text-xs text-muted-foreground">Only fixed parameters. Included at their preset values.</p>
            ) : null}
          </div>
        </div>
      ))}
    </div>
  );
});

function buildCellEffects(
  base: EffectsStore,
  preset: SweepPreset,
  bounds: RangeBounds,
  overrides: Record<string, JSONValue>,
): EffectsStore {
  const store: EffectsStore = structuredClone(base);
  for (const effect of preset.effects) {
    const stageStore: Record<string, EffectEntry> = store[effect.stage] ?? (store[effect.stage] = {});
    const existing: EffectEntry = stageStore[effect.name] ?? { enabled: false, params: {} };
    const params: Record<string, JSONValue> = { ...existing.params };
    for (const param of effect.params) {
      const key = paramKey(effect.name, param.name);
      params[param.name] = key in overrides ? overrides[key] : midpointValue(param, bounds, key);
    }
    stageStore[effect.name] = { enabled: true, params };
  }
  return store;
}

function planCells(preset: SweepPreset, bounds: RangeBounds, samples: number, includeCorners: boolean): CellPlan[] {
  const sweepables = sweepableParams(preset);
  const plans: CellPlan[] = [];
  const seen = new Set<string>();

  const pushPlan = (label: string, overrides: Record<string, JSONValue>) => {
    const signature = JSON.stringify(Object.entries(overrides).sort(([a], [b]) => a.localeCompare(b)));
    if (seen.has(signature)) {
      return;
    }
    seen.add(signature);
    plans.push({ label, overrides });
  };

  pushPlan("baseline (midpoints)", {});

  for (const { effect, param } of sweepables) {
    const key = paramKey(effect.name, param.name);
    for (const value of sweepValues(param, bounds, key, samples)) {
      pushPlan(`${effect.name}.${param.name} = ${overrideValueLabel(value)}`, { [key]: value });
    }
  }

  if (includeCorners && sweepables.length > 1) {
    const minOverrides: Record<string, JSONValue> = {};
    const maxOverrides: Record<string, JSONValue> = {};
    for (const { effect, param } of sweepables) {
      const key = paramKey(effect.name, param.name);
      minOverrides[key] = extremeValue(param, bounds, key, "min");
      maxOverrides[key] = extremeValue(param, bounds, key, "max");
    }
    pushPlan("all parameters at minimum", minOverrides);
    pushPlan("all parameters at maximum", maxOverrides);
  }

  return plans.slice(0, MAX_SWEEP_CELLS);
}

function yamlScalar(value: JSONValue | undefined): string {
  if (value === null || value === undefined) {
    return "null";
  }
  if (typeof value === "number") {
    return formatNumber(value);
  }
  if (typeof value === "boolean") {
    return value ? "true" : "false";
  }
  if (typeof value === "string") {
    return JSON.stringify(value);
  }
  // Arrays and objects are emitted as JSON, which is valid YAML flow syntax.
  return JSON.stringify(value);
}

function yamlParamValue(param: SweepParamDescriptor, bounds: RangeBounds, key: string): string {
  if (param.kind === "range") {
    const entry = bounds[key];
    const min = entry ? entry.min : (param.low ?? 0);
    const max = entry ? entry.max : (param.high ?? 0);
    return `\${range:${formatNumber(min)},${formatNumber(max)}}`;
  }
  if (param.kind === "irange") {
    const entry = bounds[key];
    const min = entry ? entry.min : (param.low ?? 0);
    const max = entry ? entry.max : (param.high ?? 0);
    return `\${irange:${Math.round(min)},${Math.round(max)}}`;
  }
  if (param.kind === "choice") {
    const options = (param.options ?? []).map((option) => yamlScalar(option));
    return `\${choice:[${options.join(",")}]}`;
  }
  return yamlScalar(param.value ?? null);
}

function buildYaml(preset: SweepPreset, bounds: RangeBounds): string {
  const lines: string[] = [`${preset.name}:`, "  enabled: true"];
  if (preset.is_leaf) {
    const [effect] = preset.effects;
    if (effect) {
      for (const param of effect.params) {
        lines.push(`  ${param.name}: ${yamlParamValue(param, bounds, paramKey(effect.name, param.name))}`);
      }
    }
    return lines.join("\n");
  }
  for (const effect of preset.effects) {
    lines.push(`  ${effect.name}:`);
    for (const param of effect.params) {
      lines.push(`    ${param.name}: ${yamlParamValue(param, bounds, paramKey(effect.name, param.name))}`);
    }
  }
  return lines.join("\n");
}

const SweepCellCard = memo(function SweepCellCard({ cell }: { cell: SweepCellResult }) {
  return (
    <div className="overflow-hidden rounded-xl border border-border/70 bg-card/70">
      <div className="history-preview-wrap">
        <img src={cell.image} alt={cell.label} className="history-preview" loading="lazy" />
      </div>
      <div className="flex items-center justify-between gap-2 px-3 py-2">
        <Tooltip>
          <TooltipTrigger asChild>
            <span className="min-w-0 flex-1 truncate text-left text-xs text-foreground/90">{cell.label}</span>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="max-w-xs break-words">
            {cell.label}
          </TooltipContent>
        </Tooltip>
        <Badge variant={cell.cached ? "secondary" : "outline"}>{cell.cached ? "cache" : `${cell.render_ms.toFixed(0)}ms`}</Badge>
      </div>
    </div>
  );
});

function initialBoundsFor(preset: SweepPreset): RangeBounds {
  const bounds: RangeBounds = {};
  for (const effect of preset.effects) {
    for (const param of effect.params) {
      if (param.kind === "range" || param.kind === "irange") {
        bounds[paramKey(effect.name, param.name)] = { min: param.low ?? 0, max: param.high ?? 0 };
      }
    }
  }
  return bounds;
}

function SweepPanel({ environment, statePayload, effectsStore, rendererStore, disabled, onApplyPreset }: SweepPanelProps) {
  const [presets, setPresets] = useState<SweepPreset[]>([]);
  const [presetName, setPresetName] = useState<string>("");
  const [boundsPresetName, setBoundsPresetName] = useState<string>("");
  const [bounds, setBounds] = useState<RangeBounds>({});
  const [samples, setSamples] = useState<number>(3);
  const [includeCorners, setIncludeCorners] = useState<boolean>(true);
  const [cells, setCells] = useState<SweepCellResult[]>([]);
  const [configFiles, setConfigFiles] = useState<ConfigFileInfo[]>([]);
  const [configSource, setConfigSource] = useState<string>(GEN_DATA_SOURCE);
  const [uploadLabel, setUploadLabel] = useState<string>("");
  const [isLoadingPresets, setIsLoadingPresets] = useState<boolean>(false);
  const [isRendering, setIsRendering] = useState<boolean>(false);
  const [error, setError] = useState<string>("");

  const renderAbortRef = useRef<AbortController | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const selectedPreset = useMemo(
    () => presets.find((preset) => preset.name === presetName) ?? null,
    [presetName, presets],
  );

  // Discover the packaged configs for the environment and reset the selected
  // source whenever the environment changes.
  useEffect(() => {
    if (!environment) {
      return;
    }
    setConfigSource(GEN_DATA_SOURCE);
    setUploadLabel("");
    const abortController = new AbortController();
    void (async () => {
      try {
        const response = await fetchConfigList(environment, abortController.signal);
        if (!abortController.signal.aborted) {
          setConfigFiles(response.files);
        }
      } catch (caught) {
        if (!isAbortError(caught)) {
          setConfigFiles([]);
        }
      }
    })();
    return () => {
      abortController.abort();
    };
  }, [environment]);

  // Load presets from the active source. Uploaded configs are already parsed by
  // the upload handler, so they are kept as-is without a refetch.
  useEffect(() => {
    if (!environment || configSource === UPLOAD_SOURCE) {
      return;
    }
    const abortController = new AbortController();
    const load = async () => {
      setIsLoadingPresets(true);
      setError("");
      try {
        const response =
          configSource === GEN_DATA_SOURCE
            ? await fetchSweepPresets(environment, abortController.signal)
            : await fetchConfigParse(environment, { token: configSource }, abortController.signal);
        if (abortController.signal.aborted) {
          return;
        }
        setPresets(response.presets);
        setPresetName(response.presets[0]?.name ?? "");
        setCells([]);
      } catch (caught) {
        if (!isAbortError(caught)) {
          setError(getErrorMessage(caught));
        }
      } finally {
        if (!abortController.signal.aborted) {
          setIsLoadingPresets(false);
        }
      }
    };
    void load();
    return () => {
      abortController.abort();
    };
  }, [environment, configSource]);

  const handleUpload = useCallback(
    (event: ChangeEvent<HTMLInputElement>) => {
      const file = event.target.files?.[0];
      event.target.value = "";
      if (!file || !environment) {
        return;
      }
      setIsLoadingPresets(true);
      setError("");
      void file
        .text()
        .then((content) => fetchConfigParse(environment, { content }))
        .then((response) => {
          setPresets(response.presets);
          setPresetName(response.presets[0]?.name ?? "");
          setCells([]);
          setUploadLabel(file.name);
          setConfigSource(UPLOAD_SOURCE);
          if (response.presets.length === 0) {
            toast.message("No effect presets were found in the uploaded config");
          }
        })
        .catch((caught) => {
          const message = getErrorMessage(caught);
          setError(message);
          toast.error(message);
        })
        .finally(() => {
          setIsLoadingPresets(false);
        });
    },
    [environment],
  );

  const handleApplyToEffects = useCallback(() => {
    if (selectedPreset) {
      onApplyPreset(selectedPreset);
    }
  }, [onApplyPreset, selectedPreset]);

  // Reset editable bounds during the first render after a preset change.
  // Doing this during render avoids a synchronous state update from an effect.
  let effectiveBounds: RangeBounds = bounds;
  if (selectedPreset && boundsPresetName !== selectedPreset.name) {
    effectiveBounds = initialBoundsFor(selectedPreset);
    setBounds(effectiveBounds);
    setBoundsPresetName(selectedPreset.name);
    setCells([]);
  } else if (!selectedPreset && boundsPresetName !== "") {
    effectiveBounds = {};
    setBounds({});
    setBoundsPresetName("");
    setCells([]);
  }

  const handleBoundChange = useCallback((key: string, edge: "min" | "max", value: number) => {
    setBounds((previous) => ({
      ...previous,
      [key]: {
        min: edge === "min" ? value : (previous[key]?.min ?? value),
        max: edge === "max" ? value : (previous[key]?.max ?? value),
      },
    }));
  }, []);

  const plannedCount = useMemo(() => {
    if (!selectedPreset) {
      return 0;
    }
    return planCells(selectedPreset, bounds, samples, includeCorners).length;
  }, [bounds, includeCorners, samples, selectedPreset]);

  const handleGenerate = useCallback(() => {
    if (!selectedPreset || statePayload === null) {
      return;
    }
    const plans = planCells(selectedPreset, bounds, samples, includeCorners);
    const requestCells: SweepCellRequest[] = plans.map((plan) => ({
      label: plan.label,
      effects: buildCellEffects(effectsStore, selectedPreset, bounds, plan.overrides),
    }));

    if (renderAbortRef.current) {
      renderAbortRef.current.abort();
    }
    const abortController = new AbortController();
    renderAbortRef.current = abortController;
    setIsRendering(true);
    setError("");

    void fetchSweepRender(environment, statePayload, rendererStore, requestCells, abortController.signal)
      .then((response) => {
        if (abortController.signal.aborted) {
          return;
        }
        setCells(response.cells);
      })
      .catch((caught) => {
        if (isAbortError(caught)) {
          return;
        }
        const message = getErrorMessage(caught);
        setError(message);
        toast.error(message);
      })
      .finally(() => {
        if (!abortController.signal.aborted) {
          setIsRendering(false);
        }
      });
  }, [bounds, effectsStore, environment, includeCorners, rendererStore, samples, selectedPreset, statePayload]);

  useEffect(() => {
    return () => {
      if (renderAbortRef.current) {
        renderAbortRef.current.abort();
      }
    };
  }, []);

  const handleCopyYaml = useCallback(() => {
    if (!selectedPreset) {
      return;
    }
    const yaml = buildYaml(selectedPreset, bounds);
    void navigator.clipboard
      .writeText(yaml)
      .then(() => {
        toast.success("Copied gen_data YAML to clipboard");
      })
      .catch(() => {
        toast.error("Clipboard is unavailable in this browser");
      });
  }, [bounds, selectedPreset]);

  const handleDownloadYaml = useCallback(() => {
    if (!selectedPreset) {
      return;
    }
    const yaml = buildYaml(selectedPreset, bounds);
    const blob = new Blob([`${yaml}\n`], { type: "text/yaml" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${environment}-${selectedPreset.name}.yaml`;
    anchor.click();
    URL.revokeObjectURL(url);
  }, [bounds, environment, selectedPreset]);

  const controlsDisabled = disabled || isLoadingPresets || !selectedPreset;

  return (
    <Card className="flex h-full min-h-0 flex-col border-border/70 bg-card/70">
      <CardHeader className="shrink-0">
        <CardTitle className="text-xl">Config Sweep</CardTitle>
        <CardDescription>
          Load a packaged or uploaded config to extract its effects and ranges, sweep every range, then apply the preset
          to the effect stack or export it.
        </CardDescription>
      </CardHeader>
      <CardContent className="min-h-0 flex-1">
        <ScrollArea className="h-full pr-4">
          <div className="space-y-4">
            <div className="space-y-2">
              <Label className="text-xs uppercase tracking-[0.12em] text-muted-foreground">Config source</Label>
              <div className="flex flex-wrap items-center gap-2">
                <Select value={configSource} onValueChange={setConfigSource} disabled={disabled || isLoadingPresets}>
                  <SelectTrigger className="h-10 flex-1 bg-card/80">
                    <SelectValue placeholder="Select config source" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={GEN_DATA_SOURCE}>Built-in gen_data</SelectItem>
                    {configFiles.map((file) => (
                      <SelectItem key={file.token} value={file.token}>
                        {file.label} ({file.preset_count})
                      </SelectItem>
                    ))}
                    {configSource === UPLOAD_SOURCE ? (
                      <SelectItem value={UPLOAD_SOURCE}>Uploaded: {uploadLabel}</SelectItem>
                    ) : null}
                  </SelectContent>
                </Select>
                <input ref={fileInputRef} type="file" accept=".yaml,.yml" className="hidden" onChange={handleUpload} />
                <Button
                  type="button"
                  variant="secondary"
                  className="gap-2"
                  onClick={() => fileInputRef.current?.click()}
                  disabled={disabled || isLoadingPresets}
                >
                  <Upload className="h-4 w-4" />
                  Upload
                </Button>
              </div>
              {configSource === UPLOAD_SOURCE && uploadLabel ? (
                <Badge variant="outline">Loaded from {uploadLabel}</Badge>
              ) : null}
            </div>

            <div className="space-y-2">
              <Label className="text-xs uppercase tracking-[0.12em] text-muted-foreground">Preset</Label>
              <Select value={presetName} onValueChange={setPresetName} disabled={disabled || isLoadingPresets || presets.length === 0}>
                <SelectTrigger className="h-10 bg-card/80">
                  <SelectValue placeholder={presets.length === 0 ? "No gen_data presets" : "Select preset"} />
                </SelectTrigger>
                <SelectContent>
                  {presets.map((preset) => (
                    <SelectItem key={preset.name} value={preset.name}>
                      {preset.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {selectedPreset ? (
              <>
                <SweepControls preset={selectedPreset} bounds={effectiveBounds} onBoundChange={handleBoundChange} />

                <Separator />

                <div className="grid grid-cols-[1fr_auto] items-center gap-3">
                  <div className="space-y-1">
                    <Label className="text-xs uppercase tracking-[0.12em] text-muted-foreground">Samples per axis</Label>
                    <Input
                      type="number"
                      className="h-9 w-24"
                      min={MIN_SAMPLES}
                      max={MAX_SAMPLES}
                      value={samples}
                      onChange={(event) => {
                        const next = Number(event.target.value);
                        if (Number.isFinite(next)) {
                          setSamples(Math.min(MAX_SAMPLES, Math.max(MIN_SAMPLES, Math.round(next))));
                        }
                      }}
                    />
                  </div>
                  <div className="flex items-center gap-2">
                    <Label className="text-xs text-muted-foreground">Extreme corners</Label>
                    <Switch checked={includeCorners} onCheckedChange={setIncludeCorners} />
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-2">
                  <Button type="button" className="gap-2" onClick={handleGenerate} disabled={controlsDisabled || isRendering || statePayload === null}>
                    {isRendering ? <Loader2 className="h-4 w-4 animate-spin" /> : <Grid3x3 className="h-4 w-4" />}
                    Generate grid ({plannedCount})
                  </Button>
                  <Button type="button" variant="default" className="gap-2" onClick={handleApplyToEffects} disabled={controlsDisabled}>
                    <Wand2 className="h-4 w-4" />
                    Apply to Effects
                  </Button>
                  <Button type="button" variant="secondary" className="gap-2" onClick={handleCopyYaml} disabled={controlsDisabled}>
                    <ClipboardCopy className="h-4 w-4" />
                    Copy YAML
                  </Button>
                  <Button type="button" variant="secondary" className="gap-2" onClick={handleDownloadYaml} disabled={controlsDisabled}>
                    <Download className="h-4 w-4" />
                    Download YAML
                  </Button>
                </div>

                {error ? <p className="text-sm text-destructive">{error}</p> : null}

                {cells.length > 0 ? (
                  <TooltipProvider delayDuration={120}>
                    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
                      {cells.map((cell, index) => (
                        <SweepCellCard key={`${cell.label}-${index}`} cell={cell} />
                      ))}
                    </div>
                  </TooltipProvider>
                ) : (
                  <div className="flex flex-col items-center gap-2 rounded-xl border border-dashed border-border/80 p-6 text-sm text-muted-foreground">
                    <Sparkles className="h-6 w-6" />
                    <p>Generate a grid to compare parameter extremes side by side.</p>
                  </div>
                )}
              </>
            ) : (
              <div className="rounded-xl border border-dashed border-border/80 p-6 text-sm text-muted-foreground">
                {isLoadingPresets ? "Loading effect presets..." : "No effect presets are available for this config source."}
              </div>
            )}
          </div>
        </ScrollArea>
      </CardContent>
    </Card>
  );
}

export default memo(SweepPanel);
