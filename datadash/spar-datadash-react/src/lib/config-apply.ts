import type {
  EffectEntry,
  EffectSpec,
  EffectSpecsByStage,
  EffectsStore,
  JSONValue,
  ParameterSpec,
  SweepParamDescriptor,
  SweepPreset,
} from "@/lib/types";

/**
 * Derive the type-and-range tailored input metadata for a single parameter from a
 * loaded config descriptor.
 *
 * A `range`/`irange` descriptor becomes a bounded numeric control, a `choice`
 * becomes a select with its allowed options, and a `fixed` value leaves the
 * base spec untouched (only the stored value changes).
 */
function paramSpecOverride(base: ParameterSpec, descriptor: SweepParamDescriptor): ParameterSpec {
  if (descriptor.kind === "range" || descriptor.kind === "irange") {
    const low = descriptor.low ?? 0;
    const high = descriptor.high ?? 0;
    const min = Math.min(low, high);
    const max = Math.max(low, high);
    return {
      ...base,
      kind: "number",
      min,
      max,
      step: descriptor.kind === "irange" ? 1 : (base.step ?? 0.05),
    };
  }
  if (descriptor.kind === "choice") {
    const options = (descriptor.options ?? []).map((option) => ({
      label: String(option),
      value: option as JSONValue,
    }));
    return { ...base, kind: "select", options };
  }
  return base;
}

/**
 * Compute the concrete value a parameter should take when a config is applied to
 * the effect stack, using the midpoint of numeric ranges and the first option of
 * a choice.
 */
function paramValueFromDescriptor(descriptor: SweepParamDescriptor): JSONValue {
  if (descriptor.kind === "range") {
    const low = descriptor.low ?? 0;
    const high = descriptor.high ?? 0;
    return (low + high) / 2;
  }
  if (descriptor.kind === "irange") {
    const low = descriptor.low ?? 0;
    const high = descriptor.high ?? 0;
    return Math.round((low + high) / 2);
  }
  if (descriptor.kind === "choice") {
    return (descriptor.options?.[0] ?? null) as JSONValue;
  }
  return descriptor.value ?? null;
}

/**
 * Merge a loaded preset's effect ranges and choices into the effect specs so the
 * Effects tab renders type-and-range tailored controls (bounded sliders, selects).
 *
 * Only effects and parameters already present in `specs` are updated, so a config
 * can never introduce an effect the environment does not support.
 */
export function mergePresetIntoSpecs(specs: EffectSpecsByStage, preset: SweepPreset): EffectSpecsByStage {
  const next: EffectSpecsByStage = { ...specs };
  for (const effect of preset.effects) {
    const stageSpecs = next[effect.stage];
    if (!stageSpecs) {
      continue;
    }
    const index = stageSpecs.findIndex((spec) => spec.name === effect.name);
    if (index < 0) {
      continue;
    }
    const spec: EffectSpec = stageSpecs[index];
    const descriptorByName = new Map(effect.params.map((descriptor) => [descriptor.name, descriptor]));
    const parameters = (spec.parameters ?? []).map((parameter) => {
      const descriptor = descriptorByName.get(parameter.name);
      return descriptor ? paramSpecOverride(parameter, descriptor) : parameter;
    });
    const nextStage = stageSpecs.slice();
    nextStage[index] = { ...spec, parameters };
    next[effect.stage] = nextStage;
  }
  return next;
}

/**
 * Enable a loaded preset's effects in the store and set each parameter to the
 * value implied by the config (range midpoints, first choice option, fixed value).
 */
export function applyPresetToStore(store: EffectsStore, preset: SweepPreset): EffectsStore {
  const next: EffectsStore = structuredClone(store);
  for (const effect of preset.effects) {
    const stageStore: Record<string, EffectEntry> = next[effect.stage] ?? (next[effect.stage] = {});
    const existing: EffectEntry = stageStore[effect.name] ?? { enabled: false, params: {} };
    const params: Record<string, JSONValue> = { ...existing.params };
    for (const descriptor of effect.params) {
      params[descriptor.name] = paramValueFromDescriptor(descriptor);
    }
    stageStore[effect.name] = { enabled: true, params };
  }
  return next;
}
