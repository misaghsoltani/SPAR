export type JSONPrimitive = string | number | boolean | null;
export type JSONValue = JSONPrimitive | JSONValue[] | { [key: string]: JSONValue };

export type StageName = "PRE_RENDER" | "OBJECT_RENDER" | "POST_RENDER";

export interface EnvironmentOption {
  label: string;
  value: string;
}

export interface ParameterSpec {
  name: string;
  label: string;
  annotation?: string;
  kind?: string;
  placeholder?: string;
  default?: JSONValue;
  step?: number;
  min?: number;
  max?: number;
  options?: Array<{ label: string; value: JSONValue }>;
}

export interface EffectSpec {
  name: string;
  stage?: string;
  category: string;
  description?: string;
  performance?: string | number;
  requires_rng?: boolean;
  target_param?: string | null;
  target_type?: string;
  parameters?: ParameterSpec[];
}

export type EffectSpecsByStage = Record<string, EffectSpec[]>;

export interface EffectEntry {
  enabled: boolean;
  params: Record<string, JSONValue>;
}

export type EffectsStore = Record<string, Record<string, EffectEntry>>;
export type RendererStore = Record<string, JSONValue>;
export type StatePayload = Record<string, JSONValue>;

export interface BootstrapResponse {
  env: string;
  effect_specs: EffectSpecsByStage;
  effects_store: EffectsStore;
  renderer: RendererStore;
  state: StatePayload;
}

export interface RenderResponse {
  image: string;
  render_ms: number;
  cached: boolean;
}

export interface EnvironmentsResponse {
  environments: EnvironmentOption[];
  default_env: string;
}

export interface RandomizeResponse {
  env: string;
  state: StatePayload;
}

export interface InteractiveBindings {
  version?: number;
  keyboard?: {
    enabled?: boolean;
    events?: string[];
    key_to_action?: Record<string, number>;
  };
  pointer?: {
    enabled?: boolean;
    events?: string[];
    directional?: Partial<Record<"up" | "down" | "left" | "right" | "noop", number>>;
    button_to_action?: Record<string, number>;
    event_to_action?: Record<string, number>;
    swipe_threshold?: number;
  };
  wheel?: {
    enabled?: boolean;
    events?: string[];
    vertical?: Partial<Record<"negative" | "positive", number>>;
    horizontal?: Partial<Record<"negative" | "positive", number>>;
  };
}

export interface InteractiveStartResponse {
  session_id: string;
  env: string;
  state: StatePayload;
  action_count: number;
  action_labels: string[];
  interactive_bindings: InteractiveBindings;
  image: string;
  render_ms: number;
}

export interface InteractiveRenderResponse {
  session_id: string;
  env: string;
  state: StatePayload;
  action_count: number;
  action_labels: string[];
  interactive_bindings: InteractiveBindings;
  image: string;
  render_ms: number;
}

export interface InteractiveStepResponse {
  session_id: string;
  env: string;
  state: StatePayload;
  action_applied: number;
  action_count: number;
  action_labels: string[];
  interactive_bindings: InteractiveBindings;
  image: string;
  render_ms: number;
}

export interface InteractiveEventPayload {
  kind: "keyboard" | "pointer" | "wheel";
  type: string;
  key?: string;
  code?: string;
  button?: number;
  pointer_id?: number;
  client_x?: number;
  client_y?: number;
  start_x?: number;
  start_y?: number;
  delta_x?: number;
  delta_y?: number;
}

export interface InteractiveEventResponse {
  session_id: string;
  env: string;
  state: StatePayload;
  action_count: number;
  action_labels: string[];
  interactive_bindings: InteractiveBindings;
  actions_applied: number[];
  handled: boolean;
  image?: string;
  render_ms?: number;
}

export interface InteractiveStopResponse {
  stopped: boolean;
}

export type SweepParamKind = "range" | "irange" | "choice" | "fixed";

export interface SweepParamDescriptor {
  name: string;
  label: string;
  kind: SweepParamKind;
  low?: number;
  high?: number;
  options?: Array<string | number | boolean | null>;
  value?: JSONValue;
}

export interface SweepPresetEffect {
  name: string;
  stage: StageName;
  params: SweepParamDescriptor[];
}

export interface SweepPreset {
  name: string;
  is_leaf: boolean;
  effects: SweepPresetEffect[];
}

export interface SweepPresetsResponse {
  env: string;
  presets: SweepPreset[];
}

export interface ConfigFileInfo {
  token: string;
  label: string;
  preset_count: number;
}

export interface ConfigListResponse {
  env: string;
  files: ConfigFileInfo[];
}

export interface ConfigParseResponse {
  env: string;
  source: string;
  presets: SweepPreset[];
}

export interface SweepCellRequest {
  label: string;
  effects: EffectsStore;
}

export interface SweepCellResult {
  label: string;
  image: string;
  render_ms: number;
  cached: boolean;
}

export interface SweepRenderResponse {
  env: string;
  cells: SweepCellResult[];
}

export interface HistoryEntry {
  id: string;
  requestKey: string;
  createdAt: string;
  env: string;
  image: string;
  state: StatePayload;
  effects: EffectsStore;
  renderer: RendererStore;
  effectSpecs: EffectSpecsByStage;
  renderMs: number;
  cached: boolean;
}
