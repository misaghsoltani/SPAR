import type {
  BootstrapResponse,
  ConfigListResponse,
  ConfigParseResponse,
  EnvironmentsResponse,
  InteractiveEventPayload,
  InteractiveEventResponse,
  InteractiveRenderResponse,
  InteractiveStartResponse,
  InteractiveStepResponse,
  InteractiveStopResponse,
  RandomizeResponse,
  RenderResponse,
  StatePayload,
  EffectsStore,
  RendererStore,
  SweepCellRequest,
  SweepPresetsResponse,
  SweepRenderResponse,
} from "@/lib/types";

const API_BASE = import.meta.env.VITE_SPAR_API_BASE ?? "/api";

class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function requestJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    let message = `Request failed (${response.status})`;
    try {
      const body = (await response.json()) as { error?: string; hint?: string };
      if (body.error) {
        message = body.hint ? `${body.error}: ${body.hint}` : body.error;
      }
    } catch {
      // keep default error message
    }
    throw new ApiError(message, response.status);
  }
  return (await response.json()) as T;
}

export async function fetchEnvironments(signal?: AbortSignal): Promise<EnvironmentsResponse> {
  return requestJSON<EnvironmentsResponse>("/environments", { method: "GET", signal });
}

export async function fetchBootstrap(env: string, signal?: AbortSignal): Promise<BootstrapResponse> {
  return requestJSON<BootstrapResponse>("/bootstrap", {
    method: "POST",
    body: JSON.stringify({ env }),
    signal,
  });
}

export async function fetchRandomize(env: string, signal?: AbortSignal): Promise<RandomizeResponse> {
  return requestJSON<RandomizeResponse>("/randomize", {
    method: "POST",
    body: JSON.stringify({ env }),
    signal,
  });
}

export async function fetchRender(
  env: string,
  state: StatePayload,
  effects: EffectsStore,
  renderer: RendererStore,
  signal?: AbortSignal,
): Promise<RenderResponse> {
  return requestJSON<RenderResponse>("/render", {
    method: "POST",
    body: JSON.stringify({
      env,
      state,
      effects,
      renderer,
    }),
    signal,
  });
}

export async function fetchSweepPresets(env: string, signal?: AbortSignal): Promise<SweepPresetsResponse> {
  return requestJSON<SweepPresetsResponse>("/sweep/presets", {
    method: "POST",
    body: JSON.stringify({ env }),
    signal,
  });
}

export async function fetchConfigList(env: string, signal?: AbortSignal): Promise<ConfigListResponse> {
  return requestJSON<ConfigListResponse>("/config/list", {
    method: "POST",
    body: JSON.stringify({ env }),
    signal,
  });
}

export async function fetchConfigParse(
  env: string,
  source: { token: string } | { content: string },
  signal?: AbortSignal,
): Promise<ConfigParseResponse> {
  return requestJSON<ConfigParseResponse>("/config/parse", {
    method: "POST",
    body: JSON.stringify({ env, ...source }),
    signal,
  });
}

export async function fetchSweepRender(
  env: string,
  state: StatePayload,
  renderer: RendererStore,
  cells: SweepCellRequest[],
  signal?: AbortSignal,
): Promise<SweepRenderResponse> {
  return requestJSON<SweepRenderResponse>("/sweep/render", {
    method: "POST",
    body: JSON.stringify({
      env,
      state,
      renderer,
      cells,
    }),
    signal,
  });
}

export async function fetchInteractiveStart(
  env: string,
  state: StatePayload,
  effects: EffectsStore,
  renderer: RendererStore,
  signal?: AbortSignal,
): Promise<InteractiveStartResponse> {
  return requestJSON<InteractiveStartResponse>("/interactive/start", {
    method: "POST",
    body: JSON.stringify({
      env,
      state,
      effects,
      renderer,
    }),
    signal,
  });
}

export async function fetchInteractiveRender(
  sessionId: string,
  effects: EffectsStore,
  renderer: RendererStore,
  signal?: AbortSignal,
): Promise<InteractiveRenderResponse> {
  return requestJSON<InteractiveRenderResponse>("/interactive/render", {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      effects,
      renderer,
    }),
    signal,
  });
}

export async function fetchInteractiveStep(
  sessionId: string,
  action: number,
  effects: EffectsStore,
  renderer: RendererStore,
  signal?: AbortSignal,
): Promise<InteractiveStepResponse> {
  return requestJSON<InteractiveStepResponse>("/interactive/step", {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      action,
      effects,
      renderer,
    }),
    signal,
  });
}

export async function fetchInteractiveEvent(
  sessionId: string,
  event: InteractiveEventPayload,
  effects: EffectsStore,
  renderer: RendererStore,
  signal?: AbortSignal,
): Promise<InteractiveEventResponse> {
  return requestJSON<InteractiveEventResponse>("/interactive/event", {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      event,
      effects,
      renderer,
    }),
    signal,
  });
}

export async function fetchInteractiveReset(
  sessionId: string,
  effects: EffectsStore,
  renderer: RendererStore,
  signal?: AbortSignal,
): Promise<InteractiveRenderResponse> {
  return requestJSON<InteractiveRenderResponse>("/interactive/reset", {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
      effects,
      renderer,
    }),
    signal,
  });
}

export async function fetchInteractiveStop(sessionId: string, signal?: AbortSignal): Promise<InteractiveStopResponse> {
  return requestJSON<InteractiveStopResponse>("/interactive/stop", {
    method: "POST",
    body: JSON.stringify({
      session_id: sessionId,
    }),
    signal,
  });
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

export function getErrorMessage(error: unknown): string {
  if (error instanceof Error) {
    return error.message;
  }
  return "Unknown error";
}
