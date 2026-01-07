// Data collector - reads Codex CLI storage and returns raw data

import { createReadStream } from "node:fs";
import { readdir, stat } from "node:fs/promises";
import { join } from "node:path";
import os from "node:os";
import { createInterface } from "node:readline";

const CODEX_DATA_PATH = join(os.homedir(), ".codex");
const CODEX_HISTORY_PATH = join(CODEX_DATA_PATH, "history.jsonl");
const CODEX_SESSIONS_PATH = join(CODEX_DATA_PATH, "sessions");

export type ModelUsageTotals = {
  inputTokens: number;
  cachedInputTokens: number;
  outputTokens: number;
  reasoningTokens: number;
  totalTokens: number;
};

export interface CodexUsageData {
  dailyActivity: Map<string, number>;
  totalMessages: number;
  totalSessions: number;
  projects: Set<string>;
  earliestSessionDate: Date | null;
  modelUsageTotals: Map<string, ModelUsageTotals>;
  totalInputTokens: number;
  totalCachedInputTokens: number;
  totalOutputTokens: number;
  totalReasoningTokens: number;
  totalTokens: number;
}

export async function checkCodexDataExists(): Promise<boolean> {
  try {
    const info = await stat(CODEX_SESSIONS_PATH);
    return info.isDirectory();
  } catch {
    return false;
  }
}

export async function listCodexSessionFiles(year: number): Promise<string[]> {
  const yearPath = join(CODEX_SESSIONS_PATH, String(year));
  const files: string[] = [];

  let monthDirs: Array<string> = [];
  try {
    const entries = await readdir(yearPath, { withFileTypes: true });
    monthDirs = entries.filter((e) => e.isDirectory()).map((e) => e.name);
  } catch {
    return files;
  }

  for (const month of monthDirs) {
    const monthPath = join(yearPath, month);
    let dayDirs: Array<string> = [];
    try {
      const entries = await readdir(monthPath, { withFileTypes: true });
      dayDirs = entries.filter((e) => e.isDirectory()).map((e) => e.name);
    } catch {
      continue;
    }

    for (const day of dayDirs) {
      const dayPath = join(monthPath, day);
      try {
        const entries = await readdir(dayPath, { withFileTypes: true });
        for (const entry of entries) {
          if (entry.isFile() && entry.name.endsWith(".jsonl")) {
            files.push(join(dayPath, entry.name));
          }
        }
      } catch {
        // Ignore unreadable day directories
      }
    }
  }

  return files;
}

export async function getCodexFirstPromptTimestamp(): Promise<number | null> {
  try {
    let minTs: number | null = null;

    // Stream rather than `readFile` to avoid loading large history files into memory.
    const rl = createInterface({
      input: createReadStream(CODEX_HISTORY_PATH),
      crlfDelay: Infinity,
    });

    for await (const line of rl) {
      const trimmed = line.trim();
      if (!trimmed) continue;

      // Avoid JSON.parse in the hot path; the history file is JSONL and we only need `ts`.
      const match = trimmed.match(/"ts"\s*:\s*(\d+)/);
      if (!match) continue;
      const ts = Number(match[1]);
      if (!Number.isFinite(ts) || ts <= 0) continue;

      if (minTs === null || ts < minTs) {
        minTs = ts;
      }
    }
    return minTs;
  } catch {
    return null;
  }
}

export async function collectCodexUsageData(year: number): Promise<CodexUsageData> {
  const files = await listCodexSessionFiles(year);
  const dailyActivity = new Map<string, number>();
  const projects = new Set<string>();
  let totalMessages = 0;
  let earliestSessionDate: Date | null = null;

  const modelUsageTotals = new Map<string, ModelUsageTotals>();
  let totalInputTokens = 0;
  let totalCachedInputTokens = 0;
  let totalOutputTokens = 0;
  let totalReasoningTokens = 0;
  let totalTokens = 0;

  const concurrency = Math.max(1, Math.min(os.cpus()?.length ?? 4, 8));
  const results = await asyncPool(concurrency, files, (filePath) => processSessionFile(filePath));

  for (const res of results) {
    totalMessages += res.totalMessages;
    if (res.earliestSessionDate && (!earliestSessionDate || res.earliestSessionDate < earliestSessionDate)) {
      earliestSessionDate = res.earliestSessionDate;
    }
    for (const project of res.projects) projects.add(project);
    mergeCountMap(dailyActivity, res.dailyActivity);

    totalInputTokens += res.totalInputTokens;
    totalCachedInputTokens += res.totalCachedInputTokens;
    totalOutputTokens += res.totalOutputTokens;
    totalReasoningTokens += res.totalReasoningTokens;
    totalTokens += res.totalTokens;

    for (const [model, usage] of res.modelUsageTotals.entries()) {
      const acc = getOrCreateModelUsage(modelUsageTotals, model);
      acc.inputTokens += usage.inputTokens;
      acc.cachedInputTokens += usage.cachedInputTokens;
      acc.outputTokens += usage.outputTokens;
      acc.reasoningTokens += usage.reasoningTokens;
      acc.totalTokens += usage.totalTokens;
    }
  }

  return {
    dailyActivity,
    totalMessages,
    totalSessions: files.length,
    projects,
    earliestSessionDate,
    modelUsageTotals,
    totalInputTokens,
    totalCachedInputTokens,
    totalOutputTokens,
    totalReasoningTokens,
    totalTokens,
  };
}

type RawUsage = {
  input_tokens: number;
  cached_input_tokens: number;
  output_tokens: number;
  reasoning_output_tokens: number;
  total_tokens: number;
};

const LEGACY_FALLBACK_MODEL = "gpt-5";

function ensureNumber(value: unknown): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function normalizeRawUsage(value: unknown): RawUsage | null {
  if (value == null || typeof value !== "object") return null;
  const record = value as Record<string, unknown>;
  const input = ensureNumber(record.input_tokens);
  const cached = ensureNumber(record.cached_input_tokens ?? record.cache_read_input_tokens);
  const output = ensureNumber(record.output_tokens);
  const reasoning = ensureNumber(record.reasoning_output_tokens);
  const total = ensureNumber(record.total_tokens);

  return {
    input_tokens: input,
    cached_input_tokens: cached,
    output_tokens: output,
    reasoning_output_tokens: reasoning,
    total_tokens: total > 0 ? total : input + output,
  };
}

function subtractRawUsage(current: RawUsage, previous: RawUsage | null): RawUsage {
  return {
    input_tokens: Math.max(current.input_tokens - (previous?.input_tokens ?? 0), 0),
    cached_input_tokens: Math.max(current.cached_input_tokens - (previous?.cached_input_tokens ?? 0), 0),
    output_tokens: Math.max(current.output_tokens - (previous?.output_tokens ?? 0), 0),
    reasoning_output_tokens: Math.max(current.reasoning_output_tokens - (previous?.reasoning_output_tokens ?? 0), 0),
    total_tokens: Math.max(current.total_tokens - (previous?.total_tokens ?? 0), 0),
  };
}

function convertToDelta(raw: RawUsage): CodexUsageEvent {
  const total = raw.total_tokens > 0 ? raw.total_tokens : raw.input_tokens + raw.output_tokens;
  const cached = Math.min(raw.cached_input_tokens, raw.input_tokens);
  return {
    inputTokens: raw.input_tokens,
    cachedInputTokens: cached,
    outputTokens: raw.output_tokens,
    reasoningOutputTokens: raw.reasoning_output_tokens,
    totalTokens: total,
  };
}

type CodexUsageEvent = {
  inputTokens: number;
  cachedInputTokens: number;
  outputTokens: number;
  reasoningOutputTokens: number;
  totalTokens: number;
};

function getOrCreateModelUsage(map: Map<string, ModelUsageTotals>, modelId: string): ModelUsageTotals {
  const existing = map.get(modelId);
  if (existing) return existing;
  const fresh = {
    inputTokens: 0,
    cachedInputTokens: 0,
    outputTokens: 0,
    reasoningTokens: 0,
    totalTokens: 0,
  };
  map.set(modelId, fresh);
  return fresh;
}

function extractModel(value: unknown): string | undefined {
  if (value == null || typeof value !== "object") return undefined;
  const payload = value as Record<string, unknown>;

  const info = payload.info;
  if (info && typeof info === "object") {
    const infoRecord = info as Record<string, unknown>;
    const direct = [infoRecord.model, infoRecord.model_name];
    for (const candidate of direct) {
      const model = asNonEmptyString(candidate);
      if (model) return model;
    }
    if (infoRecord.metadata && typeof infoRecord.metadata === "object") {
      const model = asNonEmptyString((infoRecord.metadata as Record<string, unknown>).model);
      if (model) return model;
    }
  }

  const fallbackModel = asNonEmptyString(payload.model);
  if (fallbackModel) return fallbackModel;

  if (payload.metadata && typeof payload.metadata === "object") {
    const model = asNonEmptyString((payload.metadata as Record<string, unknown>).model);
    if (model) return model;
  }

  return undefined;
}

function asNonEmptyString(value: unknown): string | undefined {
  if (typeof value !== "string") return undefined;
  const trimmed = value.trim();
  return trimmed === "" ? undefined : trimmed;
}

function formatDateKey(date: Date): string {
  const year = date.getFullYear();
  const month = String(date.getMonth() + 1).padStart(2, "0");
  const day = String(date.getDate()).padStart(2, "0");
  return `${year}-${month}-${day}`;
}

type FileUsageResult = {
  dailyActivity: Map<string, number>;
  totalMessages: number;
  projects: Set<string>;
  earliestSessionDate: Date | null;
  modelUsageTotals: Map<string, ModelUsageTotals>;
  totalInputTokens: number;
  totalCachedInputTokens: number;
  totalOutputTokens: number;
  totalReasoningTokens: number;
  totalTokens: number;
};

async function processSessionFile(filePath: string): Promise<FileUsageResult> {
  const dailyActivity = new Map<string, number>();
  const projects = new Set<string>();
  let totalMessages = 0;
  let earliestSessionDate: Date | null = null;

  const modelUsageTotals = new Map<string, ModelUsageTotals>();
  let totalInputTokens = 0;
  let totalCachedInputTokens = 0;
  let totalOutputTokens = 0;
  let totalReasoningTokens = 0;
  let totalTokens = 0;

  let previousTotals: RawUsage | null = null;
  let currentModel: string | undefined;
  let currentModelIsFallback = false;
  let legacyFallbackUsed = false;

  const rl = createInterface({
    input: createReadStream(filePath),
    crlfDelay: Infinity,
  });

  for await (const line of rl) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    const topType = getTopLevelType(trimmed);
    if (topType !== "session_meta" && topType !== "turn_context" && topType !== "event_msg") {
      continue;
    }

    let entry: any;
    try {
      entry = JSON.parse(trimmed);
    } catch {
      continue;
    }

    if (topType === "session_meta") {
      const sessionTimestamp = entry?.payload?.timestamp ?? entry?.timestamp;
      if (sessionTimestamp) {
        const sessionDate = new Date(sessionTimestamp);
        if (!earliestSessionDate || sessionDate < earliestSessionDate) {
          earliestSessionDate = sessionDate;
        }
      }
      const cwd = entry?.payload?.cwd;
      if (cwd) projects.add(cwd);
      continue;
    }

    if (topType === "turn_context") {
      const model = extractModel(entry?.payload);
      if (model) {
        currentModel = model;
        currentModelIsFallback = false;
      }
      continue;
    }

    // topType === "event_msg"
    const payload = entry?.payload;
    if (payload?.type === "user_message") {
      totalMessages += 1;
      const timestamp = entry?.timestamp;
      if (timestamp) {
        const dateKey = formatDateKey(new Date(timestamp));
        dailyActivity.set(dateKey, (dailyActivity.get(dateKey) || 0) + 1);
      }
      continue;
    }

    if (payload?.type !== "token_count") {
      continue;
    }

    const timestamp = entry?.timestamp;
    if (!timestamp) continue;

    const info = payload?.info;
    const lastUsage = normalizeRawUsage(info?.last_token_usage);
    const totalUsage = normalizeRawUsage(info?.total_token_usage);

    let raw = lastUsage;
    if (!raw && totalUsage) {
      raw = subtractRawUsage(totalUsage, previousTotals);
    }

    if (totalUsage) {
      previousTotals = totalUsage;
    }

    if (!raw) continue;

    const delta = convertToDelta(raw);
    if (
      delta.inputTokens === 0 &&
      delta.cachedInputTokens === 0 &&
      delta.outputTokens === 0 &&
      delta.reasoningOutputTokens === 0
    ) {
      continue;
    }

    // `info` is already part of the payload; avoid object spreading on the hot path.
    const extractedModel = extractModel(payload);
    let isFallback = false;
    if (extractedModel) {
      currentModel = extractedModel;
      currentModelIsFallback = false;
    }

    let model = extractedModel ?? currentModel;
    if (!model) {
      model = LEGACY_FALLBACK_MODEL;
      isFallback = true;
      legacyFallbackUsed = true;
      currentModel = model;
      currentModelIsFallback = true;
    } else if (!extractedModel && currentModelIsFallback) {
      isFallback = true;
    }

    if (isFallback) {
      // No-op for now; kept for parity with ccusage
    }

    const eventTotal = Math.max(delta.totalTokens, delta.inputTokens + delta.outputTokens);
    totalInputTokens += delta.inputTokens;
    totalCachedInputTokens += delta.cachedInputTokens;
    totalOutputTokens += delta.outputTokens;
    totalReasoningTokens += delta.reasoningOutputTokens;
    totalTokens += eventTotal;

    const usage = getOrCreateModelUsage(modelUsageTotals, model);
    usage.inputTokens += delta.inputTokens;
    usage.cachedInputTokens += delta.cachedInputTokens;
    usage.outputTokens += delta.outputTokens;
    usage.reasoningTokens += delta.reasoningOutputTokens;
    usage.totalTokens += eventTotal;
  }

  if (legacyFallbackUsed) {
    // ignore - best-effort
  }

  return {
    dailyActivity,
    totalMessages,
    projects,
    earliestSessionDate,
    modelUsageTotals,
    totalInputTokens,
    totalCachedInputTokens,
    totalOutputTokens,
    totalReasoningTokens,
    totalTokens,
  };
}

function mergeCountMap(into: Map<string, number>, from: Map<string, number>) {
  for (const [k, v] of from.entries()) {
    into.set(k, (into.get(k) || 0) + v);
  }
}

function getTopLevelType(line: string): string | null {
  const keyIdx = line.indexOf("\"type\"");
  if (keyIdx === -1) return null;
  const colonIdx = line.indexOf(":", keyIdx + 6);
  if (colonIdx === -1) return null;
  let i = colonIdx + 1;
  while (i < line.length && (line[i] === " " || line[i] === "\t")) i++;
  if (line[i] !== "\"") return null;
  const start = i + 1;
  const end = line.indexOf("\"", start);
  if (end === -1) return null;
  return line.slice(start, end);
}

async function asyncPool<T, R>(
  concurrency: number,
  items: readonly T[],
  worker: (item: T, index: number) => Promise<R>
): Promise<R[]> {
  const results = new Array<R>(items.length);
  let nextIndex = 0;

  const runners = new Array(Math.min(concurrency, items.length)).fill(null).map(async () => {
    while (true) {
      const index = nextIndex++;
      if (index >= items.length) return;
      results[index] = await worker(items[index], index);
    }
  });

  await Promise.all(runners);
  return results;
}
