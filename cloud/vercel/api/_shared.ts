import type { VercelRequest, VercelResponse } from "./_types.js";

const ALLOWED_HOSTS = new Set(["x.com", "www.x.com", "twitter.com", "www.twitter.com"]);
const ALLOWED_PATHS = ["/i/spaces/", "/i/broadcasts/"];

export function normalizeXUrl(value: unknown): string {
  if (typeof value !== "string") throw new Error("X Spaces URL is required");
  let parsed: URL;
  try {
    parsed = new URL(value.trim());
  } catch {
    throw new Error("Invalid X Spaces URL");
  }
  if (!(["http:", "https:"].includes(parsed.protocol) && ALLOWED_HOSTS.has(parsed.hostname))) {
    throw new Error("Invalid X Spaces URL");
  }
  if (!ALLOWED_PATHS.some((prefix) => parsed.pathname.startsWith(prefix))) {
    throw new Error("Only /i/spaces/ and /i/broadcasts/ URLs are supported");
  }
  return `https://x.com${parsed.pathname.replace(/\/$/, "")}`;
}

export function workerConfig(): { baseUrl: string; headers: Record<string, string> } {
  const baseUrl = (process.env.WORKER_API_URL || "").replace(/\/$/, "");
  if (!baseUrl) throw new Error("WORKER_API_URL is not configured");
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (process.env.WORKER_API_TOKEN) headers["X-Worker-Token"] = process.env.WORKER_API_TOKEN;
  return { baseUrl, headers };
}

export async function proxyJson(
  response: VercelResponse,
  url: string,
  options: RequestInit,
): Promise<void> {
  const upstream = await fetch(url, options);
  const body = await upstream.text();
  response.status(upstream.status).setHeader("Content-Type", "application/json").send(body);
}

export function fail(response: VercelResponse, error: unknown, status = 500): void {
  response.status(status).json({ error: error instanceof Error ? error.message : "Unexpected error" });
}

export type Handler = (request: VercelRequest, response: VercelResponse) => Promise<void>;
