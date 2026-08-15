import { fail, normalizeXUrl, proxyJson, workerConfig } from "./_shared.js";
import type { VercelRequest, VercelResponse } from "./_types.js";

export const config = { maxDuration: 60 };

export default async function handler(request: VercelRequest, response: VercelResponse): Promise<void> {
  if (request.method !== "POST") {
    response.setHeader("Allow", "POST").status(405).json({ error: "Method not allowed" });
    return;
  }
  try {
    const url = normalizeXUrl(request.body?.url);
    const worker = workerConfig();
    await proxyJson(response, `${worker.baseUrl}/transcribe`, {
      method: "POST",
      headers: worker.headers,
      body: JSON.stringify({ url }),
    });
  } catch (error) {
    fail(response, error, error instanceof Error && error.message.includes("URL") ? 400 : 502);
  }
}
