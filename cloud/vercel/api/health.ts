import { fail, proxyJson, workerConfig } from "./_shared.js";
import type { VercelRequest, VercelResponse } from "./_types.js";

export default async function handler(request: VercelRequest, response: VercelResponse): Promise<void> {
  if (request.method !== "GET") {
    response.setHeader("Allow", "GET").status(405).json({ error: "Method not allowed" });
    return;
  }
  try {
    const worker = workerConfig();
    await proxyJson(response, `${worker.baseUrl}/health`, { headers: worker.headers });
  } catch (error) {
    fail(response, error, 502);
  }
}
