import { fail, proxyJson, workerConfig } from "../_shared.js";
import type { VercelRequest, VercelResponse } from "../_types.js";

export const config = { maxDuration: 60 };

export default async function handler(request: VercelRequest, response: VercelResponse): Promise<void> {
  if (request.method !== "GET") {
    response.setHeader("Allow", "GET").status(405).json({ error: "Method not allowed" });
    return;
  }
  const jobId = Array.isArray(request.query.jobId) ? request.query.jobId[0] : request.query.jobId;
  if (!jobId || !/^[a-f0-9]{32}$/.test(jobId)) {
    response.status(400).json({ error: "Invalid job ID" });
    return;
  }
  try {
    const worker = workerConfig();
    await proxyJson(response, `${worker.baseUrl}/jobs/${jobId}`, { headers: worker.headers });
  } catch (error) {
    fail(response, error, 502);
  }
}
