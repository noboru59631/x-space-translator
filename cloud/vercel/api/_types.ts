export interface VercelRequest {
  method?: string;
  body?: Record<string, unknown>;
  query: Record<string, string | string[] | undefined>;
}

export interface VercelResponse {
  status(code: number): VercelResponse;
  setHeader(name: string, value: string): VercelResponse;
  json(value: unknown): void;
  send(value: string): void;
}
