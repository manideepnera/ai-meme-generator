/**
 * Type declarations for Puter.js (https://js.puter.com/v2/)
 * Used for free OpenAI API access via Puter (no API key required).
 * @see https://developer.puter.com/tutorials/free-unlimited-openai-api/
 */

export interface PuterAIOptions {
  model?: string;
  stream?: boolean;
  temperature?: number;
  max_tokens?: number;
  driver?: string;
  tools?: unknown[];
}

export interface PuterAI {
  chat(
    message: string,
    options?: PuterAIOptions
  ): Promise<string>;
  chat(
    message: string,
    imageUrl: string,
    options?: PuterAIOptions
  ): Promise<string>;
  txt2img(
    prompt: string,
    options?: { model?: string }
  ): Promise<HTMLImageElement>;
}

/** Puter when loaded (non-optional). Use getPuter() which throws if not loaded. */
export type PuterLoaded = { ai: PuterAI };

declare global {
  interface Window {
    puter?: PuterLoaded;
  }
}

export {};
