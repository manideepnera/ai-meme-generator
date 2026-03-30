/**
 * Meme generation using Puter.js (free OpenAI API, no API key).
 * @see https://developer.puter.com/tutorials/free-unlimited-openai-api/
 */

import type { PuterLoaded } from '@/types/puter';
import { ENHANCE_PROMPT_INSTRUCTION, CAPTION_INSTRUCTION } from './puterPrompts';

const PUTER_CHAT_MODEL = 'gpt-5-nano';
const PUTER_IMAGE_MODEL = 'dall-e-3';

function getPuter(): PuterLoaded {
  if (typeof window === 'undefined' || !window.puter?.ai) {
    throw new Error('Puter.js is not loaded. Ensure the script https://js.puter.com/v2/ is included.');
  }
  return window.puter as PuterLoaded;
}

/** Convert HTMLImageElement to base64 data URL (waits for load if needed). */
function imageElementToBase64(img: HTMLImageElement): Promise<string> {
  return new Promise((resolve, reject) => {
    if (img.complete && img.naturalWidth > 0) {
      try {
        resolve(canvasToDataURL(img));
      } catch (e) {
        reject(e);
      }
      return;
    }
    img.onload = () => {
      try {
        resolve(canvasToDataURL(img));
      } catch (e) {
        reject(e);
      }
    };
    img.onerror = () => reject(new Error('Failed to load generated image'));
  });
}

function canvasToDataURL(img: HTMLImageElement): string {
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('Canvas 2d not available');
  ctx.drawImage(img, 0, 0);
  return canvas.toDataURL('image/png');
}

/** Extract plain text from Puter chat response (handles string or various object shapes). */
function extractChatText(response: unknown): string {
  if (typeof response === 'string') return response.trim();
  if (response && typeof response === 'object') {
    const o = response as Record<string, unknown>;
    if (typeof o.content === 'string') return o.content.trim();
    if (typeof o.text === 'string') return o.text.trim();
    if (typeof o.result === 'string') return o.result.trim();
    if (typeof o.output === 'string') return o.output.trim();
    if (o.message && typeof o.message === 'object') {
      const m = o.message as Record<string, unknown>;
      if (typeof m.content === 'string') return m.content.trim();
      if (typeof m.text === 'string') return m.text.trim();
    }
    // OpenAI-style: choices[0].message.content
    const choices = o.choices as Array<{ message?: { content?: string } }> | undefined;
    if (Array.isArray(choices) && choices[0]?.message?.content) {
      return String(choices[0].message.content).trim();
    }
  }
  return '';
}

/**
 * Enhance user description into a two-sentence image prompt (brand-aware, photorealistic).
 * Uses only the "Describe your company or product" field. If Puter returns empty, we use that description as the prompt.
 */
export async function enhancePromptWithPuter(companyDescription: string): Promise<string> {
  const puter = getPuter();
  const message = `${ENHANCE_PROMPT_INSTRUCTION}\n\nUser input: ${companyDescription}\n\nOutput only the image prompt:`;
  const response = await puter.ai.chat(message, {
    model: PUTER_CHAT_MODEL,
    max_tokens: 300,
  });
  const text = extractChatText(response).trim();
  // If Puter returns nothing, use the user's description as the image prompt (from the field only)
  if (!text) return companyDescription.trim();
  return text;
}

/**
 * Generate meme image from prompt using Puter (DALL-E 3 or GPT Image).
 * Strong photorealistic constraints so the image looks like a real photograph.
 */
export async function generateImageWithPuter(prompt: string): Promise<string> {
  const puter = getPuter();
  const photorealisticSuffix =
    ' Photorealistic. Actual photograph of real people, 35mm film, natural skin texture with pores, real hair, documentary or candid style, natural lighting. No 3D, no CGI, no illustration, no cartoon, no anime, no smooth or glossy skin—only real-world photography.';
  const fullPrompt = `${prompt.trim()}. ${photorealisticSuffix}`.trim();
  const imageElement = await puter.ai.txt2img(fullPrompt, { model: PUTER_IMAGE_MODEL });
  return imageElementToBase64(imageElement);
}

/**
 * Generate a short meme caption from the image concept. Always returns an actual caption, not a placeholder.
 */
export async function generateCaptionWithPuter(
  enhancedPrompt: string,
  companyDescription?: string
): Promise<string> {
  const puter = getPuter();
  const context = companyDescription
    ? `Product/idea: ${companyDescription}\nImage concept: ${enhancedPrompt}`
    : `Image concept: ${enhancedPrompt}`;
  const message = `${CAPTION_INSTRUCTION}\n\n${context}\n\nWrite one meme caption now:`;
  const response = await puter.ai.chat(message, {
    model: PUTER_CHAT_MODEL,
    max_tokens: 150,
  });
  const text = extractChatText(response).trim();
  // Fallback only if API truly returned nothing: use a short line derived from the concept
  if (!text) {
    const fallback = enhancedPrompt.split(/[.!?]/)[0]?.trim().slice(0, 60) || companyDescription?.slice(0, 60) || 'When the product hits different';
    return fallback + (fallback.length >= 55 ? '…' : '');
  }
  return text.slice(0, 150);
}

export interface PuterMemeResult {
  imageBase64: string;
  caption: string;
}

/**
 * Full pipeline: enhance prompt -> generate image -> generate caption.
 * Returns base64 image and caption. Call backend overlay-caption to add text on image.
 */
export async function generateMemeWithPuter(companyDescription: string): Promise<PuterMemeResult> {
  const enhanced = await enhancePromptWithPuter(companyDescription);
  const imageBase64 = await generateImageWithPuter(enhanced);
  const caption = await generateCaptionWithPuter(enhanced, companyDescription);
  return { imageBase64, caption };
}
