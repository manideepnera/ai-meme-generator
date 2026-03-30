import { NextRequest, NextResponse } from 'next/server';

const getBackendUrl = (): string => {
  const url = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://127.0.0.1:8000';
  return url.endsWith('/') ? url.slice(0, -1) : url;
};

/**
 * POST /api/generate-meme-concept
 * Gets a funny meme concept (caption + image_prompt) from LLaMA.
 * Image prompt describes a scene that illustrates the joke so the meme makes people laugh.
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { companyDescription } = body;

    if (!companyDescription || typeof companyDescription !== 'string' || !companyDescription.trim()) {
      return NextResponse.json(
        { error: 'companyDescription is required' },
        { status: 400 }
      );
    }

    const backendUrl = getBackendUrl();
    const res = await fetch(`${backendUrl}/api/v1/generate-meme-concept`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ company_description: companyDescription.trim() }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      return NextResponse.json(
        { error: err.detail?.message || 'Failed to generate meme concept' },
        { status: res.status }
      );
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (e) {
    console.error('[API] generate-meme-concept error:', e);
    return NextResponse.json(
      { error: 'Failed to generate meme concept' },
      { status: 500 }
    );
  }
}
