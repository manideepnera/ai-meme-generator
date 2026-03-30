import { NextRequest, NextResponse } from 'next/server';

const getBackendUrl = (): string => {
  const url = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://127.0.0.1:8000';
  return url.endsWith('/') ? url.slice(0, -1) : url;
};

/**
 * POST /api/overlay-caption
 * Proxies to backend to overlay caption on image (used after Puter.js generates image).
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { image_base64: imageBase64, caption } = body;

    if (!imageBase64 || !caption || typeof caption !== 'string' || caption.trim().length === 0) {
      return NextResponse.json(
        { error: 'image_base64 and caption are required' },
        { status: 400 }
      );
    }

    const backendUrl = getBackendUrl();
    const res = await fetch(`${backendUrl}/api/v1/overlay-caption`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_base64: imageBase64, caption: caption.trim() }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      return NextResponse.json(
        { error: err.detail?.message || 'Overlay failed' },
        { status: res.status }
      );
    }

    const data = await res.json();
    return NextResponse.json(data);
  } catch (e) {
    console.error('[API] overlay-caption error:', e);
    return NextResponse.json(
      { error: 'Overlay caption failed' },
      { status: 500 }
    );
  }
}
