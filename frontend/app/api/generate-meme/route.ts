import { NextRequest, NextResponse } from 'next/server';
import { MemeGenerationRequest, MemeGenerationResponse, BackendMemeResponse } from '@/types';

// ============================================
// BACKEND INTEGRATION CONFIGURATION
// ============================================

/**
 * Get the backend URL from environment variables
 * Falls back to localhost if not configured
 */
const getBackendUrl = (): string => {
    const url = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://127.0.0.1:8000';
    // Remove trailing slash if present
    return url.endsWith('/') ? url.slice(0, -1) : url;
};

/**
 * POST /api/generate-meme
 * 
 * This endpoint proxies meme generation requests to the FastAPI backend.
 * It normalizes the backend response format for frontend consumption.
 */
export async function POST(request: NextRequest) {
    try {
        const body: MemeGenerationRequest = await request.json();
        const { companyDescription } = body;

        // ============================================
        // INPUT VALIDATION
        // ============================================
        if (!companyDescription || companyDescription.trim().length < 10) {
            return NextResponse.json(
                { error: 'Company description must be at least 10 characters long' },
                { status: 400 }
            );
        }

        // ============================================
        // BACKEND API INTEGRATION
        // ============================================

        const backendUrl = getBackendUrl();
        const apiEndpoint = `${backendUrl}/api/v1/generate-meme`;

        console.log(`[API] Calling backend at: ${apiEndpoint}`);

        // Send POST request to FastAPI backend
        const backendResponse = await fetch(apiEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                company_description: companyDescription,
            }),
        });

        // ============================================
        // ERROR HANDLING FOR BACKEND RESPONSE
        // ============================================

        if (!backendResponse.ok) {
            let errorDetails: any = {};
            try {
                errorDetails = await backendResponse.json();
            } catch (e) {
                errorDetails = { raw: await backendResponse.text().catch(() => 'No details available') };
            }

            console.error(`[API] Backend error (${backendResponse.status}):`, errorDetails);

            return NextResponse.json(
                {
                    error: `Backend error: ${backendResponse.status}`,
                    details: errorDetails.detail || errorDetails.message || JSON.stringify(errorDetails)
                },
                { status: backendResponse.status }
            );
        }

        // Parse backend response
        const backendData: BackendMemeResponse = await backendResponse.json();

        // ============================================
        // VALIDATE BACKEND RESPONSE
        // ============================================

        // Ensure at least one image source is present
        if (!backendData.image_url && !backendData.image_base64) {
            console.error('[API] Backend response missing image data');
            return NextResponse.json(
                { error: 'Backend response missing image data' },
                { status: 500 }
            );
        }

        // ============================================
        // NORMALIZE RESPONSE FOR FRONTEND
        // ============================================

        // Convert image source to a normalized URL
        // - If image_url exists, use it directly
        // - If image_base64 exists, convert to data URI
        let imageUrl: string;
        if (backendData.image_url) {
            imageUrl = backendData.image_url;
        } else {
            // Convert base64 to data URI for img src
            imageUrl = `data:image/png;base64,${backendData.image_base64}`;
        }

        // Build the normalized response for the frontend
        const response: MemeGenerationResponse = {
            imageUrl,
            caption: backendData.caption,
            memeIdea: '',  // Backend doesn't provide this, keeping for UI compatibility
            textPosition: backendData.text_position,
        };

        console.log('[API] Successfully generated meme');
        return NextResponse.json(response);

    } catch (error) {
        // ============================================
        // CATCH-ALL ERROR HANDLING
        // ============================================

        console.error('[API] Error generating meme:', error);

        // Check if it's a network error (backend not running)
        if (error instanceof TypeError && error.message.includes('fetch')) {
            return NextResponse.json(
                {
                    error: 'Cannot connect to backend server',
                    details: 'Make sure the FastAPI backend is running at the configured URL'
                },
                { status: 503 }
            );
        }

        return NextResponse.json(
            { error: 'Failed to generate meme' },
            { status: 500 }
        );
    }
}
