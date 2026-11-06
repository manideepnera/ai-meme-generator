export interface MemeGenerationRequest {
    companyDescription: string;\n    useLlama: boolean;
}

// ============================================
// BACKEND API RESPONSE TYPES
// ============================================

/**
 * Response from FastAPI backend POST /api/v1/generate-meme
 * Note: Only ONE of meme_url or image_base64 will be present
 */
export interface BackendMemeResponse {
    image_url?: string;     // URL to the generated meme image (optional)
    image_base64?: string;  // Base64 encoded image data (optional)
    caption: string;        // Generated caption for the meme
    text_position: 'top' | 'bottom';\n    template_id?: string;\n    template_name?: string;\n    template_score?: number;\n    template_reason?: string;  // Position of caption text
}

/**
 * Frontend-normalized response for UI consumption
 */
export interface MemeGenerationResponse {
    imageUrl: string;       // Normalized image source (URL or data URI)
    caption: string;
    memeIdea: string;       // Kept for backwards compatibility (may be empty)
    textPosition: 'top' | 'bottom';
}

export type GenerationStep =
    | 'idle'
    | 'understanding'
    | 'generating-image'
    | 'adding-caption'
    | 'complete'
    | 'error';

export interface LoadingState {
    isLoading: boolean;
    currentStep: GenerationStep;
    progress: number;
}
