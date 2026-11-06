export interface MemeGenerationRequest {
    companyDescription: string;
    useLlama: boolean;
}

export interface BackendMemeResponse {
    image_url?: string;
    image_base64?: string;
    caption: string;
    text_position: 'top' | 'bottom';
    template_id?: string;
    template_name?: string;
    template_score?: number;
    template_reason?: string;
    ai_provider?: 'llama' | 'gemini' | 'local';
}

export interface MemeGenerationResponse {
    imageUrl: string;
    caption: string;
    memeIdea: string;
    textPosition: 'top' | 'bottom';
    templateName?: string;
    templateReason?: string;
    aiProvider?: 'llama' | 'gemini' | 'local';
}

export type GenerationStep = 'idle' | 'understanding' | 'generating-image' | 'adding-caption' | 'complete' | 'error';

export interface LoadingState {
    isLoading: boolean;
    currentStep: GenerationStep;
    progress: number;
}
