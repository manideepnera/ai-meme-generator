export interface MemeGenerationRequest {
    companyDescription: string;
}

export interface MemeGenerationResponse {
    imageUrl: string;
    caption: string;
    memeIdea: string;
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
