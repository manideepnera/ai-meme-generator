'use client';

import React, { useEffect, useState } from 'react';
import styles from './page.module.css';
import { Header } from './components/Header';
import { CompanyInputForm } from './components/CompanyInputForm';
import { LoadingState } from './components/LoadingState';
import { MemeResult } from './components/MemeResult';
import { ErrorState } from './components/ErrorState';  // Added for error handling
import {
  MemeGenerationResponse,
  BackendMemeResponse,
  GenerationStep,
  LoadingState as LoadingStateType,
} from '@/types';

// ============================================
// ERROR STATE TYPE FOR API ERRORS
// ============================================
interface ErrorInfo {
  message: string;
  details?: string;
}

export default function Home() {
  const [loadingState, setLoadingState] = useState<LoadingStateType>({
    isLoading: false,
    currentStep: 'idle',
    progress: 0,
  });
  const [result, setResult] = useState<MemeGenerationResponse | null>(null);
  const [currentDescription, setCurrentDescription] = useState('');
  const [currentUseLlama, setCurrentUseLlama] = useState(true);
  // ============================================
  // ERROR STATE - Tracks API errors
  // ============================================
  const [error, setError] = useState<ErrorInfo | null>(null);
  const [providerReadiness, setProviderReadiness] = useState<{ llama: boolean; gemini: boolean } | null>(null);

  useEffect(() => {
    let cancelled = false;
    const checkReadiness = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/v1/health/ready');
        if (!response.ok) return;
        const data = await response.json();
        if (!cancelled) setProviderReadiness({ llama: Boolean(data.configuration?.llama_api_configured), gemini: Boolean(data.configuration?.gemini_api_configured) });
      } catch { if (!cancelled) setProviderReadiness(null); }
    };
    checkReadiness();
    return () => { cancelled = true; };
  }, []);

  const simulateProgress = (
    stepSequence: GenerationStep[],
    onComplete: () => void
  ) => {
    let currentStepIndex = 0;
    const totalSteps = stepSequence.length;
    const stepDuration = 1000; // 1 second per step

    const interval = setInterval(() => {
      if (currentStepIndex < totalSteps) {
        const step = stepSequence[currentStepIndex];
        const progress = ((currentStepIndex + 1) / totalSteps) * 100;

        setLoadingState({
          isLoading: true,
          currentStep: step,
          progress,
        });

        currentStepIndex++;
      } else {
        clearInterval(interval);
        onComplete();
      }
    }, stepDuration);
  };

  const handleGenerate = async (description: string, useLlama = true) => {
    setCurrentDescription(description);
    setCurrentUseLlama(useLlama);
    setResult(null);
    // ============================================
    // CLEAR PREVIOUS ERROR ON NEW REQUEST
    // ============================================
    setError(null);

    // Start loading state
    setLoadingState({
      isLoading: true,
      currentStep: 'understanding',
      progress: 0,
    });

    // Simulate step-by-step progress
    const steps: GenerationStep[] = [
      'understanding',
      'generating-image',
      'adding-caption',
    ];

    simulateProgress(steps, async () => {
      try {
        // ============================================
        // DIRECT BACKEND API CALL (Option A)
        // Bypassing Next.js API routes to call FastAPI directly
        // ============================================
        const backendEndpoint = 'http://localhost:8000/api/v1/generate-meme';

        const response = await fetch(backendEndpoint, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            company_description: description,
            use_llama: useLlama
          }),
        });

        // ============================================
        // ERROR HANDLING FOR API RESPONSE
        // ============================================
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          const errorMsg = errorData.detail?.message || errorData.error || `Request failed with status ${response.status}`;
          throw new Error(errorMsg);
        }

        const backendData: BackendMemeResponse = await response.json();

        // ============================================
        // NORMALIZE RESPONSE FOR UI
        // ============================================
        // Map backend fields to frontend interface
        const normalizedData: MemeGenerationResponse = {
          imageUrl: backendData.image_url || `data:image/png;base64,${backendData.image_base64}`,
          caption: backendData.caption,
          memeIdea: '', // Kept for UI compatibility
          textPosition: backendData.text_position,
          templateName: backendData.template_name,
          templateReason: backendData.template_reason,
          aiProvider: backendData.ai_provider,
        };

        // Set result and complete
        setLoadingState({
          isLoading: false,
          currentStep: 'complete',
          progress: 100,
        });

        setResult(normalizedData);
      } catch (err) {
        // ============================================
        // CATCH AND DISPLAY ERRORS TO USER
        // ============================================
        console.error('Error generating meme:', err);

        const errorMessage = err instanceof Error ? err.message : 'Unknown error occurred';

        setError({
          message: 'Failed to generate meme',
          details: errorMessage,
        });

        setLoadingState({
          isLoading: false,
          currentStep: 'error',
          progress: 0,
        });
      }
    });
  };

  const handleRegenerate = () => {
    if (currentDescription) {
      handleGenerate(currentDescription, currentUseLlama);
    }
  };

  // ============================================
  // RETRY HANDLER FOR ERROR STATE
  // ============================================
  const handleRetry = () => {
    if (currentDescription) {
      handleGenerate(currentDescription);
    }
  };

  return (
    <main className={styles.main}>
      <div className={styles.background}>
        <div className={styles.gradientOrb1}></div>
        <div className={styles.gradientOrb2}></div>
        <div className={styles.gradientOrb3}></div>
      </div>

      <div className={styles.content}>
        <Header />

        <CompanyInputForm
          onGenerate={handleGenerate}
          isLoading={loadingState.isLoading}
          providerReadiness={providerReadiness}
        />

        {loadingState.isLoading && (
          <LoadingState
            currentStep={loadingState.currentStep}
            progress={loadingState.progress}
          />
        )}

        {result && !loadingState.isLoading && (
          <MemeResult result={result} onRegenerate={handleRegenerate} />
        )}

        {/* ============================================
            ERROR STATE DISPLAY
            Shows user-friendly error with retry option
           ============================================ */}
        {error && !loadingState.isLoading && (
          <ErrorState
            message={error.message}
            details={error.details}
            onRetry={handleRetry}
          />
        )}

        {/* Footer */}
        <footer className={styles.footer}>
          <p className={styles.footerText}>
            Existing meme templates • LLaMA primary • Gemini automatic backup
          </p>
        </footer>
      </div>
    </main>
  );
}
