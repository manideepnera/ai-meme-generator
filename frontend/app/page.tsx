'use client';

import React, { useState } from 'react';
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
  // ============================================
  // ERROR STATE - Tracks API errors
  // ============================================
  const [error, setError] = useState<ErrorInfo | null>(null);

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

  const handleGenerate = async (description: string) => {
    setCurrentDescription(description);
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
            company_description: description  // Backend expects snake_case
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
        // Backend may send image_url OR image_base64. image_base64 can be
        // either raw base64 or already a data URI (e.g. from Stable Diffusion).
        let imageUrl: string;
        if (backendData.image_url) {
          imageUrl = backendData.image_url;
        } else if (backendData.image_base64) {
          imageUrl = backendData.image_base64.startsWith('data:')
            ? backendData.image_base64
            : `data:image/png;base64,${backendData.image_base64}`;
        } else {
          imageUrl = '';
        }

        const normalizedData: MemeGenerationResponse = {
          imageUrl,
          caption: backendData.caption,
          memeIdea: '', // Kept for UI compatibility
          textPosition: backendData.text_position,
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
      handleGenerate(currentDescription);
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
            Powered by AI • LLaMA, Stable Diffusion & BLIP
          </p>
        </footer>
      </div>
    </main>
  );
}
