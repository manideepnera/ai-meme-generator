'use client';

import React, { useState } from 'react';
import styles from './page.module.css';
import { Header } from './components/Header';
import { CompanyInputForm } from './components/CompanyInputForm';
import { LoadingState } from './components/LoadingState';
import { MemeResult } from './components/MemeResult';
import { ErrorState } from './components/ErrorState';
import {
  MemeGenerationResponse,
  GenerationStep,
  LoadingState as LoadingStateType,
} from '@/types';
import { generateImageWithPuter } from '@/lib/puterMeme';

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

  const handleGenerate = async (description: string) => {
    setCurrentDescription(description);
    setResult(null);
    setError(null);

    setLoadingState({
      isLoading: true,
      currentStep: 'understanding',
      progress: 10,
    });

    try {
      // 1) LLaMA: funny meme concept (caption + image prompt that illustrates the joke)
      setLoadingState({
        isLoading: true,
        currentStep: 'understanding',
        progress: 20,
      });
      const conceptRes = await fetch('/api/generate-meme-concept', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ companyDescription: description }),
      });
      if (!conceptRes.ok) {
        const errData = await conceptRes.json().catch(() => ({}));
        throw new Error(errData.error || 'Failed to generate meme concept');
      }
      const concept = await conceptRes.json();
      const { caption, image_prompt: imagePrompt } = concept;
      if (!caption || !imagePrompt) {
        throw new Error('Invalid meme concept from server');
      }

      setLoadingState({
        isLoading: true,
        currentStep: 'generating-image',
        progress: 45,
      });
      // 2) Puter: generate image from LLaMA's scene; prepend product so image stays on-topic
      const imagePromptWithProduct = `Meme about this product: "${description.slice(0, 100)}". Scene to draw: ${imagePrompt}`;
      const imageBase64 = await generateImageWithPuter(imagePromptWithProduct);

      const puterResult = { imageBase64, caption };

      setLoadingState({
        isLoading: true,
        currentStep: 'adding-caption',
        progress: 85,
      });

      // Overlay caption on image via backend (no API key needed)
      const overlayRes = await fetch('/api/overlay-caption', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_base64: puterResult.imageBase64,
          caption: puterResult.caption,
        }),
      });

      if (!overlayRes.ok) {
        const errData = await overlayRes.json().catch(() => ({}));
        throw new Error(errData.error || 'Caption overlay failed');
      }

      const overlayData = await overlayRes.json();
      const imageUrl = overlayData.image_base64?.startsWith('data:')
        ? overlayData.image_base64
        : `data:image/png;base64,${overlayData.image_base64 || ''}`;

      const normalizedData: MemeGenerationResponse = {
        imageUrl,
        caption: puterResult.caption,
        memeIdea: '',
        textPosition: 'bottom',
      };

      setLoadingState({
        isLoading: false,
        currentStep: 'complete',
        progress: 100,
      });
      setResult(normalizedData);
    } catch (err) {
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
            Powered by • LLaMA, Stable Diffusion & BLIP
          </p>
        </footer>
      </div>
    </main>
  );
}
