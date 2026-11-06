'use client';

import React, { useEffect, useState } from 'react';
import styles from './page.module.css';
import { Header } from './components/Header';
import { CompanyInputForm } from './components/CompanyInputForm';
import { LoadingState } from './components/LoadingState';
import { MemeResult } from './components/MemeResult';
import { ErrorState } from './components/ErrorState';
import {
  MemeGenerationResponse,
  BackendMemeResponse,
  LoadingState as LoadingStateType,
} from '@/types';

interface ErrorInfo {
  message: string;
  details?: string;
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

export default function Home() {
  const [loadingState, setLoadingState] = useState<LoadingStateType>({
    isLoading: false,
    currentStep: 'idle',
    progress: 0,
  });
  const [result, setResult] = useState<MemeGenerationResponse | null>(null);
  const [currentDescription, setCurrentDescription] = useState('');
  const [error, setError] = useState<ErrorInfo | null>(null);
  const [geminiConfigured, setGeminiConfigured] = useState<boolean | null>(null);

  useEffect(() => {
    let cancelled = false;
    const checkReadiness = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/api/v1/health/ready`);
        if (!response.ok) return;
        const data = await response.json();
        if (!cancelled) {
          setGeminiConfigured(Boolean(data.configuration?.gemini_api_configured));
        }
      } catch {
        if (!cancelled) setGeminiConfigured(null);
      }
    };
    checkReadiness();
    return () => { cancelled = true; };
  }, []);

  const handleGenerate = async (description: string) => {
    setCurrentDescription(description);
    setResult(null);
    setError(null);
    setLoadingState({
      isLoading: true,
      currentStep: 'understanding',
      progress: 15,
    });

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/generate-meme`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ company_description: description }),
      });

      const responseData = await response.json().catch(() => ({}));
      if (!response.ok) {
        const detail = responseData.detail;
        const message = typeof detail === 'string'
          ? detail
          : detail?.message || responseData.message || `Request failed with status ${response.status}`;
        throw new Error(message);
      }

      const backendData: BackendMemeResponse = responseData;
      if (!backendData.image_base64 && !backendData.image_url) {
        throw new Error('The backend completed without returning a meme image.');
      }

      const normalizedData: MemeGenerationResponse = {
        imageUrl: backendData.image_url || `data:image/png;base64,${backendData.image_base64}`,
        caption: backendData.caption,
        memeIdea: '',
        textPosition: backendData.text_position,
        templateName: backendData.template_name,
        templateReason: backendData.template_reason,
        aiProvider: backendData.ai_provider,
        jokeCandidateCount: backendData.joke_candidate_count,
        jokeSelectionReason: backendData.joke_selection_reason,
      };

      setResult(normalizedData);
      setLoadingState({ isLoading: false, currentStep: 'complete', progress: 100 });
    } catch (err) {
      console.error('Error generating meme:', err);
      setError({
        message: 'Failed to generate meme',
        details: err instanceof Error ? err.message : 'Unknown error occurred',
      });
      setLoadingState({ isLoading: false, currentStep: 'error', progress: 0 });
    }
  };

  const handleRetry = () => {
    if (currentDescription) void handleGenerate(currentDescription);
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
          onGenerate={(description) => void handleGenerate(description)}
          isLoading={loadingState.isLoading}
          geminiConfigured={geminiConfigured}
        />
        {loadingState.isLoading && (
          <LoadingState currentStep={loadingState.currentStep} progress={loadingState.progress} />
        )}
        {result && !loadingState.isLoading && (
          <MemeResult result={result} onRegenerate={() => void handleGenerate(currentDescription)} />
        )}
        {error && !loadingState.isLoading && (
          <ErrorState message={error.message} details={error.details} onRetry={handleRetry} />
        )}
        <footer className={styles.footer}>
          <p className={styles.footerText}>
            Existing meme templates • Google Gemini • 3 joke candidates • Hybrid quality checks
          </p>
        </footer>
      </div>
    </main>
  );
}
