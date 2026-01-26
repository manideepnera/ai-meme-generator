'use client';

import React, { useEffect, useState } from 'react';
import styles from './LoadingState.module.css';
import { GenerationStep } from '@/types';

interface LoadingStateProps {
    currentStep: GenerationStep;
    progress: number;
}

const stepMessages: Record<GenerationStep, string> = {
    idle: '',
    understanding: 'Creating meme concept...',
    'generating-image': 'Generating image...',
    'adding-caption': 'Finalizing meme...',
    complete: '✅ Complete!',
    error: '❌ Something went wrong',
};

export const LoadingState: React.FC<LoadingStateProps> = ({
    currentStep,
    progress,
}) => {
    const [dots, setDots] = useState('');

    useEffect(() => {
        const interval = setInterval(() => {
            setDots((prev) => (prev.length >= 3 ? '' : prev + '.'));
        }, 500);

        return () => clearInterval(interval);
    }, []);

    return (
        <div className={styles.loadingContainer}>
            <div className={styles.loadingCard}>
                {/* Animated Icon */}
                <div className={styles.iconContainer}>
                    <svg
                        className={styles.icon}
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                    >
                        <circle
                            className={styles.iconCircle}
                            cx="12"
                            cy="12"
                            r="10"
                            stroke="url(#gradient)"
                            strokeWidth="2"
                        />
                        <path
                            className={styles.iconPath}
                            d="M12 6v6l4 2"
                            stroke="url(#gradient)"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                        />
                        <defs>
                            <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                                <stop offset="0%" stopColor="#667eea" />
                                <stop offset="100%" stopColor="#764ba2" />
                            </linearGradient>
                        </defs>
                    </svg>
                </div>

                {/* Status Message */}
                <p className={styles.message}>
                    {stepMessages[currentStep]}
                    <span className={styles.dots}>{dots}</span>
                </p>

                {/* Progress Bar */}
                <div className={styles.progressBarContainer}>
                    <div
                        className={styles.progressBar}
                        style={{ width: `${progress}%` }}
                    >
                        <div className={styles.progressGlow}></div>
                    </div>
                </div>

                {/* Progress Percentage */}
                <p className={styles.progressText}>{Math.round(progress)}%</p>

                {/* Step Indicators */}
                <div className={styles.stepIndicators}>
                    <StepIndicator
                        label="Concept"
                        isActive={currentStep === 'understanding'}
                        isComplete={
                            ['generating-image', 'adding-caption', 'complete'].includes(currentStep)
                        }
                    />
                    <StepIndicator
                        label="Image"
                        isActive={currentStep === 'generating-image'}
                        isComplete={['adding-caption', 'complete'].includes(currentStep)}
                    />
                    <StepIndicator
                        label="Finalizing"
                        isActive={currentStep === 'adding-caption'}
                        isComplete={currentStep === 'complete'}
                    />
                </div>
            </div>
        </div>
    );
};

interface StepIndicatorProps {
    label: string;
    isActive: boolean;
    isComplete: boolean;
}

const StepIndicator: React.FC<StepIndicatorProps> = ({
    label,
    isActive,
    isComplete,
}) => {
    return (
        <div
            className={`${styles.stepIndicator} ${isActive ? styles.active : ''
                } ${isComplete ? styles.complete : ''}`}
        >
            <div className={styles.stepDot}>
                {isComplete && (
                    <svg
                        className={styles.checkmark}
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                    >
                        <path
                            d="M5 13l4 4L19 7"
                            stroke="white"
                            strokeWidth="3"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                        />
                    </svg>
                )}
            </div>
            <span className={styles.stepLabel}>{label}</span>
        </div>
    );
};
