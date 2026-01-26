'use client';

import React from 'react';
import styles from './ErrorState.module.css';

/**
 * ============================================
 * ERROR STATE COMPONENT
 * Displays user-friendly error messages when meme generation fails
 * ============================================
 */
interface ErrorStateProps {
    message?: string;
    details?: string;
    onRetry?: () => void;
}

export const ErrorState: React.FC<ErrorStateProps> = ({
    message = 'Something went wrong',
    details,
    onRetry,
}) => {
    return (
        <div className={styles.errorContainer}>
            <div className={styles.errorCard}>
                {/* Error Icon */}
                <div className={styles.iconWrapper}>
                    <svg
                        className={styles.errorIcon}
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                    >
                        <path
                            d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                        />
                    </svg>
                </div>

                {/* Error Message */}
                <h3 className={styles.errorTitle}>{message}</h3>

                {details && (
                    <p className={styles.errorDetails}>{details}</p>
                )}

                {/* Retry Button */}
                {onRetry && (
                    <button className={styles.retryButton} onClick={onRetry}>
                        <svg
                            className={styles.icon}
                            viewBox="0 0 24 24"
                            fill="none"
                            xmlns="http://www.w3.org/2000/svg"
                        >
                            <path
                                d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"
                                stroke="currentColor"
                                strokeWidth="2"
                                strokeLinecap="round"
                                strokeLinejoin="round"
                            />
                        </svg>
                        Try Again
                    </button>
                )}
            </div>
        </div>
    );
};
