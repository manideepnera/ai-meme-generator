'use client';

import React, { useState } from 'react';
// NOTE: Using native <img> instead of Next.js Image to support base64 data URIs
import styles from './MemeResult.module.css';
import { MemeGenerationResponse } from '@/types';

interface MemeResultProps {
    result: MemeGenerationResponse;
    onRegenerate: () => void;
}

export const MemeResult: React.FC<MemeResultProps> = ({
    result,
    onRegenerate,
}) => {
    const [copiedCaption, setCopiedCaption] = useState(false);
    const [isDownloading, setIsDownloading] = useState(false);
    const [isInsightOpen, setIsInsightOpen] = useState(false);

    const handleCopyCaption = async () => {
        try {
            await navigator.clipboard.writeText(result.caption);
            setCopiedCaption(true);
            setTimeout(() => setCopiedCaption(false), 2000);
        } catch (error) {
            console.error('Failed to copy caption:', error);
        }
    };

    const handleDownload = async () => {
        setIsDownloading(true);
        try {
            const imageUrl = result.imageUrl;
            const fileName = `meme-${Date.now()}.png`;

            if (imageUrl.startsWith('data:')) {
                // Handle base64 image download
                const link = document.createElement('a');
                link.href = imageUrl;
                link.download = fileName;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);
            } else {
                // Handle remote URL image download
                // Fetch as blob to avoid CORS issues and force download
                const response = await fetch(imageUrl);
                const blob = await response.blob();
                const blobUrl = URL.createObjectURL(blob);

                const link = document.createElement('a');
                link.href = blobUrl;
                link.download = fileName;
                document.body.appendChild(link);
                link.click();
                document.body.removeChild(link);

                // Cleanup
                URL.revokeObjectURL(blobUrl);
            }
        } catch (error) {
            console.error('Failed to download meme:', error);
            alert("Failed to download the image. Please try right-clicking the image and selecting 'Save Image As'.");
        } finally {
            setIsDownloading(false);
        }
    };

    return (
        <div className={styles.resultContainer}>
            <div className={styles.resultCard}>
                {/* Success Badge */}
                <div className={styles.successBadge}>
                    <svg
                        className={styles.successIcon}
                        viewBox="0 0 24 24"
                        fill="none"
                        xmlns="http://www.w3.org/2000/svg"
                    >
                        <path
                            d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z"
                            stroke="currentColor"
                            strokeWidth="2"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                        />
                    </svg>
                    <span>Meme Generated Successfully!</span>
                </div>

                {/* ============================================
                    MEME IMAGE DISPLAY
                    Supports both URL and base64 data URIs
                   ============================================ */}
                <div className={styles.imageWrapper}>
                    <div className={styles.imageContainer}>
                        {/* Using native img to support base64 data URIs from backend */}
                        <img
                            src={result.imageUrl}
                            alt="Generated meme"
                            className={styles.image}
                            style={{ width: '100%', height: 'auto', maxWidth: '600px' }}
                        />
                    </div>
                </div>

                {/* Caption */}
                <div className={styles.captionContainer}>
                    <div className={styles.captionHeader}>
                        <span className={styles.captionLabel}>Generated Caption</span>
                        <button
                            className={styles.copyButton}
                            onClick={handleCopyCaption}
                            title="Copy caption"
                        >
                            {copiedCaption ? (
                                <>
                                    <svg
                                        className={styles.icon}
                                        viewBox="0 0 24 24"
                                        fill="none"
                                        xmlns="http://www.w3.org/2000/svg"
                                    >
                                        <path
                                            d="M5 13l4 4L19 7"
                                            stroke="currentColor"
                                            strokeWidth="2"
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                        />
                                    </svg>
                                    Copied!
                                </>
                            ) : (
                                <>
                                    <svg
                                        className={styles.icon}
                                        viewBox="0 0 24 24"
                                        fill="none"
                                        xmlns="http://www.w3.org/2000/svg"
                                    >
                                        <path
                                            d="M8 16H6a2 2 0 01-2-2V6a2 2 0 012-2h8a2 2 0 012 2v2m-6 12h8a2 2 0 002-2v-8a2 2 0 00-2-2h-8a2 2 0 00-2 2v8a2 2 0 002 2z"
                                            stroke="currentColor"
                                            strokeWidth="2"
                                            strokeLinecap="round"
                                            strokeLinejoin="round"
                                        />
                                    </svg>
                                    Copy
                                </>
                            )}
                        </button>
                    </div>
                    <p className={styles.caption}>{result.caption}</p>
                </div>

                {/* ============================================
                    MEME INSIGHT SECTION
                    Only shown if memeIdea is available from backend
                   ============================================ */}
                {result.memeIdea && result.memeIdea.trim() !== '' && (
                    <div className={styles.insightContainer}>
                        <div
                            className={styles.insightHeader}
                            onClick={() => setIsInsightOpen(!isInsightOpen)}
                        >
                            <div className={styles.insightTitle}>
                                <svg className={styles.icon} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M12 18V20M12 4V6M20 12H18M6 12H4M16.95 7.05L15.54 8.46M16.95 16.95L15.54 15.54M7.05 7.05L8.46 8.46M7.05 16.95L8.46 15.54" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
                                </svg>
                                <span>Behind the Meme (AI Insight)</span>
                            </div>
                            <div
                                className={styles.insightToggle}
                                style={{ transform: isInsightOpen ? 'rotate(180deg)' : 'rotate(0deg)' }}
                            >
                                <svg className={styles.icon} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                    <path d="M6 9L12 15L18 9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                                </svg>
                            </div>
                        </div>

                        {isInsightOpen && (
                            <div className={styles.insightContent}>
                                {result.memeIdea}
                            </div>
                        )}
                    </div>
                )}

                {/* Action Buttons */}
                <div className={styles.actionButtons}>
                    <button
                        className={`${styles.actionButton} ${styles.downloadButton}`}
                        onClick={handleDownload}
                        disabled={isDownloading}
                    >
                        {isDownloading ? (
                            <>
                                <span className={styles.spinner}></span>
                                Downloading...
                            </>
                        ) : (
                            <>
                                <svg
                                    className={styles.icon}
                                    viewBox="0 0 24 24"
                                    fill="none"
                                    xmlns="http://www.w3.org/2000/svg"
                                >
                                    <path
                                        d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4"
                                        stroke="currentColor"
                                        strokeWidth="2"
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                    />
                                </svg>
                                Download Meme
                            </>
                        )}
                    </button>

                    <button
                        className={`${styles.actionButton} ${styles.regenerateButton}`}
                        onClick={onRegenerate}
                    >
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
                        Regenerate
                    </button>
                </div>
            </div>
        </div>
    );
};
