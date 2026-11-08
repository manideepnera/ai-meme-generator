'use client';

import React, { useState } from 'react';
import styles from './CompanyInputForm.module.css';
import { MemeGenerationRequest } from '@/types';

interface CompanyInputFormProps {
    onGenerate: (brief: MemeGenerationRequest) => void;
    isLoading: boolean;
    hybridConfigured: boolean | null;
}

export const CompanyInputForm: React.FC<CompanyInputFormProps> = ({
    onGenerate,
    isLoading,
    hybridConfigured,
}) => {
    const [company, setCompany] = useState('');
    const [product, setProduct] = useState('');
    const [objective, setObjective] = useState('');
    const [audience, setAudience] = useState('');

    const valid = company.trim().length >= 2
        && product.trim().length >= 3
        && objective.trim().length >= 3
        && audience.trim().length >= 2;

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (valid && !isLoading) {
            onGenerate({
                company: company.trim(),
                product: product.trim(),
                advertising_objective: objective.trim(),
                audience: audience.trim(),
            });
        }
    };

    const loadExample = () => {
        setCompany('HydroTrack');
        setProduct('A smart reusable water bottle that tracks hydration and sends reminders');
        setObjective('Promote daily hydration and increase product awareness');
        setAudience('Busy students and software developers');
    };

    return (
        <div className={styles.formContainer}>
            <form onSubmit={handleSubmit} className={styles.form}>
                <div className={styles.inputWrapper}>
                    <label htmlFor="company" className={styles.label}>Company / Brand</label>
                    <input id="company" className={styles.textarea} type="text"
                        placeholder="e.g. HydroTrack" value={company}
                        onChange={(e) => setCompany(e.target.value)} disabled={isLoading} />
                </div>

                <div className={styles.inputWrapper}>
                    <label htmlFor="product" className={styles.label}>Product / Service</label>
                    <textarea id="product" className={styles.textarea} rows={3}
                        placeholder="What are you advertising? Describe its features and benefits."
                        value={product} onChange={(e) => setProduct(e.target.value)} disabled={isLoading} />
                </div>

                <div className={styles.inputWrapper}>
                    <label htmlFor="objective" className={styles.label}>Advertising objective</label>
                    <textarea id="objective" className={styles.textarea} rows={2}
                        placeholder="e.g. Build product awareness, promote a launch, or drive sign-ups"
                        value={objective} onChange={(e) => setObjective(e.target.value)} disabled={isLoading} />
                </div>

                <div className={styles.inputWrapper}>
                    <label htmlFor="audience" className={styles.label}>Target audience</label>
                    <input id="audience" className={styles.textarea} type="text"
                        placeholder="e.g. Busy students and software developers"
                        value={audience} onChange={(e) => setAudience(e.target.value)} disabled={isLoading} />
                </div>

                <p className={styles.helperText}>
                    LLaMA 3 expands this advertising brief and develops meme captions. SDXL creates a visual, BLIP-2 analyses it, and OpenCV renders the result.
                </p>

                <div className={styles.llamaControl}>
                    <div className={styles.llamaControlText}>
                        <span className={styles.llamaTitle}>AI pipeline</span>
                        <span className={styles.llamaDescription}>
                            LLaMA 3 · Template matching · SDXL · BLIP-2 · OpenCV
                            {hybridConfigured !== null && (
                                <span className={styles.providerStatus}>
                                    Hybrid pipeline: {hybridConfigured ? 'configured' : 'not configured'}
                                </span>
                            )}
                        </span>
                    </div>
                    <span className={styles.modeOn}>HYBRID</span>
                </div>

                <button type="submit" className={styles.generateButton}
                    disabled={!valid || isLoading}>
                    {isLoading ? (
                        <><span className={styles.spinner}></span>Generating...</>
                    ) : (
                        <>
                            <svg className={styles.buttonIcon} viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M13 10V3L4 14h7v7l9-11h-7z" fill="currentColor" />
                            </svg>
                            Generate Marketing Meme
                        </>
                    )}
                </button>

                <div className={styles.examplesSection}>
                    <p className={styles.examplesLabel}>Try an example:</p>
                    <div className={styles.exampleChips}>
                        <button type="button" className={styles.exampleChip} onClick={loadExample} disabled={isLoading}>
                            Smart hydration bottle campaign
                        </button>
                    </div>
                </div>
            </form>
        </div>
    );
};
