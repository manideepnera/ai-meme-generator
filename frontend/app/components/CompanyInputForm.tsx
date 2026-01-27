'use client';

import React, { useState } from 'react';
import styles from './CompanyInputForm.module.css';

interface CompanyInputFormProps {
    onGenerate: (description: string) => void;
    isLoading: boolean;
}

export const CompanyInputForm: React.FC<CompanyInputFormProps> = ({
    onGenerate,
    isLoading,
}) => {
    const [description, setDescription] = useState('');
    const [isFocused, setIsFocused] = useState(false);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (description.trim() && !isLoading) {
            onGenerate(description.trim());
        }
    };

    const exampleIdeas = [
        "A new energy drink for students who study all night",
        "Eco-friendly reusable water bottles for outdoor enthusiasts",
        "AI-powered productivity app for remote teams",
        "Artisanal coffee subscription service for coffee lovers",
    ];

    const handleExampleClick = (example: string) => {
        setDescription(example);
    };

    return (
        <div className={styles.formContainer}>
            <form onSubmit={handleSubmit} className={styles.form}>
                <div className={styles.inputWrapper}>
                    <label htmlFor="company-description" className={styles.label}>
                        Describe your company or product
                    </label>
                    <p className={styles.helperText}>
                        AI will create a fun, relatable meme for public-facing advertisements.
                    </p>
                    <div className={`${styles.textareaContainer} ${isFocused ? styles.focused : ''}`}>
                        <textarea
                            id="company-description"
                            className={styles.textarea}
                            placeholder="Example: A cold drink that helps people survive summer heat"
                            value={description}
                            onChange={(e) => setDescription(e.target.value)}
                            onFocus={() => setIsFocused(true)}
                            onBlur={() => setIsFocused(false)}
                            rows={5}
                            disabled={isLoading}
                        />
                        <div className={`${styles.characterCount} ${description.trim().length > 0 && description.trim().length < 10 ? styles.countError : ''}`}>
                            {description.length} characters (min 10)
                        </div>
                    </div>
                </div>

                <button
                    type="submit"
                    className={styles.generateButton}
                    disabled={description.trim().length < 10 || isLoading}
                >
                    {isLoading ? (
                        <>
                            <span className={styles.spinner}></span>
                            Generating...
                        </>
                    ) : (
                        <>
                            <svg
                                className={styles.buttonIcon}
                                viewBox="0 0 24 24"
                                fill="none"
                                xmlns="http://www.w3.org/2000/svg"
                            >
                                <path
                                    d="M13 10V3L4 14h7v7l9-11h-7z"
                                    fill="currentColor"
                                />
                            </svg>
                            Generate Marketing Meme
                        </>
                    )}
                </button>

                <div className={styles.examplesSection}>
                    <p className={styles.examplesLabel}>Try an example:</p>
                    <div className={styles.exampleChips}>
                        {exampleIdeas.map((example, index) => (
                            <button
                                key={index}
                                type="button"
                                className={styles.exampleChip}
                                onClick={() => handleExampleClick(example)}
                                disabled={isLoading}
                            >
                                {example}
                            </button>
                        ))}
                    </div>
                </div>
            </form>
        </div>
    );
};
