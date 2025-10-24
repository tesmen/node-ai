export interface ModelConfig {
    nCtx?: number;
    nHidden?: number;
    nEmbd?: number;
    vocabSize?: number | null;
    corpusFile: string;
    wpeFile?: string;
    wteFile?: string;
}