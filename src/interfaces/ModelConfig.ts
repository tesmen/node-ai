export interface ModelConfig {
    nCtx?: number;
    nHidden?: number;
    nEmbd?: number;
    corpusFile: string;
    wpeFile?: string;
    wteFile?: string;
    id?: number;
}