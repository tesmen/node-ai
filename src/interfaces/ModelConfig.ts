import { Matrix } from '../types';

export interface ModelConfig {
    nCtx?: number;
    nHidden?: number;
    nEmbd?: number;
    corpusFile: string;
    id?: number;

    wte?: Matrix;
    wpe?: Matrix;
}