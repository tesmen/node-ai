import { Matrix } from '../types';

export interface ModelConfig {
    nCtx?: number;
    nHidden?: number;
    nEmbd?: number;
    source: string;
    id?: number;

    wte?: Matrix;
    wpe?: Matrix;
}