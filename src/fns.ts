import { Matrix, Vector } from './types';

export function layerNormRowwise(X: Matrix, eps = 1e-5, gamma: Vector = null, beta: Vector = null) {
    const m = X.length, n = X[0].length;
    const out = zeros(m, n);

    for (let i = 0; i < m; i++) {
        let mean = 0;
        for (let j = 0; j < n; j++) mean += X[i][j];
        mean /= n;
        let varSum = 0;
        for (let j = 0; j < n; j++) {
            const d = X[i][j] - mean;
            varSum += d * d;
        }
        const invStd = 1 / Math.sqrt(varSum / n + eps);
        for (let j = 0; j < n; j++) {
            let v = (X[i][j] - mean) * invStd;
            if (gamma) v *= gamma[j];
            if (beta) v += beta[j];
            out[i][j] = v;
        }
    }
    return out;
}

export function reduceM2Vector(M: Matrix): Vector {
    const vectorLength = M[0].length;
    const res: Vector = (new Array(vectorLength)).fill(0);

    for (let i = 0; i < M.length; i++) {
        for (let j = 0; j < vectorLength; j++) {
            res[j] += M[i][j];
        }
    }

    // this.log({ res })
    return res;
}

export function normalizeVector(v: Vector) {
    const norm = Math.sqrt(
      v.reduce((acc, num) => acc + num * num, 0)
    );
    // this.log({norm})

    return v.map(x => x / (norm || 1));
}


export function euclideanDistance(a: Vector, b: Vector) {
    if (a.length !== b.length) {
        throw new Error('Vectors must have the same length');
    }

    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        const diff = a[i] - b[i];
        sum += diff * diff;
    }

    return Math.sqrt(sum);
}


export function zeros(rows: number, cols: number): Matrix {
    const array = new Array(rows);

    for (let i = 0; i < rows; i++) {
        array[i] = new Array(cols);
    }

    return array;
}

export function initMat(M: Matrix): void {
    const magnitude: number = 0.2;
    for (let i = 0; i < M.length; i++) {

        for (let j = 0; j < M[i].length; j++) {
            M[i][j] = magnitude / 2 - Math.random() * magnitude;
        }
    }
}

export function dotProduct(a: Vector, b: Vector) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}


export function adjustEmbeddings(promptVec: Vector, targetVec: Vector, learningRate = 0.05) {
    if (promptVec.length !== targetVec.length) {
        throw new Error('Vectors must have the same length');
    }

    const newPrompt = [];
    const newTarget = [];

    for (let i = 0; i < promptVec.length; i++) {
        const delta = learningRate * (promptVec[i] - targetVec[i]);
        newPrompt.push(promptVec[i] - delta);  // move prompt toward target
        newTarget.push(targetVec[i] + delta);  // move target toward prompt
    }

    return { newPrompt, newTarget };
}
