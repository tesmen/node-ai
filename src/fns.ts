import { Matrix, Vector } from './types';

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

export function normalizeVectorL2(v: Vector) {
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

export function layerNormRowwise(matrix: Matrix, eps = 1e-5, gamma: Vector = null, beta: Vector = null): Matrix {
    const rows = matrix.length;
    const columns = matrix[0].length;
    const out = zeros(rows, columns);

    for (let ri = 0; ri < rows; ri++) {
        let mean = 0;

        for (let ci = 0; ci < columns; ci++) {
            mean += matrix[ri][ci];
        }

        mean /= columns;
        console.log({ mean });
        let varianceSum = 0;

        for (let ci = 0; ci < columns; ci++) {
            const delta = matrix[ri][ci] - mean;
            varianceSum += delta ** 2;
        }
        console.log({varianceSum})
        // Compute 1 / standard deviation
        const invStd = 1 / Math.sqrt(varianceSum / columns + eps);
        console.log({ invStd });
        for (let ci = 0; ci < columns; ci++) {
            let val = (matrix[ri][ci] - mean) * invStd;
            if (gamma) val *= gamma[ci];
            if (beta) val += beta[ci];
            out[ri][ci] = val;
        }
    }

    return out;
}

export function layerNorm1D(v: Vector, eps = 1e-5): Vector {
    const n = v.length;
    const mean = v.reduce((a, x) => a + x, 0) / n;
    const variance = v.reduce((a, x) => a + (x - mean) ** 2, 0) / n;
    const invStd = 1 / Math.sqrt(variance + eps);

    return v.map(x => (x - mean) * invStd);
}
