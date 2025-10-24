import { ModelConfig } from '../interfaces/ModelConfig';
import { FileServiceAdapter } from './file-service.adapter';
import { CharTokenizer } from './char.tokenizer';

type Vector = number[] | Float32Array;
type Matrix = Vector[];

export class SimpleBookProcessor {
    wte: Matrix = [];
    wpe: Matrix = [];
    private fileService: FileServiceAdapter;
    private tokenizer: CharTokenizer;
    private cfg: ModelConfig;

    constructor(cfg: ModelConfig) {
        this.cfg = cfg;
        this.fileService = new FileServiceAdapter();
        this.tokenizer = new CharTokenizer(this.getCorpus(cfg.corpusFile));
        console.log('vocab size', this.tokenizer.vocabSize);

        // Embeddings
        this.wte = this.zeros(this.tokenizer.vocabSize, cfg.nEmbd); // token embeddings
        this.wpe = this.zeros(cfg.nCtx, cfg.nEmbd);      // positional embeddings

        [this.wte, this.wpe,
            // this.Wq, this.Wk, this.Wv, this.Wo, this.W1, this.W2, this.Wout
        ].forEach(this.initMat);
    }

    generate(prompt: string, maxNewTokens: number = 30) {
        const ids = this.tokenizer.encode(prompt);

        console.log(ids);

        for (let step = 0; step < maxNewTokens; step++) {
            const logits = this.forward(ids);
        }

        return '';
    }

    forward(inputIds: number[]) {
        let promptMatrix = this.zeros(inputIds.length, this.cfg.nEmbd);

        for (let inputsArrIndex = 0; inputsArrIndex < inputIds.length; inputsArrIndex++) {
            const token = this.wte[inputIds[inputsArrIndex]];
            const position = this.wpe[inputsArrIndex];

            for (let j = 0; j < this.cfg.nEmbd; j++) {
                promptMatrix[inputsArrIndex][j] = token[j] + position[j];
            }
        }

        const normalizedX = this.layerNormRowwise(promptMatrix, 1e-5);
        const resultingVector = this.reduceM2Vector(normalizedX);
        const normalizedResVector = this.normalizeVector(resultingVector)
        const candidates = this.findTopKCandidates(normalizedResVector, this.wte);
        console.log(this.tokenizer.decode(candidates))
    }

    reduceM2Vector(M: Matrix): Vector {
        const vectorLength = M[0].length;
        const res = new Float32Array(vectorLength);

        for (let i = 0; i < M.length; i++) {
            for (let j = 0; j < vectorLength; j++) {
                res[j] += M[i][j];
            }
        }

        // console.log({ res })
        return res as Vector;
    }

    zeros(rows: number, cols: number): Matrix {
        const array = new Array(rows);

        for (let i = 0; i < rows; i++) {
            array[i] = new Float32Array(cols);
        }

        return array;
    }

    initMat(M: Matrix): void {
        const magnitude: number = 0.2;
        for (let i = 0; i < M.length; i++) {

            for (let j = 0; j < M[i].length; j++) {
                M[i][j] = magnitude - Math.random() * magnitude;
            }
        }
    };

    getCorpus(name: string): string {
        const content = this.fileService.readFileSync(name);
        return content.toString();
    }

    dot(a: Vector, b: Vector) {
        let sum = 0;
        for (let i = 0; i < a.length; i++) sum += a[i] * b[i];

        return sum;
    }

    findTopKCandidates(v: Vector, embeddings: Matrix, k = 5) {
        const scores = embeddings.map(e => this.dot(v, e)); // dot product for each token

        // get indices sorted by score (descending)
        const sortedIndices = scores
          .map((s, i) => [s, i])
          .sort((a, b) => b[0] - a[0])
          .slice(0, k)
          .map(([_, i]) => i);

        return sortedIndices; // top-k token indices
    }

    normalizeVector(v: Vector) {
        // @ts-ignore
        const norm = Math.sqrt(v.reduce((s, x) => s + x * x, 0));

        return v.map(x => x / (norm || 1));
    }

    layerNormRowwise(X: Matrix, eps = 1e-5, gamma: Vector = null, beta: Vector = null) {
        const m = X.length, n = X[0].length;
        const out = this.zeros(m, n);

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
}
