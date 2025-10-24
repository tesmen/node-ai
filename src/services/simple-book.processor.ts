import { ModelConfig } from '../interfaces/ModelConfig';
import { FileServiceAdapter } from './file-service.adapter';
import { CharTokenizer } from './char.tokenizer';

type Vector = number[];
type Matrix = Vector[];

export class SimpleBookProcessor {
    wte: Matrix = [];
    wpe: Matrix = [];
    fileService: FileServiceAdapter;
    private tokenizer: CharTokenizer;
    private cfg: ModelConfig;

    constructor(cfg: ModelConfig) {
        this.cfg = cfg;
        this.fileService = new FileServiceAdapter();
        this.tokenizer = new CharTokenizer(this.getCorpus(cfg.corpusFile));
        console.log('vocab size', this.tokenizer.vocabSize);

        // Embeddings
        if (cfg.wteFile) {
            this.wte = JSON.parse(this.fileService.readFileSync(cfg.wteFile).toString());
            console.log('WTE read from file');
        } else {
            this.wte = this.zeros(this.tokenizer.vocabSize, cfg.nEmbd); // token embeddings
            [this.wte].forEach(this.initMat);
        }

        if (cfg.wpeFile) {
            this.wpe = JSON.parse(this.fileService.readFileSync(cfg.wpeFile).toString());
            console.log('WPE read from file');
        } else {
            this.wpe = this.zeros(cfg.nCtx, cfg.nEmbd);      // positional embeddings
            [this.wpe].forEach(this.initMat);
        }
        console.log(cfg, { wpe: this.wpe.length, wte: this.wte.length });
    }

    generate(prompt: string, maxNewTokens: number = 1) {
        const ids = this.tokenizer.encode(prompt);
        // console.log(ids);

        for (let step = 0; step < maxNewTokens; step++) {
            const logits = this.forward(ids);
            ids.push(logits.shift());
        }

        return this.tokenizer.decode(ids);
    }

    buildPromptMatrix(inputIds: number[]) {
        let promptMatrix = this.zeros(inputIds.length, this.cfg.nEmbd);
        // console.log({ promptMatrix });

        for (let inputsArrIndex = 0; inputsArrIndex < inputIds.length; inputsArrIndex++) {
            const token = this.wte[inputIds[inputsArrIndex]];
            const position = this.wpe[inputsArrIndex];

            for (let j = 0; j < this.cfg.nEmbd; j++) {
                promptMatrix[inputsArrIndex][j] = token[j] + position[j];
            }
        }

        return promptMatrix;
    }

    forward(inputIds: number[]) {
        let promptMatrix = this.buildPromptMatrix(inputIds);
        // console.log({ promptMatrix });
        const normalizedX = this.layerNormRowwise(promptMatrix, 1e-5);
        // console.log({ normalizedX })
        const resultingVector = this.reduceM2Vector(normalizedX);
        // console.log({ resultingVector })
        const normalizedResVector = this.normalizeVector(resultingVector);
        // console.log({ normalizedResVector });

        const candidates = this.findTopKCandidates(normalizedResVector, this.wte);
        // console.log(candidates)

        return candidates;
    }

    reduceM2Vector(M: Matrix): Vector {
        const vectorLength = M[0].length;
        const res: Vector = (new Array(vectorLength)).fill(0);

        for (let i = 0; i < M.length; i++) {
            for (let j = 0; j < vectorLength; j++) {
                res[j] += M[i][j];
            }
        }

        // console.log({ res })
        return res;
    }

    zeros(rows: number, cols: number): Matrix {
        const array = new Array(rows);

        for (let i = 0; i < rows; i++) {
            array[i] = new Array(cols);
        }

        return array;
    }

    initMat(M: Matrix): void {
        const magnitude: number = 0.2;
        for (let i = 0; i < M.length; i++) {

            for (let j = 0; j < M[i].length; j++) {
                M[i][j] = magnitude / 2 - Math.random() * magnitude;
            }
        }
    };

    getCorpus(name: string): string {
        const content = this.fileService.readFileSync(name);
        return content.toString();
    }

    dot(a: Vector, b: Vector) {
        let sum = 0;
        for (let i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }

        return sum;
    }

    findTopKCandidates(v: Vector, embeddings: Matrix, k = 5) {
        // console.log(`Looking for candidate`, v);
        const scores = embeddings.map(e => this.dot(v, e)); // dot product for each token
        console.log(
          scores
            .map((score, index) => {
                return {
                    index,
                    token: this.tokenizer.decodeSingle(index),
                    score,
                };

            })
            .sort((a, b) => b.score - a.score)
        );


        // get indices sorted by score (descending)
        const sortedIndices = scores
          .map((score, index) => [score, index])
          .sort((a, b) => b[0] - a[0])
          .slice(0, k)
          .map(([_, i]) => i);
        // console.log({ sortedIndices })
        return sortedIndices; // top-k token indices
    }

    normalizeVector(v: Vector) {
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
