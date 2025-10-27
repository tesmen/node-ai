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
        this.tokenizer = new CharTokenizer(this.getCorpus());
        console.log('vocab size', this.tokenizer.vocabSize);

        // Token Embeddings
        if (cfg.wteFile) {
            this.wte = JSON.parse(this.fileService.readFileSync(cfg.wteFile).toString());
            console.log('WTE read from file');
        } else {
            this.wte = this.zeros(this.tokenizer.vocabSize, cfg.nEmbd); // token embeddings
            [this.wte].forEach(this.initMat);
        }

        // Positional Embeddings
        if (cfg.wpeFile) {
            this.wpe = JSON.parse(this.fileService.readFileSync(cfg.wpeFile).toString());
            console.log('WPE read from file');
        } else {
            this.wpe = this.zeros(cfg.nCtx, cfg.nEmbd);      // positional embeddings
            [this.wpe].forEach(this.initMat);
        }
        console.log(cfg, { wpe: this.wpe.length, wte: this.wte.length });
    }

    saveWeights(prefix: string) {
        this.fileService.writeFileSync(`./weights/wpe-${prefix}.json`, JSON.stringify(this.wpe));
        this.fileService.writeFileSync(`./weights/wte-${prefix}.json`, JSON.stringify(this.wte));
    }


    generate(prompt: string, maxNewTokens: number = 10) {
        const ids = this.tokenizer.encode(prompt);
        // console.log(ids);

        for (let step = 0; step < maxNewTokens; step++) {
            const logits = this.forward(ids);
            ids.push(logits.shift());
        }

        return {
            ids,
            out: this.tokenizer.decode(ids)
        };
    }

    trainIterations(iterations: number, windowSize: number) {
        return this.train(windowSize);
    }

    train(windowSize: number) {
        // windowSize = this.cfg.nCtx
        const set: Set<string> = this.tokenizer.tokenize(this.getCorpus());
        const arrayCopy: string[] = Array.from(set);
        let sampleArray: string[];
        let step = 0;

        while ((sampleArray = arrayCopy.slice(step * windowSize, (step + 1) * windowSize)).length) {
            console.log({ sampleArray });
            step++;

            for (let i = 1; i < sampleArray.length; i++) {
                const prompt = sampleArray.slice(0, i).join(' ');
                const ids = this.tokenizer.encode(prompt);
                const logits = this.forward(ids);
                const expectedTokenId = this.tokenizer.encodeOne(sampleArray[i]);
                const logit = logits[0];

                console.log('training on:', {
                    prompt,
                    expected: sampleArray[i],
                    logit: logits[0],
                    logitText: this.tokenizer.decodeOne(logit)
                });

                if (expectedTokenId !== logit) {
                    // this.adjustEmbeddings()
                }

            }
        }
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

    initNormalizedMat(M: Matrix = []): void {
        const matrixSize = M[0].length;

        for (let i = 0; i < M.length; i++) {
            const vector: Vector = [];

            for (let j = 0; j < matrixSize; j++) {
                vector.push(1 - 2 * Math.random());
            }

            M[i] = this.normalizeVector(vector);
        }
    };

    getCorpus(): string {
        const content = this.fileService.readFileSync(this.cfg.corpusFile);
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
        // console.log(
        //   'report',
        //   scores
        //     .map((score, index) => {
        //         return {
        //             index,
        //             token: this.tokenizer.embed(index),
        //             score,
        //         };
        //     })
        //     .sort((a, b) => b.score - a.score)
        // );


        // get indices sorted by score (descending)
        const sortedIndices = scores
          .map((score, index) => [score, index])
          .sort((a, b) => b[0] - a[0])
          .slice(0, k)
          .map(([score, index]) => index);
        // console.log({ sortedIndices })
        return sortedIndices; // top-k token indices
    }

    normalizeVector(v: Vector) {
        const norm = Math.sqrt(
          v.reduce((acc, num) => acc + num * num, 0)
        );
        // console.log({norm})

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

    adjustEmbeddings(promptVec: Vector, targetVec: Vector, learningRate = 0.1) {
        const newPrompt = [];
        const newTarget = [];

        for (let i = 0; i < promptVec.length; i++) {
            const delta = learningRate * (targetVec[i] - promptVec[i]);
            newPrompt.push(promptVec[i] + delta);  // move prompt toward target
            newTarget.push(targetVec[i] + delta);  // move target toward prompt
        }

        return { newPrompt, newTarget };
    }

    embed(id: number): Vector {
        if (!this.wte[id]) {
            throw new Error('Token out of range error');
        }

        return this.wte[id];
    }
}
