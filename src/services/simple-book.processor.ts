import { Result } from '../database/result';
import { Runs } from '../database/runs';
import { ModelConfig } from '../interfaces/ModelConfig';
import { FileServiceAdapter } from './file-service.adapter';
import { CharTokenizer } from './char.tokenizer';

type Vector = number[];
type Matrix = Vector[];


export class SmartVector extends Array {
    constructor(...items: any[]) {
        super(...items);
    }

    add(v: Vector) {
        if (this.length != v.length) {
            throw Error('this.length != v.length');
        }

        this.forEach((el, i) => this[i] += v[i]);

        return this;
    }
}


export class SimpleBookProcessor {
    wte: Matrix = [];
    wpe: Matrix = [];
    fileService: FileServiceAdapter;
    tokenizer: CharTokenizer;
    private cfg: ModelConfig;

    constructor(cfg: ModelConfig) {
        this.cfg = cfg;
        this.fileService = new FileServiceAdapter();
        this.tokenizer = new CharTokenizer(this.getCorpus());
        this.setupWte();
        this.setupWpe();

        this.log(cfg, {
              wpeLength: this.wpe.length,
              wteLength: this.wte.length,
              vocabSize: this.tokenizer.vocabSize
          }
        );
    }

    saveWeights(prefix: string) {
        this.fileService.writeFileSync(`./weights/wpe-${prefix}.json`, JSON.stringify(this.wpe));
        this.fileService.writeFileSync(`./weights/wte-${prefix}.json`, JSON.stringify(this.wte));
    }

    setupWte() {
        // Token Embeddings
        if (this.cfg.wteFile) {
            this.wte = JSON.parse(this.fileService.readFileSync(this.cfg.wteFile).toString());
            this.log('WTE read from file');
        } else {
            this.wte = this.zeros(this.tokenizer.vocabSize, this.cfg.nEmbd); // token embeddings
            [this.wte].forEach(this.initMat);
        }
    }

    setupWpe() {
        // Positional Embeddings
        if (this.cfg.wpeFile) {
            this.wpe = JSON.parse(this.fileService.readFileSync(this.cfg.wpeFile).toString());
            this.log('WPE read from file');
        } else {
            this.wpe = this.zeros(this.cfg.nCtx, this.cfg.nEmbd);      // positional embeddings
            [this.wpe].forEach(this.initMat);
        }
    }

    generate(prompt: string, maxNewTokens: number = 20) {
        const ids = this.tokenizer.encode(prompt);
        // this.log(ids);

        for (let step = 0; step < maxNewTokens; step++) {
            const logits = this.forward(ids);
            ids.push(logits.shift());
        }

        return {
            ids,
            out: this.tokenizer.decode(ids)
        };
    }

    async trainIterations(config: ModelConfig): Promise<void> {
        let round = { error: 0, correct: 0, ratio: 0 };

        for (let iteration = 0; iteration < config.iterations; iteration++) {
            const windowSize = config.trainWindow || config.nCtx;
            const { error, correct, shift } = this.train(windowSize, iteration);
            round.correct = correct;
            round.error = error;
            round.ratio = Number((round.correct / (round.error + round.correct)).toFixed(3)) || 0;
            this.log('>>> Iteration finished ', { iteration, round });

            await Result.create({
                  run_id: config.id,
                  error,
                  correct,
                  iteration: iteration,
              }
            );
        }

        await Runs.finishRun(this.cfg.id, { correct_ratio: round.ratio });
    }

    // a single run over provided corpus file
    train(windowSize: number, iteration: number): { error: number; correct: number; shift: number } {
        const corpusArray: string[] = this.tokenizer.separate(this.getCorpus());
        let sampleArray: string[];
        let step = 0;
        let error = 0;
        let correct = 0;
        let shift;

        if (this.cfg.useSlide) {
            shift = corpusArray.length < windowSize
              ? iteration % corpusArray.length
              : iteration % windowSize;
        } else {
            shift = 0;
        }

        while ((sampleArray = corpusArray.slice(shift + step * windowSize, shift + (step + 1) * windowSize)).length) {
            step++;

            for (let i = 1; i < sampleArray.length; i++) {
                const prompt = sampleArray.slice(0, i).join(' ');
                const ids = this.tokenizer.encode(prompt);

                const logits = this.forward(ids);
                const expectedTokenId = this.tokenizer.encodeOne(sampleArray[i]);
                const logit = logits[0];

                if (expectedTokenId !== logit) {
                    const promptVector = this.createPromptVector(ids);
                    const adjusted = this.adjustEmbeddings(promptVector, this.embed(expectedTokenId));
                    this.wte[expectedTokenId] = adjusted.newTarget;
                    // this.wte[logit  ] = adjusted.newTarget;
                    // this.log('adjusted.', JSON.stringify(adjusted.newTarget));
                    // this.log('adjusted.oldTarget', JSON.stringify(this.embed(expectedTokenId)));
                    error++;
                } else {
                    correct++;
                }

                // this.log('training on:',
                //   {
                //       sampleArray: sampleArray.join(' '),
                //       shift,
                //       prompt,
                //       expected: sampleArray[i],
                //       logitText: this.tokenizer.decodeOne(logit),
                //       logits: logits.slice(0, 10),
                //       logit: logits[0],
                //       correct: expectedTokenId === logit,
                //   }
                // );
            }
        }

        return { error, correct, shift };
    }

    private euclideanDistance(a: Vector, b: Vector) {
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

    private buildPromptMatrix(inputIds: number[]) {
        let promptMatrix = this.zeros(inputIds.length, this.cfg.nEmbd);
        // this.log({ promptMatrix });

        for (let inputsArrIndex = 0; inputsArrIndex < inputIds.length; inputsArrIndex++) {
            const token = this.wte[inputIds[inputsArrIndex]];
            const position = this.wpe[inputsArrIndex];

            for (let j = 0; j < this.cfg.nEmbd; j++) {
                promptMatrix[inputsArrIndex][j] = token[j] + position[j];
            }
        }

        return promptMatrix;
    }

    private createPromptVector(promptIds: number[]): Vector {
        let promptMatrix = this.buildPromptMatrix(promptIds);
        // this.log({ promptMatrix });
        const normalizedX = this.layerNormRowwise(promptMatrix, 1e-5);
        // this.log({ normalizedX })
        const resultingVector = this.reduceM2Vector(normalizedX);
        // this.log({ resultingVector })
        const normalizedResVector = this.normalizeVector(resultingVector);
        // this.log({ normalizedResVector });

        return normalizedResVector;
    }

    private forward(inputIds: number[]): number[] {
        const promptVector = this.createPromptVector(inputIds);
        const candidates = this.findTopKCandidates(promptVector, this.wte);
        // this.log(candidates)

        return candidates;
    }

    private reduceM2Vector(M: Matrix): Vector {
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

    private zeros(rows: number, cols: number): Matrix {
        const array = new Array(rows);

        for (let i = 0; i < rows; i++) {
            array[i] = new Array(cols);
        }

        return array;
    }

    private initMat(M: Matrix): void {
        const magnitude: number = 0.2;
        for (let i = 0; i < M.length; i++) {

            for (let j = 0; j < M[i].length; j++) {
                M[i][j] = magnitude / 2 - Math.random() * magnitude;
            }
        }
    };

    private initNormalizedMat(M: Matrix = []): void {
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

    private dot(a: Vector, b: Vector) {
        let sum = 0;
        for (let i = 0; i < a.length; i++) {
            sum += a[i] * b[i];
        }

        return sum;
    }

    findTopKCandidates(v: Vector, embeddings: Matrix, k = 5): number[] {
        // this.log(`Looking for candidate`, v);
        const scores = embeddings.map(e => this.dot(v, e)); // dot product for each token
        // this.log(
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
        // this.log({ sortedIndices })
        return sortedIndices; // top-k token indices
    }

    normalizeVector(v: Vector) {
        const norm = Math.sqrt(
          v.reduce((acc, num) => acc + num * num, 0)
        );
        // this.log({norm})

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

    adjustEmbeddings(promptVec: Vector, targetVec: Vector, learningRate = 0.05) {
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

    embed(id: number): Vector {
        if (!this.wte[id]) {
            throw new Error('Token out of range error');
        }

        return this.wte[id];
    }

    private log(...msg: any) {
        console.log(msg);
    }
}
