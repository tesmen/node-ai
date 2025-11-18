import { ResultEntity } from '../database/result.entity';
import { RunEntity } from '../database/run.entity';
import {
    adjustEmbeddings,
    dotProduct,
    initMat,
    layerNormRowwise,
    normalizeVector,
    reduceM2Vector,
    zeros
} from '../fns';
import { ModelConfig } from '../interfaces/ModelConfig';
import { Matrix, Vector } from '../types';
import { FileServiceAdapter } from './file-service.adapter';
import { CharTokenizer } from './char.tokenizer';

export class SimpleBookProcessor {
    wte: Matrix = [];
    wpe: Matrix = [];
    fileService: FileServiceAdapter;
    tokenizer: CharTokenizer;
    cfg: ModelConfig;

    constructor(cfg: ModelConfig) {
        this.cfg = cfg;
        this.fileService = new FileServiceAdapter();
        this.tokenizer = new CharTokenizer();
        this.tokenizer.init(FileServiceAdapter.getTextContent(this.cfg.corpusFile));
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
            this.wte = zeros(this.tokenizer.vocabSize, this.cfg.nEmbd); // token embeddings
            [this.wte].forEach(initMat);
        }
    }

    setupWpe() {
        // Positional Embeddings
        if (this.cfg.wpeFile) {
            this.wpe = JSON.parse(this.fileService.readFileSync(this.cfg.wpeFile).toString());
            this.log('WPE read from file');
        } else {
            this.wpe = zeros(this.cfg.nCtx, this.cfg.nEmbd);      // positional embeddings
            [this.wpe].forEach(initMat);
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

            await ResultEntity.create({
                  run_id: config.id,
                  error,
                  correct,
                  iteration: iteration,
              }
            );
        }

        await RunEntity.finishRun(this.cfg.id, { correct_ratio: round.ratio });
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
                    const adjusted = adjustEmbeddings(promptVector, this.embed(expectedTokenId));
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

    private buildPromptMatrix(inputIds: number[]) {
        let promptMatrix = zeros(inputIds.length, this.cfg.nEmbd);
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

    createPromptVector(promptIds: number[]): Vector {
        let promptMatrix = this.buildPromptMatrix(promptIds);
        const normalizedX = layerNormRowwise(promptMatrix, 1e-5);
        const resultingVector = reduceM2Vector(normalizedX);
        const normalizedResVector = normalizeVector(resultingVector);

        return normalizedResVector;
    }

    forward(inputIds: number[]): number[] {
        const promptVector = this.createPromptVector(inputIds);
        return this.findTopKCandidates(promptVector, this.wte);
    }

    getCorpus(): string {
        const content = this.fileService.readFileSync(this.cfg.corpusFile);

        return content.toString();
    }

    findTopKCandidates(v: Vector, embeddings: Matrix, k = 5): number[] {
        // this.log(`Looking for candidate`, v);
        const scores = embeddings.map(e => dotProduct(v, e)); // dot product for each token
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

