import {
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
    id: number;
    wte: Matrix = [];
    wpe: Matrix = [];
    fileService: FileServiceAdapter;
    tokenizer: CharTokenizer;
    cfg: ModelConfig;
    sourceLength: number;

    constructor(cfg: ModelConfig) {
        this.cfg = cfg;
        this.fileService = new FileServiceAdapter();
        this.tokenizer = new CharTokenizer();
        const corpus = FileServiceAdapter.getTextContent(this.cfg.corpusFile);
        this.tokenizer.init(corpus);
        this.sourceLength = corpus.length;
        this.setupWte();
        this.setupWpe();

        this.log(
          // cfg,
          {
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
        if (this.cfg.wte) {
            this.wte = this.cfg.wte;
            this.log('WTE read from cfg');
        } else {
            this.wte = zeros(this.tokenizer.vocabSize, this.cfg.nEmbd); // token embeddings
            [this.wte].forEach(initMat);
        }
    }

    setupWpe() {
        // Positional Embeddings
        if (this.cfg.wpe) {
            this.wpe = this.cfg.wpe;
            this.log('WPE read from cfg');
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

    buildPromptMatrix(inputIds: number[]) {
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

