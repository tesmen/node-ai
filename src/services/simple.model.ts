import {
    dotProduct,
    initMat,
    layerNormRowwise,
    normalizeVectorL2,
    reduceM2Vector,
    zeros
} from '../fns';
import { ModelConfig } from '../interfaces/ModelConfig';
import { Matrix, Vector } from '../types';
import { FileServiceAdapter } from './file-service.adapter';
import { CharTokenizer } from './char.tokenizer';

export class SimpleModel {
    id: number;
    wte: Matrix = [];
    wpe: Matrix = [];
    tokenizer: CharTokenizer;
    cfg: ModelConfig;
    sourceLength: number;

    constructor(cfg: ModelConfig, tokenizer: CharTokenizer = null) {
        this.id = cfg.id;
        this.cfg = cfg;

        if (tokenizer) {
            this.tokenizer = tokenizer;
        } else {
            this.tokenizer = new CharTokenizer();
            const corpus = FileServiceAdapter.getTextContent(this.cfg.source);
            this.tokenizer.init(corpus);
            this.sourceLength = corpus.length;
        }

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
        if(maxNewTokens> this.cfg.nCtx){
            throw Error(`Context length exceeded of ${this.cfg.nCtx} by ${maxNewTokens}`)
        }
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

    buildPromptMatrix(inputIds: number[]): Matrix {
        let promptMatrix = zeros(inputIds.length, this.cfg.nEmbd);

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
        const normalizedResVector = normalizeVectorL2(resultingVector);

        return normalizedResVector;
    }

    forward(inputIds: number[]): number[] {
        const promptVector = this.createPromptVector(inputIds);
        return this.findTopKCandidates(promptVector, this.wte);
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

