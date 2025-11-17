import { Model } from '../database/model';
import { CharTokenizer } from '../services/char.tokenizer';
import { Matrix } from '../types';

export class LlmModel {
    wte: Matrix;
    wpe: Matrix;
    tokenizer: CharTokenizer;
    static async fromDatabase(id: number): Promise<LlmModel> {
        return await Model.getById(id) as LlmModel;
    }

    static create(nemb: number, nctx: number, nHidden: number): LlmModel {
        return new LlmModel();
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
    private forward(inputIds: number[]): number[] {
        const promptVector = this.createPromptVector(inputIds);
        const candidates = this.findTopKCandidates(promptVector, this.wte);
        // this.log(candidates)

        return candidates;
    }

    async save() {
        await Model.create(this);
    }
}