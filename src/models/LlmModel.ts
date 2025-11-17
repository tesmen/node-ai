import { Model } from '../database/model';
import { Matrix } from '../types';

export class LlmModel {
    wte: Matrix;
    wpe: Matrix;

    static async fromDatabase(id: number): Promise<LlmModel> {
        return await Model.getById(id) as LlmModel;
    }

    static create(nemb: number, nctx: number, nHidden: number): LlmModel {
        return new LlmModel();
    }

    async save() {
        await Model.create(this);
    }
}