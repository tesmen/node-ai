import db from '../../db/knex';
import { SimpleBookProcessor } from '../services/simple-book.processor';

export class ModelEntity {
    static async create(data: ModelInterface): Promise<number> {
        const res = await db('models')
          .insert(data)
          .returning('id');

        return res.pop().id;
    }

    static async getById(id: number): Promise<ModelInterface> {
        const res = await db('models')
          .where({ id });

        return res.pop() as ModelInterface;
    }

    static async update(id: number, data: ModelInterface) {
        await db('models')
          .where({ id })
          .update(data);
    }

    static async save(model: SimpleBookProcessor) {
        const data: ModelInterface = {
            nemb: model.cfg.nEmbd,
            nctx: model.cfg.nCtx,
            nhidden: model.cfg.nHidden,
            source: model.cfg.corpusFile,
            corpus_length: model.sourceLength,

            wpe: JSON.stringify(model.wpe),
            wte: JSON.stringify(model.wte),
        };

        if (model.id) {
            await ModelEntity.update(model.id, data);
        } else {
            model.id = await ModelEntity.create(data);
        }
    }
}

interface ModelInterface {
    id?: number;
    created_at?: number;
    finished_at?: number;
    source: string;
    nemb: number;
    nctx: number;
    nhidden: number;
    token_length?: number;
    corpus_length?: number;
    correct_ratio?: number;

    wpe: any;
    wte: any;
}