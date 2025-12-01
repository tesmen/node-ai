import db from '../../db/knex';
import { CharTokenizer } from '../services/char.tokenizer';
import { SimpleModel } from '../services/simple.model';

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

    static async load(id: number): Promise<SimpleModel> {
        const raw = await ModelEntity.getById(id);
        const tokenizer = new CharTokenizer(raw.itos, new Map(raw.stoi));

        return new SimpleModel(
          {
              source: raw.source,
              nEmbd: raw.nemb,
              nCtx: raw.nctx,
              wte: raw.wte,
              wpe: raw.wpe,
              id: raw.id
          },
          tokenizer
        );
    }

    static async save(model: SimpleModel) {
        const data: ModelInterface = {
            nemb: model.cfg.nEmbd,
            nctx: model.cfg.nCtx,
            nhidden: model.cfg.nHidden,
            source: model.cfg.source,
            corpus_length: model.sourceLength,

            wpe: JSON.stringify(model.wpe),
            wte: JSON.stringify(model.wte),
            itos: JSON.stringify(model.tokenizer.itos),
            stoi: JSON.stringify(Array.from(model.tokenizer.stoi.entries())),
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

    itos: any;
    stoi: any;
}