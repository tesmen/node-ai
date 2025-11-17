import db from '../../db/knex';

export class Model {
    static async create(data: ModelInterface): Promise<number> {
        const res = await db('models')
          .insert(data)
          .returning('id');

        return res.pop().id;
    }

    static async getById(id: number): Promise<ModelInterface> {
        const res = await db('models')
          .where({ id });

        return res as ModelInterface;
    }

}

interface ModelInterface {
    id?: number;
    created_at?: number;
    finished_at?: number;
    source?: string;
    nemb?: number;
    nctx?: number;
    iterations?: number;
    wte_length?: number;
    corpus_length?: number;
    correct_ratio?: number;
    window_size?: number;
    use_slide?: boolean;
}