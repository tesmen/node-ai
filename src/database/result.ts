import db from '../../db/knex';

export class Result {
    static async create(data: ResultInterface): Promise<number> {
        const res = await db('results')
          .insert(data)
          .returning('id');

        return res.pop().id;
    }

}

interface ResultInterface {
    id?: number;
    run_id: number;
    iteration: number;
    error: number;
    correct: number;
}
