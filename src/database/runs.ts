import db from '../../db/knex';

export class Runs {
    static async createRun(source: string, nemb: number, nctx: number, iterations: number): Promise<number> {
        const res = await db('runs').insert(
          {
              source,
              iterations,
              nemb,
              nctx,
          }
        ).returning('id');

        return res.pop().id;
    }

    static async finishRun(runId: number) {
        const res = await db('runs').update(
          { finished_at: new Date }
        ).where({ id: runId });
    }
}