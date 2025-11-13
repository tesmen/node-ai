import db from '../../db/knex';

export class Runs {
    static async createRun(source: string,
                           nemb: number,
                           nctx: number,
                           iterations: number,
                           corpus_length: number,
                           wte_length: number
    ): Promise<number> {
        const res = await db('runs').insert(
          {
              source,
              iterations,
              nemb,
              nctx,
              corpus_length,
              wte_length
          }
        ).returning('id');

        return res.pop().id;
    }

    static async createRun2(data: RunInterface): Promise<number> {
        const res = await db('runs')
          .insert(data)
          .returning('id');

        return res.pop().id;
    }

    static async finishRun(runId: number, data: Record<string, any>) {
        await db('runs').update(
          Object.assign({ finished_at: new Date }, data)
        ).where({ id: runId });
    }
}

interface RunInterface {
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
}