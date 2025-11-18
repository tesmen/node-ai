import db from '../../db/knex';

export class SessionEntity {
    static async create(data: SessionInterface): Promise<number> {
        const res = await db('sessions')
          .insert(data)
          .returning('id');

        return res.pop().id;
    }

    static async finishRun(runId: number, data: Record<string, any>) {
        await db('sessions').update(
          Object.assign({ finished_at: new Date }, data)
        ).where({ id: runId });
    }
}

interface SessionInterface {
    id?: number;
    model_id?: number;
    created_at?: number;
    finished_at?: number;
    iterations?: number;
    correct_ratio?: number;
    window_size?: number;
    use_slide?: boolean;
}