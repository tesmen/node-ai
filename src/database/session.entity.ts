import db from '../../db/knex';
import { TrainingSessionConfig } from '../models/training-session.config';

export class SessionEntity {
    static async create(data: TrainingSessionConfig): Promise<TrainingSessionConfig> {
        const res = await db('sessions')
          .insert(data)
          .returning('id');

        const id = res.pop().id;

        return this.getById(id);
    }

    static async finishRun(runId: number, data: Record<string, any>) {
        await db('sessions').update(
          Object.assign({ finished_at: new Date }, data)
        ).where({ id: runId });
    }

    static async getById(id: number): Promise<TrainingSessionConfig> {
        const res = await db('sessions')
          .where({ id });

        return res.pop() as TrainingSessionConfig;
    }
}