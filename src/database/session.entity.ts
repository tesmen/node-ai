import db from '../../db/knex';
import { TrainingSessionConfig } from '../models/training-session.config';

export class SessionEntity {
    static async create(data: SessionInterface): Promise<SessionInterface> {
        const res = await db('sessions')
          .insert(data)
          .returning('id');

        const id = res.pop().id;

        return this.getById(id);
    }

    static async createConfig(data: SessionInterface): Promise<TrainingSessionConfig> {
        const sessionModel = await this.create(data);

        return {
            sessionId: sessionModel.id,
            iterations: sessionModel.iterations,
            useSlide: sessionModel.use_slide,
        } as TrainingSessionConfig;
    }

    static async finishRun(runId: number, data: Record<string, any>) {
        await db('sessions').update(
          Object.assign({ finished_at: new Date }, data)
        ).where({ id: runId });
    }

    static async getById(id: number): Promise<SessionInterface> {
        const res = await db('sessions')
          .where({ id });

        return res.pop() as SessionInterface;
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
    adjust_pte?: boolean;
}