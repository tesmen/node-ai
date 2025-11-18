import { ModelEntity } from './src/database/model.entity';
import { SessionEntity } from './src/database/session.entity';
import { ModelConfig } from './src/interfaces/ModelConfig';
import { TrainingSessionConfig } from './src/models/training-session.config';
import { TrainingSession } from './src/models/training.session';
import { SimpleBookProcessor } from './src/services/simple-book.processor';

(async () => {
    const modelCfg: ModelConfig = {
        corpusFile: './books/candp.nano.txt',
        // corpusFile: './books/candp.min.txt',
        // corpusFile: './books/candp.med.txt',
        // corpusFile: './books/candp.txt',
        nEmbd: 64,
        nHidden: 128,
        nCtx: 64,
    };

    const model = new SimpleBookProcessor(modelCfg);
    await ModelEntity.save(model);

    const runId = await SessionEntity.create(
      {
          model_id: model.id,
          iterations: 100,
          window_size: modelCfg.nCtx,
          use_slide: false,
      }
    );

    const sessionConfig = {
        corpusFile: './books/candp.min.txt',
        sessionId: runId,
        iterations: 100,
        useSlide: false,
    } as TrainingSessionConfig;

    const session = new TrainingSession(sessionConfig, model);
    await session.run();

    try {
        const response = model.generate('The');
        console.log('response', response);
    } catch (e) {
        console.log(e);
    }

    process.exit();
})();

