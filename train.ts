import { ModelEntity } from './src/database/model.entity';
import { SessionEntity } from './src/database/session.entity';
import { ModelConfig } from './src/interfaces/ModelConfig';
import { TrainingSession } from './src/models/training.session';
import { SimpleBookProcessor } from './src/services/simple-book.processor';

(async () => {
    const modelCfg: ModelConfig = {
        // source: './books/candp.nano.txt',
        source: './books/abc.txt',
        // source: './books/candp.med.txt',
        // source: './books/candp.txt',
        nEmbd: 64,
        nHidden: 128,
        nCtx: 64,
    };

    const model = new SimpleBookProcessor(modelCfg);
    await ModelEntity.save(model);

    const sessionConfig = await SessionEntity.createConfig(
      {
          model_id: model.id,
          iterations: 500,
          window_size: modelCfg.nCtx,
          use_slide: false,
          adjust_pte: true,
      }
    );

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

