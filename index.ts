import { RunEntity } from './src/database/run.entity';
import { ModelConfig } from './src/interfaces/ModelConfig';
import { TrainingSessionConfig } from './src/models/training-session.config';
import { TrainingSession } from './src/models/training.session';
import { SimpleBookProcessor } from './src/services/simple-book.processor';

(async () => {
    const modelCfg: ModelConfig = {
        // corpusFile: './books/Robert Sheckley - The Dream of Misunderstanding - 2002.txt',
        // corpusFile: './books/candp.nano.txt',
        corpusFile: './books/candp.min.txt',
        // corpusFile: './books/candp.med.txt',
        // corpusFile: './books/candp.txt',
        nEmbd: 64,
        nHidden: 128,
        nCtx: 64,
        iterations: 100,
        useSlide: false,
        // useSlide: true,
    };


    const model = new SimpleBookProcessor(modelCfg);

    const runId = await RunEntity.createRun2(
      {
          source: modelCfg.corpusFile,
          nemb: modelCfg.nEmbd,
          nctx: modelCfg.nCtx,
          iterations: modelCfg.iterations,
          corpus_length: model.getCorpus().length,
          wte_length: model.wte.length,
          window_size: modelCfg.trainWindow,
          use_slide: modelCfg.useSlide,
      }
    );

    const sessionConfig = {
        corpusFile: './books/candp.min.txt',
        id: runId,
        iterations: 100,
        useSlide: false
    } as TrainingSessionConfig;

    const session = new TrainingSession(sessionConfig, model);
    await session.run();


    try {
        const response = model.generate('The');
        console.log('response', response);
    } catch (e) {
        console.log(e);
    }
    process.exit()
})();

