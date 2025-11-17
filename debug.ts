import { Runs } from './src/database/runs';
import { ModelConfig } from './src/interfaces/ModelConfig';
import { SimpleBookProcessor } from './src/services/simple-book.processor';

(async () => {
    const cfg: ModelConfig = {
        // corpusFile: './books/Robert Sheckley - The Dream of Misunderstanding - 2002.txt',
        corpusFile: './books/candp.nano.txt',
        // corpusFile: './books/candp.med.txt',
        // corpusFile: './books/candp.min.txt',
        // corpusFile: './books/candp.txt',
        nEmbd: 64,
        nHidden: 128,
        nCtx: 64,
        iterations: 500,
        trainWindow: 64,
        // useSlide: false,
        useSlide: true,
    };

    const model = new SimpleBookProcessor(cfg);

    cfg.id = await Runs.createRun2(
      {
          source: cfg.corpusFile,
          nemb: cfg.nEmbd,
          nctx: cfg.nCtx,
          iterations: cfg.iterations,
          corpus_length: model.getCorpus().length,
          wte_length: model.wte.length,
          window_size: cfg.trainWindow,
          use_slide: cfg.useSlide,
      }
    );

    await model.trainIterations(cfg);

    // const a = model.generate('actions had stirred');
    // console.log(a);
    process.exit();

})();