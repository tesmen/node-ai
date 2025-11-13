import { Runs } from './src/database/runs';
import { ModelConfig } from './src/interfaces/ModelConfig';
import { SimpleBookProcessor } from './src/services/simple-book.processor';

const useDbStorage = false;

(async () => {
    const cfg: ModelConfig = {
        // corpusFile: './books/Robert Sheckley - The Dream of Misunderstanding - 2002.txt',
        // corpusFile: './books/candp.nano.txt',
        // corpusFile: './books/candp.min.txt',
        corpusFile: './books/candp.txt',
        nEmbd: 64,
        nHidden: 128,
        nCtx: 64,
        iterations: 100,
        trainWindow: 64
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
      }
    );


    const stat = await model.trainIterations(cfg);

    // const a = model.generate('actions had stirred');
    // console.log(a);
    const lastStat = stat.pop();
    const ratio = lastStat.correct / (lastStat.error + lastStat.correct);
    await Runs.finishRun(cfg.id, { correct_ratio: ratio });
    process.exit();

})();