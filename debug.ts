import { Runs } from './src/database/runs';
import { ModelConfig } from './src/interfaces/ModelConfig';
import { SimpleBookProcessor } from './src/services/simple-book.processor';

(async () => {
    const cfg: ModelConfig = {
        corpusFile: './books/candp.min.txt',
        nEmbd: 256,
        nHidden: 128,
        nCtx: 16,
        iterations: 500
    };

    const model = new SimpleBookProcessor(cfg);

    const batch = Math.round((new Date).getTime() / 1000);
    cfg.id = await Runs.createRun(cfg.corpusFile, cfg.nEmbd, cfg.nCtx, cfg.iterations);

    model.saveWeights(batch + 'pre');
    const stat = await model.trainIterations(cfg);
    model.saveWeights(batch + 'post');
    model.fileService.writeFileSync(`./weights/${batch}-stat.json`, JSON.stringify(stat));

    await Runs.finishRun(cfg.id);
    process.exit();

})();