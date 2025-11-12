import { Runs } from './src/database/runs';
import { ModelConfig } from './src/interfaces/ModelConfig';
import { SimpleBookProcessor } from './src/services/simple-book.processor';

(async () => {
    const cfg: ModelConfig = {
        corpusFile: './books/Robert Sheckley - The Dream of Misunderstanding - 2002.txt',
        nEmbd: 256,
        nHidden: 128,
        nCtx: 16,
        iterations: 1
    };

    const model = new SimpleBookProcessor(cfg);

    const batch = Math.round((new Date).getTime() / 1000);
    cfg.id = await Runs.createRun(cfg.corpusFile, cfg.nEmbd, cfg.nCtx, cfg.iterations, model.getCorpus().length, model.wte.length);

    model.saveWeights(batch + 'pre');
    const stat = await model.trainIterations(cfg);
    model.saveWeights(batch + 'post');
    model.fileService.writeFileSync(`./weights/${batch}-stat.json`, JSON.stringify(stat));
    model.generate('I saw her ')

    await Runs.finishRun(cfg.id);
    process.exit();

})();