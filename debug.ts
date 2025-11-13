import { Runs } from './src/database/runs';
import { ModelConfig } from './src/interfaces/ModelConfig';
import { SimpleBookProcessor } from './src/services/simple-book.processor';

const useDbStorage = false;

(async () => {
    const cfg: ModelConfig = {
        // corpusFile: './books/Robert Sheckley - The Dream of Misunderstanding - 2002.txt',
        // corpusFile: './books/candp.nano.txt',
        corpusFile: './books/candp.min.txt',
        // corpusFile: './books/candp.txt',
        nEmbd: 64,
        nHidden: 128,
        nCtx: 64,
        iterations: 400,
        trainWindow: 10
    };

    const model = new SimpleBookProcessor(cfg);
// console.log(model.tokenizer.stoi)
//     process.exit()
    const batch = Math.round((new Date).getTime() / 1000);
    cfg.id = await Runs.createRun(cfg.corpusFile, cfg.nEmbd, cfg.nCtx, cfg.iterations, model.getCorpus().length, model.wte.length);

    // model.saveWeights(batch + 'pre');
    const stat = await model.trainIterations(cfg);
    // model.saveWeights(batch + 'post');
    model.fileService.writeFileSync(`./weights/${batch}-stat.json`, JSON.stringify(stat));
    const a = model.generate('actions had stirred');
    console.log(a);
    const lastStat = stat.pop();
    const ratio = lastStat.correct / (lastStat.error + lastStat.correct);
    await Runs.finishRun(cfg.id, { correct_ratio: ratio });
    process.exit();

})();