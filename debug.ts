import db from './db/knex';
import { SimpleBookProcessor } from './src/services/simple-book.processor';

(async () => {
    const model = new SimpleBookProcessor(
      {
          corpusFile: './books/candp.min.txt',
          nEmbd: 64,
          nHidden: 128,
          nCtx: 64
      }
    );
    const batch = Math.round((new Date).getTime() / 1000);
    model.saveWeights(batch + 'pre');
    const stat = await model.trainIterations(10, 64);
    model.saveWeights(batch + 'post');
    model.fileService.writeFileSync(`./weights/${batch}-stat.json`, JSON.stringify(stat));

    process.exit()
})();