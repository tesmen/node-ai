import { SimpleBookProcessor } from './src/services/simple-book.processor';

(async () => {
    const model = new SimpleBookProcessor(
      {
          corpusFile: './books/candp.min.txt',
          nEmbd: 4096,
          nHidden: 128,
          nCtx: 64
      }
    );

    model.saveWeights('pre');
    const stat = model.trainIterations(500, 10);
    model.saveWeights('post');

    model.fileService.writeFileSync('stat.json', JSON.stringify(stat));

})();