import { SimpleBookProcessor } from './src/services/simple-book.processor';

(async () => {
    const model = new SimpleBookProcessor(
      {
          corpusFile: './books/candp.min.txt',
          wpeFile: './weights/wpe.json',
          wteFile: './weights/wte.json',
          nEmbd: 16,
          nHidden: 128,
          nCtx: 64
      }
    );

    model.saveWeights('pre')
    model.trainIterations(100, 10)
    model.saveWeights('post')

})();