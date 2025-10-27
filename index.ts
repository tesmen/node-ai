import { SimpleBookProcessor } from './src/services/simple-book.processor';

(async () => {
    const model = new SimpleBookProcessor(
      {
          corpusFile: './books/candp.min.txt',
          // corpusFile: './books/candp.txt',
          wpeFile: './weights/wpe.json',
          wteFile: './weights/wte.json',
          nEmbd: 16,
          nHidden: 128,
          nCtx: 64
      }
    );

    model.train(10)

    try {
        // const response = model.generate('most other');
        // console.log('response', response);
    } catch (e) {
        console.log(e);
    }
})();