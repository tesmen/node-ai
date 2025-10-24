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
    try {
        const response = model.generate('most other parts');

        console.log(response);
    } catch (e) {
        console.log(e);
    }
})();