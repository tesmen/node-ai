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
    try {
        model.generate('most other parts');
    } catch (e) {
        console.log(e);
    }
})();