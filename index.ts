import { BookProcessor } from './src/services/BookProcessor';

(async () => {
    const model = new BookProcessor(
      {
          corpusFile: './books1/0 - Asimov, Isaac - Foundation Trilogy.txt',
          nEmbd: 64,
          nHidden: 128,
          nCtx: 64
      }
    );
    try {
        model.prompt('small town');
    } catch (e) {
        console.log(e);
    }
})();