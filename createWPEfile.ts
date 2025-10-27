import { SimpleBookProcessor } from './src/services/simple-book.processor';

const model = new SimpleBookProcessor(
  {
      corpusFile: './books/candp.min.txt',
      // corpusFile: './books/candp.txt',
      nEmbd: 16,
      nHidden: 128,
      nCtx: 64
  }
);

model.fileService.writeFileSync('./weights/wpe.json', JSON.stringify(model.wpe));
model.fileService.writeFileSync('./weights/wte.json', JSON.stringify(model.wte));