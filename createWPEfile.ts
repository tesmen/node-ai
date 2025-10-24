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
// const zeros = model.zeros(3, 3);
// console.log(zeros);
// model.initMat(zeros);
// console.log(zeros);
// console.log(JSON.stringify(zeros));

// console.log(
//   JSON.stringify(model.wpe)
// );

model.fileService.writeFileSync('./weights/wte.json', JSON.stringify(model.wpe));