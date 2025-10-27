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

// const normalized = model.normalizeVector([0.1, 1]);
// const normalized = model.normalizeVector([3, 4]);
// console.log(normalized);
// const projection = normalized.reduce((acc, num) => acc + num, 0);
// console.log({ projection });
// console.log(Math.sqrt(normalized.reduce((acc, num) => acc + num * num, 0)));

const mat = model.zeros(10,10);
model.initNormalizedMat(mat);
console.log(mat);
console.log('--')
console.log(mat.map(v => Math.sqrt(v.reduce((acc, num) => acc + num * num, 0))));