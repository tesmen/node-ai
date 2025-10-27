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

const a = [0.07, 0.14, 0.21, 0.28, 0.35, 0.42, 0.49, 0.56];
const b = [0.56, 0.49, 0.42, 0.35, 0.28, 0.21, 0.14, 0.07];

// console.log(model.normalizeVector(a).map(num => Math.round(num * 100, 2) / 100));
// console.log(model.normalizeVector(b).map(num => Math.round(num * 100, 2) / 100));

console.log(model.dot(a, b));