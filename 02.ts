import { layerNorm1D, layerNormRowwise, normalizeVectorL2, calculateErrorVector } from './src/fns';

const a = [
    // [5, 5, 5, 5, 5],
    [1, 2, 3, 4, 5],
    // [.1, .2, .3, .4, .5],
];
console.log(calculateErrorVector([1, 2, 0, 4, 5], [1, 2, 3, 4, 5]));