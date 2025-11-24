import { layerNorm1D, layerNormRowwise, normalizeVectorL2 } from './src/fns';

const a = [
    // [5, 5, 5, 5, 5],
    [1, 2, 3, 4, 5],
    // [.1, .2, .3, .4, .5],
];

console.log(layerNormRowwise(a));
console.log(layerNorm1D(a[0]));
// console.log(layerNorm1D(a[1]));
// console.log(layerNorm1D(a[2]));
