import { describe, expect, test } from '@jest/globals';
import { SimpleBookProcessor } from '../src/services/simple-book.processor';

const sum = (a: number, b: number) => a + b;

const config = {
    corpusFile: './books/candp.min.txt',
    wpeFile: './weights/wpe.json',
    wteFile: './weights/wte.json',
    nEmbd: 16,
    nHidden: 128,
    nCtx: 64
};
const model = new SimpleBookProcessor(config);
describe('Vectors module', () => {

    test('Test for test', () => {
        expect(sum(1, 2)).toBe(3);
    });

    test('Adjust the target vectors', () => {
        expect(
          model.adjustEmbeddings(
            [1, -1], // prompt vec
            [0, 0], // target vec
            0.05)
            .newTarget
        ).toStrictEqual([0.05, -0.05]);
    });

    test('Adjust the prompt vectors', () => {
        expect(
          model.adjustEmbeddings(
            [1, -1], // prompt vec
            [0, 0], // target vec
            0.05
          ).newPrompt
        ).toStrictEqual([0.95, -0.95]);
    });

    test('Adjust the target vectors NEG', () => {
        expect(
          model.adjustEmbeddings(
            [0, 0], // prompt vec
            [1, -1], // target vec
            0.05
          ).newTarget
        ).toStrictEqual([0.95, -0.95]);
    });

    test('Adjust the prompt vectors NEG', () => {
        expect(
          model.adjustEmbeddings(
            [0, 0], // prompt vec
            [1, -1], // target vec
            0.05
          ).newPrompt
        ).toStrictEqual([0.05, -0.05]);
    });

});