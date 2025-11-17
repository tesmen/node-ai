import { describe, expect, test } from '@jest/globals';
import { adjustEmbeddings } from '../src/fns';

const sum = (a: number, b: number) => a + b;

describe('Vectors module', () => {

    test('Test for test', () => {
        expect(sum(1, 2)).toBe(3);
    });

    test('Adjust zero prompt - target', () => {
        expect(
          adjustEmbeddings(
            [1, -1], // prompt vec
            [0, 0], // target vec
            0.05)
            .newTarget
        ).toStrictEqual([0.05, -0.05]);
    });

    test('Adjust zero prompt - prompt', () => {
        expect(
          adjustEmbeddings(
            [1, -1], // prompt vec
            [0, 0], // target vec
            0.05
          ).newPrompt
        ).toStrictEqual([0.95, -0.95]);
    });

    test('Adjust zero target - target', () => {
        expect(
          adjustEmbeddings(
            [0, 0], // prompt vec
            [1, -1], // target vec
            0.05
          ).newTarget
        ).toStrictEqual([0.95, -0.95]);
    });

    test('Adjust zero target - prompt', () => {
        expect(
          adjustEmbeddings(
            [0, 0], // prompt vec
            [1, -1], // target vec
            0.05
          ).newPrompt
        ).toStrictEqual([0.05, -0.05]);
    });

    test('Adjust edges - prompt', () => {
        expect(
          adjustEmbeddings(
            [1, 1], // prompt vec
            [-1, -1], // target vec
            0.05
          ).newPrompt
        ).toStrictEqual([0.9, 0.9]);
    });

    test('Adjust edges - target', () => {
        expect(
          adjustEmbeddings(
            [1, 1], // prompt vec
            [-1, -1], // target vec
            0.05
          ).newTarget
        ).toStrictEqual([-0.9, -0.9]);
    });

    test('Adjust positives - target', () => {
        expect(
          adjustEmbeddings(
            [1, 1], // prompt vec
            [0.5, 0.5], // target vec
            0.1
          ).newTarget
        ).toStrictEqual([0.55, 0.55]);
    });

    test('Adjust positives - prompt', () => {
        expect(
          adjustEmbeddings(
            [1, 1], // prompt vec
            [0.5, 0.5], // target vec
            0.1
          ).newPrompt
        ).toStrictEqual([0.95, 0.95]);
    });

    test('Adjust negatives - target', () => {
        expect(
          adjustEmbeddings(
            [-1, -1], // prompt vec
            [-0.5, -0.5], // target vec
            0.1
          ).newTarget
        ).toStrictEqual([-0.55, -0.55]);
    });

    test('Adjust negatives - prompt', () => {
        expect(
          adjustEmbeddings(
            [-1, -1], // prompt vec
            [-0.5, -0.5], // target vec
            0.1
          ).newPrompt
        ).toStrictEqual([-0.95, -0.95]);
    });

});
