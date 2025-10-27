/**
 * @deprecated
 */
export class MatrixGenerator {

    generate(dim: number, magnitude: number): number[] {
        const res: number[] = [];

        for (let i = 0; i < dim; i++) {
            res.push(magnitude - Math.random() * magnitude);
        }

        return res;
    }
}