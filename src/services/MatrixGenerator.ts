export class MatrixGenerator {

    generate(dim: number): number[] {
        const res: number[] = [];

        for (let i = 0; i < dim; i++) {
            res.push(Math.random());
        }

        return res;
    }
}