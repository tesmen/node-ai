import { Vector } from '../types';

export class SmartVector extends Array {
    constructor(...items: any[]) {
        super(...items);
    }

    add(v: Vector) {
        if (this.length != v.length) {
            throw Error('this.length != v.length');
        }

        this.forEach((el, i) => this[i] += v[i]);

        return this;
    }
}
