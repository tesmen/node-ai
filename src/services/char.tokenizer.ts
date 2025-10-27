export class CharTokenizer {

    skip = new Set(['\n', '\r']);
    itos: string[] = [];
    stoi: Map<string, number> = new Map;
    vocabSize: number;

    constructor(corpus: string) {
        const set = this.tokenize(corpus);

        this.itos = Array.from(set);
        // .sort();
        this.stoi = new Map(this.itos.map((char, index) => [char, index]));
        this.vocabSize = this.itos.length;
    }

    tokenize(corpus: string) {
        return new Set(corpus
          .split(/(\W)/)
          .map(line => line.trim())
          .filter(element => element.length > 0)
          .filter(element => !this.skip.has(element))
        );
    }

    encode(input: string) {
        return input.split(' ').map(ch => this.stoi.get(ch) ?? 0);
    }

    encodeOne(input: string) {
        return this.stoi.get(input);
    }

    decode(ids: number[]) {
        return ids
          .map(i => this.itos[i] ?? '<UT>')
          .join(' ');
    }

    decodeOne(id: number) {
        return this.itos[id] ?? '<UT>';
    }
}