export class CharTokenizer {

    skip = new Set(['\n', '\r']);
    itos: string[] = [];
    stoi: Map<string, number> = new Map;
    vocabSize: number;

    constructor(corpus: string) {
        const set = new Set(corpus
          .split(/(\W)/)
          .map(line => line.trim())
          .filter(element => element.length > 0)
          .filter(element => !this.skip.has(element))
        );

        this.itos = Array.from(set).sort();
        this.stoi = new Map(this.itos.map((ch, i) => [ch, i]));
        this.vocabSize = this.itos.length;
    }

    encode(input: string) {
        return input.split(' ').map(ch => this.stoi.get(ch) ?? 0);
    }

    decode(ids: number[]) {
        return ids
          .map(i => this.itos[i] ?? '<UT>')
          .join(' ');
    }
}