export class SimpleTokenizer {

    skip = new Set(['\n', '\r']);

    tokenize(input: string) {
        return input
          .split(/(\W)/)
          .map(line => line.trim())
          .filter(element => element.length > 0)
          .filter(element => !this.skip.has(element))
          ;
    }
}