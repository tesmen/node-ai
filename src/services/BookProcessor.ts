import { FileService } from './FileService';
import { MatrixGenerator } from './MatrixGenerator';
import { SimpleTokenizer } from './SimpleTokenizer';

export class BookProcessor {
    meaningEmbeddings: Record<string, number[]> = {};
    positionEmbedding = {};
    generator: MatrixGenerator;
    private fileService: FileService;
    private tokenizer: SimpleTokenizer;

    constructor() {
        this.generator = new MatrixGenerator();
        this.fileService = new FileService();
        this.tokenizer = new SimpleTokenizer();
    }

    async trainTheBook(name: string) {
        const content = await this.fileService.readFile(name);
        const tokens = this.tokenizer.tokenize(content.toString());
        console.log('tokens count', tokens.length);

        tokens.map(token => {
            this.meaningEmbeddings[token] = this.generator.generate(768);
        });

    }
}
