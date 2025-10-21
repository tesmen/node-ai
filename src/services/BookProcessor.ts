import { ModelConfig } from '../interfaces/ModelConfig';
import { FileService } from './FileService';
import { MatrixGenerator } from './MatrixGenerator';
import { CharTokenizer } from './char.tokenizer';

export class BookProcessor {
    tokens: string[] = [];
    wte: any[] = [];
    wpe: any[] = [];
    generator: MatrixGenerator;
    private fileService: FileService;
    private tokenizer: CharTokenizer;

    constructor(cfg: ModelConfig) {
        this.generator = new MatrixGenerator();
        this.fileService = new FileService();
        this.tokenizer = new CharTokenizer(this.getCorpus(cfg.corpusFile));
        console.log(this.tokenizer.vocabSize);

        // Embeddings
        this.wte = this.zeros(this.tokenizer.vocabSize, cfg.nEmbd); // token embeddings
        this.wpe = this.zeros(cfg.nCtx, cfg.nEmbd);      // positional embeddings

        [this.wte, this.wpe,
            // this.Wq, this.Wk, this.Wv, this.Wo, this.W1, this.W2, this.Wout
        ].forEach(this.initMat);
    }

    async trainTheBook() {
        console.log(this.wte[0]);
    }

    zeros(rows: number, cols: number) {
        const array = new Array(rows);

        for (let i = 0; i < rows; i++) {
            array[i] = new Float32Array(cols);
        }

        return array;
    }

    initMat(M: [number[]]) {
        const magnitude: number = 0.2;
        for (let i = 0; i < M.length; i++) {

            for (let j = 0; j < M[i].length; j++) {
                M[i][j] = magnitude - Math.random() * magnitude;
            }
        }
    };

    getCorpus(name: string): string {
        const content = this.fileService.readFileSync(name);
        return content.toString();
    }

    generate(prompt: string) {
        const ids = this.tokenizer.encode(prompt);
        return '';
    }
}
