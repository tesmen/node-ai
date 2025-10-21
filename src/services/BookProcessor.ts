import { ModelConfig } from '../interfaces/ModelConfig';
import { FileService } from './FileService';
import { MatrixGenerator } from './MatrixGenerator';
import { CharTokenizer } from './char.tokenizer';

type Matrix = [number[]]

export class BookProcessor {
    tokens: string[] = [];
    wte: Matrix;
    wpe: Matrix;
    generator: MatrixGenerator;
    private fileService: FileService;
    private tokenizer: CharTokenizer;
    private cfg: ModelConfig;
    private Wq: Matrix;
    private Wk: Matrix;
    private Wv: Matrix;
    private Wo: Matrix;
    private W1: Matrix;
    private W2: Matrix;
    private Wout: Matrix;
    private ln1_g: Float32Array;
    private ln1_b: Float32Array;
    private ln2_g: Float32Array;
    private ln2_b: Float32Array;

    constructor(cfg: ModelConfig) {
        this.cfg = cfg;
        this.generator = new MatrixGenerator();
        this.fileService = new FileService();
        this.tokenizer = new CharTokenizer(this.getCorpus(cfg.corpusFile));
        // console.log(this.tokenizer.vocabSize);

        // Embeddings
        this.wte = this.zeros(this.tokenizer.vocabSize, cfg.nEmbd); // token embeddings
        this.wpe = this.zeros(cfg.nCtx, cfg.nEmbd);      // positional embeddings
        // Transformer block (single head for simplicity)
        this.Wq = zeros(cfg.nEmbd, cfg.nEmbd);
        this.Wk = zeros(cfg.nEmbd, cfg.nEmbd);
        this.Wv = zeros(cfg.nEmbd, cfg.nEmbd);
        this.Wo = zeros(cfg.nEmbd, cfg.nEmbd);

        // MLP
        this.W1 = zeros(cfg.nEmbd, cfg.nHidden);
        this.W2 = zeros(cfg.nHidden, cfg.nEmbd);

        // LayerNorm params (gamma/beta per channel)
        this.ln1_g = new Float32Array(cfg.nEmbd).fill(1);
        this.ln1_b = new Float32Array(cfg.nEmbd).fill(0);
        this.ln2_g = new Float32Array(cfg.nEmbd).fill(1);
        this.ln2_b = new Float32Array(cfg.nEmbd).fill(0);

        // Output head (tied to embeddings in many GPTs; here separate)
        this.Wout = zeros(cfg.nEmbd, this.tokenizer.vocabSize);

        // @ts-ignore
        [this.wte, this.wpe, this.Wq, this.Wk, this.Wv, this.Wo, this.W1, this.W2, this.Wout].forEach(this.initMat);
    }


    zeros(rows: number, cols: number): Matrix {
        const array = new Array(rows);

        for (let i = 0; i < rows; i++) {
            array[i] = new Float32Array(cols);
        }

        return array as Matrix;
    }

    initMat(M: Matrix) {
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

    prompt(prompt: string) {
        const ids = this.tokenizer.encode(prompt);
        const out = this.generate(ids, 120, 20); // generate 120 tokens with top-k 20

        return this.tokenizer.decode(out);
    }

    // Sample one token at a time (greedy or top-k)
    generate(promptIds: number[], maxNewTokens = 50, topK = 0) {
        let ids = promptIds.slice();

        for (let step = 0; step < maxNewTokens; step++) {
            console.log({ promptIds: ids });
            const ctx = ids.slice(-this.cfg.nCtx); // crop to context window
            const logits = this.forward(ctx);
            const last = logits[logits.length - 1];
            // optional top-k filter
            let probs;

            if (topK > 0) {
                const indexed = Array
                  .from(last)
                  .map((v, i) => [v, i])
                  .sort((a, b) => b[0] - a[0])
                  .slice(0, topK);

                const maxv = indexed[0][0];
                let sum = 0;
                probs = new Float32Array(this.cfg.vocabSize);
                for (const [v, i] of indexed) {
                    sum += Math.exp(v - maxv);
                }

                for (const [v, i] of indexed) {
                    probs[i] = Math.exp(v - maxv) / sum;
                }
            } else {
                // full softmax
                let maxv = -Infinity;
                for (const v of last) maxv = Math.max(maxv, v);
                let sum = 0;
                probs = new Float32Array(last.length);
                for (let i = 0; i < last.length; i++) {
                    const e = Math.exp(last[i] - maxv);
                    probs[i] = e;
                    sum += e;
                }
                for (let i = 0; i < probs.length; i++) probs[i] /= sum;
            }
            // sample
            const r = Math.random();
            let cum = 0, nextId = probs.length - 1;
            for (let i = 0; i < probs.length; i++) {
                cum += probs[i];
                if (r <= cum) {
                    nextId = i;
                    break;
                }
            }
            ids.push(nextId);
            if (ids.length >= this.cfg.nCtx) ids = ids.slice(-this.cfg.nCtx);
        }
        return ids;
    }

    forward(inputIds: number[]) {
        const { nEmbd } = this.cfg;
        const inputLength = inputIds.length; // sequence length
console.log(' sequence length', inputIds.length)
        // --- Embedding lookup + positional add
        let X = zeros(inputLength, nEmbd);

        for (let t = 0; t < inputLength; t++) {
            console.log('inputIds[t]', inputIds[t] );
            const tok = this.wte[inputIds[t]];

            console.log(t);
            const pos = this.wpe[t];
            for (let j = 0; j < nEmbd; j++) {
                X[t][j] = tok[j] + pos[j];
            }
        }

        // --- Block 1: LN -> self-attn -> residual
        const normalizedX = layerNormRowwise(X, 1e-5, this.ln1_g, this.ln1_b);

        // Project to Q,K,V
        const Q = matmul(normalizedX, this.Wq); // [T, C]
        const K = matmul(normalizedX, this.Wk);
        const V = matmul(normalizedX, this.Wv);

        // Scaled dot-product attention with causal mask
        // scores = Q K^T / sqrt(C)
        const C = Q[0].length;
        const scale = 1 / Math.sqrt(C);
        const scores = this.zeros(inputLength, inputLength);
        for (let i = 0; i < inputLength; i++) {
            for (let j = 0; j <= i; j++) { // causal: j <= i
                let dot = 0;
                for (let c = 0; c < C; c++) dot += Q[i][c] * K[j][c];
                scores[i][j] = dot * scale;
            }
            for (let j = i + 1; j < inputLength; j++) scores[i][j] = -1e9; // mask future
        }

        const attn = softmaxRowwise(scores); // [T, T]

        // context = attn * V
        const context = zeros(inputLength, C);
        for (let i = 0; i < inputLength; i++) {
            for (let j = 0; j < inputLength; j++) {
                const a = attn[i][j];
                for (let c = 0; c < C; c++) context[i][c] += a * V[j][c];
            }
        }

        // output proj
        const attnOut = matmul(context, this.Wo);

        // residual 1
        const H = zeros(inputLength, C);
        for (let i = 0; i < inputLength; i++) for (let c = 0; c < C; c++) H[i][c] = X[i][c] + attnOut[i][c];

        // --- Block 1: LN -> MLP -> residual
        const Hn = layerNormRowwise(H, 1e-5, this.ln2_g, this.ln2_b);
        const M1 = matmul(Hn, this.W1);
        const M2 = geluRowwise(M1);
        const M3 = matmul(M2, this.W2);

        const Y = zeros(inputLength, C);
        for (let i = 0; i < inputLength; i++) for (let c = 0; c < C; c++) Y[i][c] = H[i][c] + M3[i][c];

        // logits = Y * Wout
        const logits = matmul(Y, this.Wout); // [T, vocab]
        return logits;
    }
}

function matmul(A: Matrix, B: Matrix) {
    // A: [m,k], B: [k,n] -> [m,n]
    const m = A.length, k = A[0].length, n = B[0].length;
    const C = zeros(m, n);
    for (let i = 0; i < m; i++) {
        for (let p = 0; p < k; p++) {
            const a = A[i][p];
            for (let j = 0; j < n; j++) {
                C[i][j] += a * B[p][j];
            }
        }
    }
    return C;
}

function zeros(rows: number, cols: number): Matrix {
    const a = new Array(rows);

    for (let i = 0; i < rows; i++) {
        a[i] = new Float32Array(cols);
    }

    return a as Matrix;
}

function layerNormRowwise(X: Matrix, eps = 1e-5, gamma: any = null, beta: any = null) {
    const m = X.length, n = X[0].length;
    const out = zeros(m, n);
    for (let i = 0; i < m; i++) {
        let mean = 0;
        for (let j = 0; j < n; j++) mean += X[i][j];
        mean /= n;
        let varSum = 0;
        for (let j = 0; j < n; j++) {
            const d = X[i][j] - mean;
            varSum += d * d;
        }
        const invStd = 1 / Math.sqrt(varSum / n + eps);
        for (let j = 0; j < n; j++) {
            let v = (X[i][j] - mean) * invStd;
            if (gamma) v *= gamma[j];
            if (beta) v += beta[j];
            out[i][j] = v;
        }
    }
    return out;
}

function geluRowwise(X: Matrix) {
    const m = X.length, n = X[0].length;
    const out = zeros(m, n);
    // tanh approximation of GELU
    for (let i = 0; i < m; i++) {
        for (let j = 0; j < n; j++) {
            const x = X[i][j];
            const c = Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x));
            out[i][j] = 0.5 * x * (1 + c);
        }
    }
    return out;
}


function softmaxRowwise(X: Matrix) {
    const m = X.length, n = X[0].length;
    const out = zeros(m, n);
    for (let i = 0; i < m; i++) {
        let maxv = -Infinity;
        for (let j = 0; j < n; j++) maxv = Math.max(maxv, X[i][j]);
        let sum = 0;
        for (let j = 0; j < n; j++) sum += Math.exp(X[i][j] - maxv);
        const inv = 1 / sum;
        for (let j = 0; j < n; j++) out[i][j] = Math.exp(X[i][j] - maxv) * inv;
    }
    return out;
}