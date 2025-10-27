// Tiny GPT-like Transformer in pure JavaScript (Node.js)
// -----------------------------------------------------
// Goal: show how an LLM-style model can be built in a high-level language (JS) without external deps.
// This is a *toy*, focused on clarity over speed. It implements:
//  - Char-level tokenizer
//  - Token & positional embeddings
//  - 1 Transformer block: LayerNorm -> Self-Attention (single head) -> MLP -> residuals
//  - Causal masking for autoregressive generation
//  - Text sampling
// No training loop is included (backprop would make this file much longer). You can still
// generate from random weights, and wire in your own weights if you have them.
//
// Usage:
//   node tiny_gpt.js  (if you save this file as tiny_gpt.js)
//
// Optional tweaks at the bottom under `demo()`.

// ---------------------------
// Utilities
// ---------------------------
// import * as fs from 'node:fs'
const fs = require('fs')

function seedRandom(seed = 1337) {
    // Simple LCG for reproducible randomness
    let s = seed >>> 0
    return () => (s = (s * 1664525 + 1013904223) >>> 0) / 0x100000000
}

function randn(rng) {
    // Box–Muller transform for standard normal
    let u = 0, v = 0
    while(u === 0) u = rng()
    while(v === 0) v = rng()
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v)
}

function zeros(rows, cols) {
    const a = new Array(rows)
    for(let i = 0; i < rows; i++) a[i] = new Float32Array(cols)
    return a
}

/**
 * @param {Matrix} A
 * @param {Matrix} B
 * @returns {any[]}
 */
function matmul(A, B) {
    // A: [m,k], B: [k,n] -> [m,n]
    const m = A.length, k = A[0].length, n = B[0].length
    const C = zeros(m, n)
    for(let i = 0; i < m; i++) {
        for(let p = 0; p < k; p++) {
            const a = A[i][p]
            for(let j = 0; j < n; j++) {
                C[i][j] += a * B[p][j]
            }
        }
    }
    return C
}

function addInPlace(A, B) {
    for(let i = 0; i < A.length; i++) {
        for(let j = 0; j < A[i].length; j++) A[i][j] += B[i][j]
    }
}

function softmaxRowwise(X) {
    const m = X.length, n = X[0].length
    const out = zeros(m, n)
    for(let i = 0; i < m; i++) {
        let maxv = -Infinity
        for(let j = 0; j < n; j++) {
            maxv = Math.max(maxv, X[i][j])
        }

        let sum = 0
        for(let j = 0; j < n; j++) {
            sum += Math.exp(X[i][j] - maxv)
        }

        const inv = 1 / sum
        for(let j = 0; j < n; j++) {
            out[i][j] = Math.exp(X[i][j] - maxv) * inv
        }
    }
    return out
}

function geluRowwise(X) {
    const m = X.length, n = X[0].length
    const out = zeros(m, n)
    // tanh approximation of GELU
    for(let i = 0; i < m; i++) {
        for(let j = 0; j < n; j++) {
            const x = X[i][j]
            const c = Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x))
            out[i][j] = 0.5 * x * (1 + c)
        }
    }
    return out
}

/**
 * @param {Matrix} X
 * @param eps
 * @param {Vector} gamma
 * @param {Vector} beta
 * @returns {any[]}
 */
function layerNormRowwise(X, eps = 1e-5, gamma = null, beta = null) {
    const m = X.length, n = X[0].length
    const out = zeros(m, n)
    for(let i = 0; i < m; i++) {
        let mean = 0
        for(let j = 0; j < n; j++) mean += X[i][j]
        mean /= n
        let varSum = 0
        for(let j = 0; j < n; j++) {
            const d = X[i][j] - mean
            varSum += d * d
        }
        const invStd = 1 / Math.sqrt(varSum / n + eps)
        for(let j = 0; j < n; j++) {
            let v = (X[i][j] - mean) * invStd
            if(gamma) v *= gamma[j]
            if(beta) v += beta[j]
            out[i][j] = v
        }
    }
    return out
}

// ---------------------------
// Tokenizer (char-level)
// ---------------------------
class CharTokenizer {
    constructor(text) {
        const set = new Set(text.split(' '))
        this.itos = Array.from(set).sort()
        this.stoi = new Map(this.itos.map((ch, i) => [ ch, i ]))
        this.vocabSize = this.itos.length
    }

    encode(s) {
        return s.split(' ').map(ch => this.stoi.get(ch) ?? 0)
    }

    decode(ids) {
        return ids.map(i => this.itos[i] ?? '').join(' ')
    }
}

// ---------------------------
// Parameters & Model
// ---------------------------
class TinyGPT {
    constructor(cfg, rng = seedRandom(42)) {
        this.cfg = cfg
        const { vocabSize, nEmbd, nCtx } = cfg

        // Embeddings
        this.wte = zeros(vocabSize, nEmbd) // token embeddings
        this.wpe = zeros(nCtx, nEmbd)      // positional embeddings

        // Transformer block (single head for simplicity)
        this.Wq = zeros(nEmbd, nEmbd)
        this.Wk = zeros(nEmbd, nEmbd)
        this.Wv = zeros(nEmbd, nEmbd)
        this.Wo = zeros(nEmbd, nEmbd)

        // MLP
        this.W1 = zeros(nEmbd, cfg.nHidden)
        this.W2 = zeros(cfg.nHidden, nEmbd)

        // LayerNorm params (gamma/beta per channel)
        this.ln1_g = new Float32Array(nEmbd).fill(1)
        this.ln1_b = new Float32Array(nEmbd).fill(0)
        this.ln2_g = new Float32Array(nEmbd).fill(1)
        this.ln2_b = new Float32Array(nEmbd).fill(0)

        // Output head (tied to embeddings in many GPTs; here separate)
        this.Wout = zeros(nEmbd, vocabSize)

        // init
        const std = 0.02
        const initMat = M => {
            for(let i = 0; i < M.length; i++) {
                for(let j = 0; j < M[i].length; j++) M[i][j] = randn(rng) * std
            }
        };
        [ this.wte, this.wpe, this.Wq, this.Wk, this.Wv, this.Wo, this.W1, this.W2, this.Wout ].forEach(initMat)
    }

    forward(inputIds) {
        const { nEmbd } = this.cfg
        const T = inputIds.length // sequence length

        // --- Embedding lookup + positional add
        let X = zeros(T, nEmbd)
        for(let t = 0; t < T; t++) {
            const tok = this.wte[inputIds[t]]
            const pos = this.wpe[t]
            for(let j = 0; j < nEmbd; j++) {
                X[t][j] = tok[j] + pos[j]
            }
        }

        // --- Block 1: LN -> self-attn -> residual
        const normalizedX = layerNormRowwise(X, 1e-5, this.ln1_g, this.ln1_b)

        // Project to Q,K,V
        const Q = matmul(normalizedX, this.Wq) // [T, C]
        const K = matmul(normalizedX, this.Wk)
        const V = matmul(normalizedX, this.Wv)

        // Scaled dot-product attention with causal mask
        // scores = Q K^T / sqrt(C)
        const C = Q[0].length
        const scale = 1 / Math.sqrt(C)
        const scores = zeros(T, T)
        for(let i = 0; i < T; i++) {
            for(let j = 0; j <= i; j++) { // causal: j <= i
                let dot = 0

                for(let c = 0; c < C; c++) {
                    dot += Q[i][c] * K[j][c]
                }

                scores[i][j] = dot * scale
            }
            for(let j = i + 1; j < T; j++) scores[i][j] = -1e9 // mask future
        }

        const attn = softmaxRowwise(scores) // [T, T]

        // context = attn * V
        const context = zeros(T, C)
        for(let i = 0; i < T; i++) {
            for(let j = 0; j < T; j++) {
                const a = attn[i][j]
                for(let c = 0; c < C; c++) context[i][c] += a * V[j][c]
            }
        }

        // output proj
        const attnOut = matmul(context, this.Wo)

        // residual 1
        const H = zeros(T, C)
        for(let i = 0; i < T; i++) for(let c = 0; c < C; c++) H[i][c] = X[i][c] + attnOut[i][c]

        // --- Block 1: LN -> MLP -> residual
        const Hn = layerNormRowwise(H, 1e-5, this.ln2_g, this.ln2_b)
        const M1 = matmul(Hn, this.W1)
        const M2 = geluRowwise(M1)
        const M3 = matmul(M2, this.W2)

        const Y = zeros(T, C)
        for(let i = 0; i < T; i++) for(let c = 0; c < C; c++) Y[i][c] = H[i][c] + M3[i][c]

        // logits = Y * Wout
        const logits = matmul(Y, this.Wout) // [T, vocab]
        return logits
    }

    // Sample one token at a time (greedy or top-k)
    generate(promptIds, maxNewTokens = 50, topK = 0) {
        let ids = promptIds.slice()
        console.log({ ids })

        for(let step = 0; step < maxNewTokens; step++) {
            const ctx = ids.slice(-this.cfg.nCtx) // crop to context window
            const logits = this.forward(ctx)
            const last = logits[logits.length - 1]
            // optional top-k filter
            let probs

            if(topK > 0) {
                const indexed = Array.from(last).map((v, i) => [ v, i ])
                    .sort((a, b) => b[0] - a[0])
                    .slice(0, topK)

                const maxv = indexed[0][0]
                let sum = 0
                probs = new Float32Array(this.cfg.vocabSize)
                for(const [ v, i ] of indexed) {
                    sum += Math.exp(v - maxv)
                }

                for(const [ v, i ] of indexed) {
                    probs[i] = Math.exp(v - maxv) / sum
                }
            } else {
                // full softmax
                let maxv = -Infinity
                for(const v of last) maxv = Math.max(maxv, v)
                let sum = 0
                probs = new Float32Array(last.length)
                for(let i = 0; i < last.length; i++) {
                    const e = Math.exp(last[i] - maxv)
                    probs[i] = e
                    sum += e
                }
                for(let i = 0; i < probs.length; i++) probs[i] /= sum
            }
            // sample
            const r = Math.random()
            let cum = 0, nextId = probs.length - 1
            for(let i = 0; i < probs.length; i++) {
                cum += probs[i]
                if(r <= cum) {
                    nextId = i
                    break
                }
            }
            ids.push(nextId)
            if(ids.length >= this.cfg.nCtx) ids = ids.slice(-this.cfg.nCtx)
        }
        return ids
    }
}

// ---------------------------
// Demo
// ---------------------------
function demo() {
    // Tiny corpus just to build a tokenizer
    const corpusOld = 'the cat sat on the mat\n' +
        'the dog sat on the rug\n' +
        'gpt in js\n'

    const corpus = getCorpus()
    const tokenizer = new CharTokenizer(corpus)
    const cfg = { vocabSize: tokenizer.vocabSize, nEmbd: 64, nHidden: 128, nCtx: 64 }
    const model = new TinyGPT(cfg)

    const prompt = 'that you have room on your small world '
    const ids = tokenizer.encode(prompt)
    const out = model.generate(ids, 120, 20) // generate 120 tokens with top-k 20
    const text = tokenizer.decode(out)

    console.log('Vocab size:', tokenizer.vocabSize)
    console.log('Prompt:', JSON.stringify(prompt))
    console.log('Generated (random weights, will be gibberish):\n')
    console.log(text)

    console.log('\nTip: To make this produce sensible text, you need to *train* the weights.\n' +
        'You can implement backprop for each op (matmul, layernorm, attention, GELU)\n' +
        'or port weights from a trained tiny model. This scaffold handles the forward pass and sampling.')
}

function getCorpus() {
    const list = fs
        .readdirSync('./books')
        .filter(str => str.endsWith('.txt'))

    let text = ''

    for(const textKey of list) {
        const cleaned = fs
            .readFileSync(`./books/${textKey}`)
            .toString()
            .replaceAll(/\W/g , ' ')
            .toLowerCase()
        // console.log(cleaned)
        // process.exit()
        text += cleaned
    }

    console.log(`Corpus length: ${text.length}`)
    return text
}

if(require.main === module) {
    demo()
}
