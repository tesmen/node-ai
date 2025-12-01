import { ModelEntity } from '../database/model.entity';
import { ResultEntity } from '../database/result.entity';
import { SessionEntity } from '../database/session.entity';
import { adjustEmbeddings, calculateErrorVector, layerNorm1D } from '../fns';
import { FileServiceAdapter } from '../services/file-service.adapter';
import { SimpleModel } from '../services/simple.model';
import { TrainingSessionConfig } from './training-session.config';

export class TrainingSession {
    cfg: TrainingSessionConfig;
    corpusArray: string[];
    corpus: string;
    model: SimpleModel;

    constructor(config: TrainingSessionConfig, model: SimpleModel) {
        this.cfg = config;
        this.model = model;
        this.corpus = FileServiceAdapter.getTextContent(this.model.cfg.source);
        this.corpusArray = this.model.tokenizer.separate(this.corpus);
    }

    async run(): Promise<void> {
        let round = { error: 0, correct: 0, ratio: 0 };

        for (let iteration = 0; iteration < this.cfg.iterations; iteration++) {
            const windowSize = this.model.cfg.nCtx;
            const { error, correct, shift } = this.train(windowSize, iteration);
            round.correct = correct;
            round.error = error;
            round.ratio = Number((round.correct / (round.error + round.correct)).toFixed(3)) || 0;
            this.log('>>> Iteration finished ', { iteration, round });

            await ResultEntity.create({
                  session_id: this.cfg.id,
                  error,
                  correct,
                  iteration: iteration,
              }
            );
        }

        await SessionEntity.finishRun(this.cfg.id, { correct_ratio: round.ratio });
        await ModelEntity.save(this.model)
    }

    // a single run over provided corpus file
    private train(windowSize: number, iteration: number): { error: number; correct: number; shift: number } {
        let sampleArray: string[];
        let step = 0;
        let error = 0;
        let correct = 0;
        let shift;

        if (this.cfg.use_slide) {
            shift = this.corpusArray.length < windowSize
              ? iteration % this.corpusArray.length
              : iteration % windowSize;
        } else {
            shift = 0;
        }

        while ((sampleArray = this.corpusArray.slice(shift + step * windowSize, shift + (step + 1) * windowSize)).length) {
            step++;

            for (let i = 1; i < sampleArray.length; i++) {
                const promptTokens = sampleArray.slice(0, i);
                const prompt = promptTokens.join(' ');
                const ids = this.model.tokenizer.encode(prompt);

                const logits = this.model.forward(ids);
                const expectedTokenId = this.model.tokenizer.encodeOne(sampleArray[i]);
                const logit = logits[0];

                if (expectedTokenId !== logit) {
                    error++;
                    const promptVector = this.model.createPromptVector(ids);
                    const adjusted = adjustEmbeddings(promptVector, this.model.embed(expectedTokenId));
                    this.model.wte[expectedTokenId] = adjusted.newTarget;

                    if (this.cfg.adjust_pte) {
                        const errorVec = calculateErrorVector(promptVector, this.model.embed(expectedTokenId));
                        const scale = 0.05 * (1 / promptTokens.length);

                        for (let pos = 0; pos < promptTokens.length; pos++) {
                            const pe = this.model.wpe[pos];
                            for (let d = 0; d < errorVec.length; d++) {
                                pe[d] += scale * errorVec[d];
                            }
                            // optional: renormalize or clamp here if you want
                            this.model.wpe[pos] = layerNorm1D(pe);
                        }
                    }
                } else {
                    correct++;
                }

                // this.log('training on:',
                //   {
                //       sampleArray: sampleArray.join(' '),
                //       shift,
                //       prompt,
                //       expected: sampleArray[i],
                //       logitText: this.tokenizer.decodeOne(logit),
                //       logits: logits.slice(0, 10),
                //       logit: logits[0],
                //       correct: expectedTokenId === logit,
                //   }
                // );
            }
        }

        return { error, correct, shift };
    }


    private log(...msg: any) {
        console.log(msg);
    }
}