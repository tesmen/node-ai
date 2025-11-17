import { Result } from '../database/result';
import { Runs } from '../database/runs';
import { ModelConfig } from '../interfaces/ModelConfig';
import { CharTokenizer } from '../services/char.tokenizer';
import { FileServiceAdapter } from '../services/file-service.adapter';
import { TrainingSessionConfig } from './training-session.config';

export class TrainingSession {
    cfg: TrainingSessionConfig;
    tokenizer: CharTokenizer;
    corpusArray: string[];
    corpus: string;

    constructor(config: TrainingSessionConfig) {
        this.cfg = config;
        this.tokenizer = new CharTokenizer();
        this.corpus = FileServiceAdapter.getTextContent(config.corpusFile);
        this.tokenizer.init(this.corpus);
        this.corpusArray = this.tokenizer.separate(this.corpus);


    }

    async run(config: ModelConfig): Promise<void> {
        let round = { error: 0, correct: 0, ratio: 0 };

        for (let iteration = 0; iteration < config.iterations; iteration++) {
            const windowSize = config.trainWindow || config.nCtx;
            const { error, correct, shift } = this.train(windowSize, iteration);
            round.correct = correct;
            round.error = error;
            round.ratio = Number((round.correct / (round.error + round.correct)).toFixed(3)) || 0;
            this.log('>>> Iteration finished ', { iteration, round });

            await Result.create({
                  run_id: config.id,
                  error,
                  correct,
                  iteration: iteration,
              }
            );
        }

        await Runs.finishRun(this.cfg.id, { correct_ratio: round.ratio });
    }

    // a single run over provided corpus file
    train(windowSize: number, iteration: number): { error: number; correct: number; shift: number } {
        let sampleArray: string[];
        let step = 0;
        let error = 0;
        let correct = 0;
        let shift;

        if (this.cfg.useSlide) {
            shift = this.corpusArray.length < windowSize
              ? iteration % this.corpusArray.length
              : iteration % windowSize;
        } else {
            shift = 0;
        }

        while ((sampleArray = this.corpusArray.slice(shift + step * windowSize, shift + (step + 1) * windowSize)).length) {
            step++;

            for (let i = 1; i < sampleArray.length; i++) {
                const prompt = sampleArray.slice(0, i).join(' ');
                const ids = this.tokenizer.encode(prompt);

                const logits = this.forward(ids);
                const expectedTokenId = this.tokenizer.encodeOne(sampleArray[i]);
                const logit = logits[0];

                if (expectedTokenId !== logit) {
                    const promptVector = this.createPromptVector(ids);
                    const adjusted = this.adjustEmbeddings(promptVector, this.embed(expectedTokenId));
                    this.wte[expectedTokenId] = adjusted.newTarget;
                    // this.wte[logit  ] = adjusted.newTarget;
                    // this.log('adjusted.', JSON.stringify(adjusted.newTarget));
                    // this.log('adjusted.oldTarget', JSON.stringify(this.embed(expectedTokenId)));
                    error++;
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