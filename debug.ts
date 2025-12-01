import { SessionEntity } from './src/database/session.entity';
import { ModelConfig } from './src/interfaces/ModelConfig';
import { SimpleModel } from './src/services/simple.model';

(async () => {
    const cfg: ModelConfig = {
        // corpusFile: './books/Robert Sheckley - The Dream of Misunderstanding - 2002.txt',
        // corpusFile: './books/candp.nano.txt',
        source: './books/candp.min.txt',
        // corpusFile: './books/candp.med.txt',
        // corpusFile: './books/candp.txt',
        nEmbd: 64,
        nHidden: 128,
        nCtx: 64,
        // useSlide: true,
    };

    const model = new SimpleModel(cfg);

    cfg.id = await SessionEntity.create(
      {
          iterations: 100,
      }
    );

    const a = model.generate('The', 20);
    console.log(a);
    process.exit();

})();
