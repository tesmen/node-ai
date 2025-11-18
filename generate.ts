import { ModelEntity } from './src/database/model.entity';
import { SessionEntity } from './src/database/session.entity';
import { ModelConfig } from './src/interfaces/ModelConfig';
import { TrainingSessionConfig } from './src/models/training-session.config';
import { TrainingSession } from './src/models/training.session';
import { SimpleBookProcessor } from './src/services/simple-book.processor';

(async () => {
    const model = await ModelEntity.load(6);
    try {
        const response = model.generate('The');
        console.log('response', response);
    } catch (e) {
        console.log(e);
    }

    process.exit();
})();

