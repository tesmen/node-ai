import { ModelEntity } from './src/database/model.entity';

(async () => {
    const model = await ModelEntity.load(14);
    console.log('Loaded model', model.cfg.source, model.id);
    try {
        const response = model.generate('T', 64);
        console.log('response', response);
    } catch (e) {
        console.error(e.message);
    }

    process.exit();
})();

