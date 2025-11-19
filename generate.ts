import { ModelEntity } from './src/database/model.entity';

(async () => {
    const model = await ModelEntity.load(3);
    console.log('Loaded model', model.id)
    try {
        const response = model.generate('The');
        console.log('response', response);
    } catch (e) {
        console.log(e);
    }

    process.exit();
})();

