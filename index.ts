import { BookProcessor } from './src/services/BookProcessor';

(async () => {
    try {
        await (new BookProcessor()).trainTheBook('./books1/0 - Asimov, Isaac - Foundation Trilogy.txt')

        console.log('text');
    } catch (e) {
        console.log(e)
    }
    // `text` is not available here
})();