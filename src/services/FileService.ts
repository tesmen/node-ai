import * as fs from 'fs';

export class FileService {
    async readFile(file: string): Promise<Buffer> {
        return fs.readFileSync(file);
    }

    readFileSync(file: string): Buffer {
        return fs.readFileSync(file);
    }
}
