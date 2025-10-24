import * as fs from 'fs';

export class FileServiceAdapter {
    async readFile(file: string): Promise<Buffer> {
        return fs.readFileSync(file);
    }

    readFileSync(file: string): Buffer {
        return fs.readFileSync(file);
    }
}
