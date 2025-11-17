import * as fs from 'fs';

export class FileServiceAdapter {
    async readFile(file: string): Promise<Buffer> {
        return fs.readFileSync(file);
    }

    readFileSync(file: string): Buffer {
        return fs.readFileSync(file);
    }

    writeFileSync(file: string, data: string) {
        return fs.writeFileSync(file, data);
    }

    static readFileSync(file: string): Buffer {
        return fs.readFileSync(file);
    }

    static getTextContent(file: string): string {
        const content = FileServiceAdapter.readFileSync(file);

        return content.toString();
    }
}
