import { FileService } from './src/services/FileService';

async function foo() {
    const buffer = await (new FileService).readFile('hello.txt');
    console.log(buffer);
    console.log(buffer.toString());
}

foo();