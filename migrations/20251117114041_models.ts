import type { Knex } from "knex";


export async function up(knex: Knex): Promise<void> {
    return knex.schema.createTable('models', function (table) {
        table.increments('id');
        table.timestamp('created_at').notNullable().defaultTo(knex.raw('NOW()'));
        table.string('source', 255).notNullable();
        table.integer('corpus_length').notNullable();
        table.integer('nemb').notNullable();
        table.integer('nctx').notNullable();
        table.integer('nhidden').notNullable();
        table.json('wte');
        table.json('wpe');
        table.json('itos');
        table.json('stoi');
    });
}


export async function down(knex: Knex): Promise<void> {
}

