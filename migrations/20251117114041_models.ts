import type { Knex } from "knex";


export async function up(knex: Knex): Promise<void> {
    knex.schema.createTable('models', function (table) {
        table.increments('id');
        table.timestamp('created_at').notNullable().defaultTo(knex.raw('NOW()'));
        table.string('source', 255).notNullable();
        table.integer('corpus_length').notNullable();
    });
}


export async function down(knex: Knex): Promise<void> {
}

