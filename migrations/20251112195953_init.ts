import type { Knex } from 'knex';


export async function up(knex: Knex): Promise<void> {
    return knex.schema
      .createTable('results', function (table) {
          table.increments('id');
          table.string('source', 255).notNullable();

          table.integer('nemb').notNullable();
          table.integer('nctx').notNullable();
          table.integer('iteration').notNullable();
          table.integer('error').notNullable();
          table.integer('correct').notNullable();
      });
}


export async function down(knex: Knex): Promise<void> {
}

