import type { Knex } from 'knex';


export async function up(knex: Knex): Promise<void> {
    return knex.schema
      .createTable('results', function (table) {
          table.increments('id');
          table.integer('run_id').notNullable();
          table.integer('iteration').notNullable();
          table.integer('error').notNullable();
          table.integer('correct').notNullable();
      });
}


export async function down(knex: Knex): Promise<void> {
}

