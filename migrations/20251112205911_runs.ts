import type { Knex } from 'knex';


export async function up(knex: Knex): Promise<void> {
    return knex.schema
      .createTable('runs', function (table) {
          table.increments('id');
          table.timestamp('created_at').notNullable().defaultTo(knex.raw('NOW()'));
          table.timestamp('finished_at');
          table.string('source', 255).notNullable();
          table.integer('corpus_length').notNullable();
          table.integer('wte_length').notNullable();
          table.integer('nemb').notNullable();
          table.integer('nctx').notNullable();
          table.integer('iterations').notNullable();
          table.integer('window_size');
          table.float('correct_ratio').nullable();
          table.boolean('use_slide').nullable().defaultTo(false);

      });
}


export async function down(knex: Knex): Promise<void> {
}

