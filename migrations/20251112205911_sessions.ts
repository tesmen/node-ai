import type { Knex } from 'knex';


export async function up(knex: Knex): Promise<void> {
      return knex.schema
        .createTable('sessions', function (table) {
              table.increments('id');
              table.integer('model_id').notNullable();
              table.timestamp('created_at').notNullable().defaultTo(knex.raw('NOW()'));
              table.timestamp('finished_at');
              table.integer('iterations').notNullable();
              table.integer('window_size');
              table.float('correct_ratio').nullable();
              table.boolean('use_slide').nullable().defaultTo(false);
              table.boolean('adjust_pte').nullable().defaultTo(false);
        });
}


export async function down(knex: Knex): Promise<void> {
}

