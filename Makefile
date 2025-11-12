.PHONY: mc
mc:
	knex migrate:make $(n) -x ts

.PHONY: mu
mu:
	knex migrate:latest
