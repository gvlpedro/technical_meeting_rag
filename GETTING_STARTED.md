# Getting Started

## Start project

```bash
make up
```

## Tests

In total there are 13 tests available

```bash
make test
```

## Useful commands


Ingest youtube transcriptions
```bash
make ingestion DATE=20260515
```

Generate architecture questions for sessions on date 2024-05-15 in Silver (interactive mode) :
```bash
make questions-arch DATE=20260515
```

Generate data contract questions for sessions on date 2024-05-15 in Silver (interactive mode) :
```bash
make questions-data-contracts DATE=20260515
```

Complete classification process (question generation / human answers / ADR generarions) for sessions on date 2024-05-15 in Silver :
```bash
make clarify DATE=20260515 INTERACTIVE=1
```

Chat with the agent (interactive mode) :
```bash
make chat
```

## ACTOR CRITIC BOSS TESTS

Make sure the schema is up to date first:
```bash
make migrate
```

Testing actor-critic-boss for architecture questions
```bash
make test-adr-acb
```

Testing actor-critic-boss for data contract questions
```bash
make test-data-contract-acb
```

Testing actor-critic-boss for data contract questions
```bash
make test-data-contract-acb
```

Automated E2E test — ingests 5 sequential real transcripts (~10 min)
```bash
make test-gold-arch-evolution
```