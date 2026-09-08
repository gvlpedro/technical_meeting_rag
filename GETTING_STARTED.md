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

Ingest all transcriptions in session date 2024-05-15 in Bronze :   
```bash 
make ingestion DATE=20260515
```

Process clarifications in session date 2024-05-15 in Silver :
```bash
make clarify DATE=20260515 INTERACTIVE=1
```
