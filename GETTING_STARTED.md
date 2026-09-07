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

## Useful endpoints

Ingest all transcriptions in session date 2024-05-15 in Bronze :   
```bash 
curl -X POST "http://localhost:8010/v1/ingest?session=20260515"
```

