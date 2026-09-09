# Data Contract Specification (per contract)

This is the completeness spec `data_contract_questions.jinja` reads as its `ARCHITECTURE_CHANGES`
-equivalent input — what a single data contract needs in order to have a complete
[Open Data Contract Standard (ODCS) v3.0.0](https://bitol-io.github.io/open-data-contract-standard/)
document. Applied once per contract in `IDENTIFIED_DATA_CONTRACTS` (drafted by the architecture
stage — `prompts/architecture_questions.jinja` — not rediscovered here).

---

## 1. Identity

| Field         | Description                                                          |
| ------------- | ---------------------------------------------------------------------|
| **id**        | Stable kebab-case identifier for the contract                        |
| **name**      | Human-readable contract name                                         |
| **version**   | Current version (semver)                                             |
| **status**    | `draft` / `active` / `deprecated` / `retired`                        |
| **tenant**    | Owning organizational unit, if the system has more than one          |
| **domain**    | `<producer component>` -> `<consumer component>`                     |

---

## 2. Description

### Purpose

<!-- What does this contract govern? Why does the interaction it covers exist? -->

### Usage

<!-- How do the producer and consumer actually use this contract? -->

### Limitations

<!-- What is explicitly NOT known or NOT covered yet? Honest gaps belong here, not silence. -->

---

## 3. Schema

For each field in the payload this contract carries:

| Property       | Description                                                          |
| -------------- | ---------------------------------------------------------------------|
| **name**       | Field name                                                           |
| **type**       | `string` / `integer` / `number` / `boolean` / `object` / `array` / `date` |
| **required**   | Is this field mandatory?                                             |
| **nullable**   | Can this field be null even when present?                            |
| **description**| What the field represents                                            |
| **constraints**| Format, enum, min/max, pattern, or other validation rule              |
| **evidenced**  | Was this field explicitly named in the transcript/clarifications?    |

<!-- Repeat per field. A short, honest schema (only evidenced or structurally-necessary
fields) is correct; a schema padded with plausible-sounding fields nobody actually stated is not. -->

---

## 4. Quality & SLA

| Property         | Description                                                        |
| ----------------- | ------------------------------------------------------------------|
| **Freshness**      | How current must the data be?                                     |
| **Completeness**   | What proportion of expected records/fields must be present?       |
| **Uniqueness**     | What must be unique (e.g. a key field)?                           |
| **Accuracy**       | What correctness guarantee, if any, applies?                      |
| **Availability**   | What uptime/reliability is expected of the producer?               |
| **Retention**      | How long is the data kept?                                        |

---

## 5. Servers / Interfaces

| Property        | Description                                                           |
| ---------------- | ----------------------------------------------------------------------|
| **type**          | e.g. `api`, `kafka`, `custom` (ODCS server type enum — never `other`) |
| **Type-required field** | e.g. `location` for `api`, `host` for `kafka`                  |
| **Description**   | What this channel is for                                             |

<!-- Only include a server entry when both the type AND its type-required field are
known — otherwise the technology fact belongs in Description/Usage as prose instead. -->

---

## 6. Support & Ownership

| Field              | Description                                                       |
| -------------------| --------------------------------------------------------------------|
| **Owning team**     | Who owns/maintains this contract                                  |
| **Support channel** | Where to ask questions about it                                    |

---

## 7. Versioning & Compatibility

<!-- Only applicable when this contract is `modified`/evolving, per its `action` in
IDENTIFIED_DATA_CONTRACTS. -->

| Field                    | Description                                                  |
| -------------------------| ---------------------------------------------------------------|
| **Previous version**      | Version before this change                                    |
| **New version**           | Version after this change                                     |
| **Reason for change**     | Why the version had to change                                 |
| **Breaking change?**      | Does this break existing consumers?                           |
| **Affected consumers**    | Who must adapt because of this change                          |
| **Compatibility**         | Is the new version backward compatible with the old one?      |

---

## 8. References

<!-- Related contracts, external documentation, source diagrams/transcripts this
contract's evidence came from. -->
