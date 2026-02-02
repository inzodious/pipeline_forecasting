# Deal-Level Funnel Analysis Refactor

## Current Approach

- **142K snapshot rows** scanned to answer:
  - Did deal X ever touch Solutioning?
- Complex logic
- Slow performance
- Hard to audit

---

## Proposed Approach

- **~2,700 rows** (one row per deal)
- Binary stage flags:
  - touched_qualified
  - touched_alignment
  - touched_solutioning
- Direct, auditable formula:

    =SUMIF(touched_solutioning=1 AND is_won=1)
     /
     SUMIF(touched_solutioning=1 AND (is_won=1 OR is_lost=1))

---

## Recommended Deal-Level Fact Table Schema

| Column              | Description                                   |
|---------------------|-----------------------------------------------|
| deal_id             | Unique identifier                             |
| market_segment      | Segment                                       |
| date_created        | Creation date                                 |
| date_closed         | Close date (NULL if open)                     |
| net_revenue         | Deal value                                    |
| is_won              | 1 if Closed Won or Verbal                     |
| is_lost             | 1 if Closed Lost                              |
| is_open             | 1 if still open                               |
| is_skipper          | 1 if first stage was Closed Lost              |
| touched_qualified   | 1 if ever in Qualified                        |
| touched_alignment   | 1 if ever in Alignment                        |
| touched_solutioning | 1 if ever in Solutioning                      |
