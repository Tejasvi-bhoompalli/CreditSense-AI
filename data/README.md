# Dataset directory

## What belongs here

Place the credit-risk dataset in this directory (subdirectories are searched too).
Supported formats: `.csv`, `.xls`, `.xlsx`, `.parquet`.

The loader picks the first supported file it finds, preferring the known
filenames in `src/config.py` (`PREFERRED_FILENAMES`) and, failing that, the
largest file. It never downloads anything from the network at run time.

If no dataset is present, the application **does not invent one silently**. It
raises a clear error and offers the documented synthetic generator, which is used
only when a caller explicitly requests it (the test suite does, for speed).

---

## The dataset currently supplied

| Property | Value |
| --- | --- |
| Name | Default of Credit Card Clients |
| Publisher | UCI Machine Learning Repository |
| Authors | I-Ching Yeh and Che-hui Lien |
| Year | 2009 (donated 2016) |
| DOI | [10.24432/C55S3H](https://doi.org/10.24432/C55S3H) |
| Source | <https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients> |
| Licence | **CC BY 4.0** (Creative Commons Attribution 4.0 International) |
| Geography | **Taiwan** |
| Population | Credit-card clients of a Taiwanese bank |
| Observation period | April - September 2005; the target is default in October 2005 |
| Records | 30,000 rows x 23 feature columns + 1 target column |
| Currency | New Taiwan Dollar (NT$) |
| Functions | Classification (binary default) |

### Required attribution

> Yeh, I. (2009). *Default of Credit Card Clients* [Dataset]. UCI Machine Learning
> Repository. <https://doi.org/10.24432/C55S3H>

The licence permits sharing and adaptation for any purpose provided appropriate
credit is given. Credit is given on every page of the application that displays a
prediction, in the README, and in every generated PDF report. The source file is
kept in this repository only because redistribution with attribution is permitted.

### Source paper

> Yeh, I-C., & Lien, C-h. (2009). The comparisons of data mining techniques for the
> predictive accuracy of probability of default of credit card clients.
> *Expert Systems with Applications*, 36(2), 2473-2480.

---

## Geographic scope: this is not an India-specific dataset

**The data was collected in Taiwan in 2005. It contains no Indian borrower
records and no Indian credit-product data.** Model outputs are therefore not
validated for Indian borrowers, Indian income levels, Indian credit products, or
Indian regulatory definitions of default. Applying this prototype to an Indian
lending portfolio would require re-training and validation on locally
representative, lawfully obtained data.

This disclosure appears in the application on the Overview, Risk Assessment,
Dataset Explorer, Model Intelligence, Explainable AI, About and Reports pages, and
in every generated PDF.

---

## Schema and how it is adapted

The source uses abbreviated names. `src/data_loader.py` maps verified source
columns onto the application's canonical schema; nothing is silently equated.

| Source column | Canonical name | Role |
| --- | --- | --- |
| `ID` | *(dropped)* | Identifier, excluded from the model |
| `LIMIT_BAL` | `credit_limit` | Amount of credit granted (includes family/supplementary credit) |
| `SEX` | `sex` | **Excluded** from the model (protected attribute); retained only for group-level reporting |
| `EDUCATION` | `education` | Categorical model feature |
| `MARRIAGE` | `marriage` | **Excluded** from the model (protected attribute); reporting only |
| `AGE` | `age` | Model feature (disclosed as a potential proxy variable) |
| `PAY_0` | `repay_status_1` | Repayment status in September 2005 |
| `PAY_2` .. `PAY_6` | `repay_status_2` .. `repay_status_6` | Repayment status in August .. April 2005 |
| `BILL_AMT1` .. `BILL_AMT6` | `bill_amount_1` .. `bill_amount_6` | Statement balance, September .. April |
| `PAY_AMT1` .. `PAY_AMT6` | `payment_amount_1` .. `payment_amount_6` | Amount paid, September .. April |
| `default.payment.next.month` | `default` | **Target**: 1 = defaulted in October 2005 |

Note on the repayment-status columns: the source publishes `PAY_0` then
`PAY_2` .. `PAY_6` (there is no `PAY_1`). The reason is the month ordering, not a
missing field: `PAY_0` is the most recent month. The loader maps them to
`repay_status_1 .. 6` in descending time order, so suffix `1` is always the most
recent month.

### Documented quirks (not silently repaired)

* The published repayment-status scale documents `-1` (paid duly), `0` (revolving
  credit) and `1..9` (payment delayed by that many months). The distributed
  extract also contains `-2`, which is treated as "no consumption / not
  applicable". It is documented rather than recoded away.
* Statement balances (`BILL_AMT*`) can be negative. These represent credit in the
  client's favour (overpayment or refund) and are preserved rather than clipped
  to zero.
* `EDUCATION` contains undocumented codes `0, 5, 6`; they are grouped into an
  `other` category together with the documented "others" code `4`.
* 56 rows share identical feature values (excluding the target). They are
  retained, because two genuinely different clients can have identical profiles.
* The file contains no missing values, so no imputation was needed for it. The
  pipeline nevertheless imputes (median / most-frequent) **inside the training
  split** so that it behaves correctly on data that does have gaps.

---

## Do not commit confidential data

This project is set up for a public research dataset. If you swap in a private or
client dataset, keep it out of version control: the file is readable from `data/`,
but only files you deliberately stage are committed. `database/` and
`artifacts/` are already ignored. Never place real customer data in a public
repository, and never commit a `.env` file.
