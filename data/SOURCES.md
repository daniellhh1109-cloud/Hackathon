# External benchmark snapshots

Retrieved 2026-09-13 UTC, solely for performance evaluation, never used as prediction inputs.

- TB3MS: https://fred.stlouisfed.org/graph/fredgraph.csv?id=TB3MS&cosd=2020-12-01&coed=2026-08-31
- SP500: https://fred.stlouisfed.org/graph/fredgraph.csv?id=SP500&cosd=2020-12-01&coed=2026-08-31

TB3MS is an annualised percentage yield, converted as required by the competition to monthly decimal yield by dividing by 1200.
SP500 is a daily closing price index, not a total-return index. Monthly returns use the last available closing level in each calendar month and the previous month's last level. December 2020 is included for January 2021's denominator.

The `.html` files are first-page source snapshots, which contain only 1000 rows. They are retained for provenance, not used for calculation.
No paid service, API key, or trading account was used.
