# Kalshi combo pricer — 2026 midterms

Prices Kalshi all-legs-must-win combos with a one-factor Gaussian copula and
backs out the rho implied by `KXBALANCEPOWERCOMBO` (House x Senate control)
against the separate `CONTROLH-2026` / `CONTROLS-2026` markets.

Market data only: every API call is an unauthenticated GET. Nothing here
places orders or RFQs.

| file | what |
| --- | --- |
| `pricing.py` | `combo_prob(leg_probs, sides, rho)` |
| `calibrate.py` | `implied_rho(...)`, `calibrate_series(df)`, CLI |
| `parse.py` | `load_snapshots()` -> `ticker, time, bid, ask, mid` |
| `collector.py` | saves `data/snapshot_*.json` |

## Setup (Python 3.12)

The venv lives at a short path because deep project paths can exceed
Windows' 260-character limit during `pip install`.

```
py -3.12 -m venv %USERPROFILE%\.venvs\kalshi-combo
%USERPROFILE%\.venvs\kalshi-combo\Scripts\pip install -r requirements.txt
```

## Run

```
python collector.py                         # one snapshot
python collector.py --loop --interval 3600  # hourly
python calibrate.py [--weighted]            # latest rho + per-cell residuals
python -m pytest -q
```

Hourly collection runs as the Task Scheduler job `KalshiComboCollector`
(`pythonw.exe collector.py`, on the hour, through 2026-11-05, catching up
after sleep; only while you're logged on). Check or remove it with:

```
Get-ScheduledTaskInfo KalshiComboCollector
Unregister-ScheduledTask KalshiComboCollector -Confirm:$false
```

A missed hour shows up as a gap in `data/` and a non-zero `LastTaskResult`.

## Model notes

- Z_i = sqrt(rho)·M + sqrt(1-rho)·e_i; leg YES when Z_i < Phi^-1(p_i).
- Gauss-Hermite (64 nodes) for rho <= 0.8. Above that the conditional leg
  probabilities become near-steps in M and Hermite nodes miss them (error
  reaches 0.04 at rho = 0.999), so it switches to piecewise Gauss-Legendre
  with breakpoints at each step. Both agree with the exact bivariate normal
  to < 1e-7 across rho in [0, 0.999].
- rho >= 0 only (one common factor cannot make legs negatively correlated).
- `calibrate.py` prints `cell_rho`, the rho that fits each cell alone. A wide
  spread across cells means one rho can't price all four cells; widen bid/ask
  accordingly.

## Freezing prices (by Nov 1)

The one-time Task Scheduler job `KalshiComboFreeze` runs `freeze.ps1` at
2026-11-01 23:30 local (or as soon as the PC wakes after that). It commits
everything in `data/` and creates the annotated tag `prices-frozen-2026-11-01`,
and refuses to run if that tag already exists. Output goes to `freeze.log`.

Check it afterwards with `git tag -n1` or `Get-Content freeze.log`; run it by
hand with:

```
powershell -ExecutionPolicy Bypass -File freeze.ps1 -Name <name> -Email <email>
```

The collector keeps adding snapshots after the freeze; the tag marks the
frozen set.
