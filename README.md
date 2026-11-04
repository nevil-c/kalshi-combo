# Kalshi combo pricer — 2026 midterms

Prices Kalshi all-legs-must-win combos with a one-factor Gaussian copula and
backs out the rho implied by `KXBALANCEPOWERCOMBO` (House x Senate control)
against the separate `CONTROLH-2026` / `CONTROLS-2026` markets.

# Report: model vs Kalshi market makers

**Can a model with one correlation number price Senate combos as accurately as
Kalshi's market makers?** Almost. On 195 two-leg combos it tied them for practical
purposes (0.2% worse on Brier score) and beat a no-correlation baseline by 3.5%.

> **Provisional.** 16 of the 18 races in the test have settled. Alaska and
> Mississippi are still open, with Democrats priced at 92% and 93% on Nov 4. The
> headline assumes Democrats win both. Other cases are in the table below.

## Process

**1. Model.** Each race's own Kalshi price $p_i$, plus one number $\rho$ for how much races move together. This is a one-factor Gaussian copula, where $M$ is the shared national factor:

```math
Z_i = \sqrt{\rho}\,M + \sqrt{1-\rho}\,\varepsilon_i, \qquad M,\ \varepsilon_i \sim \mathcal{N}(0,1)
```

A leg wins when $Z_i < \Phi^{-1}(p_i)$, so given $M$ the legs are independent:

```math
P(\text{leg } i \text{ wins} \mid M) = \Phi\!\left(\frac{\Phi^{-1}(p_i) - \sqrt{\rho}\,M}{\sqrt{1-\rho}}\right)
```

```math
P(\text{all legs win}) = \int_{-\infty}^{\infty} \prod_i P(\text{leg } i \text{ wins} \mid M=m)\,\phi(m)\,dm
```

Republican legs load on $M$ with the opposite sign, so a Democratic win in one state and a Republican win in another are negatively related.

**2. Correlation.** Senate control depends on the individual races, so `senate_rho.py` works out the distribution of Democratic seats $S$ among the 35 races up in 2026. Democrats hold 34 seats that aren't up and need 51 for control, so they need 17 of the 35. The estimate $\hat{\rho}$ is the value that matches Kalshi's control price:

```math
P(\text{control} \mid \rho) = \int_{-\infty}^{\infty} P(S \ge 17 \mid M=m)\,\phi(m)\,dm, \qquad P(\text{control} \mid \hat{\rho}) = \text{Kalshi's price}
```

That gave 0.34 if independents don't count as Democratic seats and 0.42 if they do. The test used 0.38.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/control-dark.svg">
  <img src="docs/figures/control-light.svg" alt="Line chart of the model's chance of Democratic Senate control falling as correlation rises. The lines cross Kalshi's 74 to 75 cent price at a correlation of 0.34 with independents not counted and 0.42 with them counted.">
</picture>

**3. Quotes.** On Nov 2, `rfq_scan.py` requested quotes on all 754 two-leg combos of the 18 Senate races in Kalshi's combo collection. 652 got a reply, and 195 got both a bid and an ask. No quote was accepted.

**4. Lock.** The model prices, the quotes and the accuracy calculation were committed and tagged `preregistered-2026-11-02` before polls opened.

**5. Score.** `score.py` compares each forecast $f_j$ with the outcome $y_j \in \{0, 1\}$ on the 195 two-sided combos. Lower is better on both measures:

```math
\text{Brier} = \frac{1}{N}\sum_{j=1}^{N} (f_j - y_j)^2, \qquad \text{Log loss} = -\frac{1}{N}\sum_{j=1}^{N} \left[ y_j \ln f_j + (1-y_j)\ln(1-f_j) \right]
```

The market makers' forecast is their bid-ask midpoint, $f_j = (b_j + a_j)/2$. The baseline treats races as unrelated ($\rho = 0$), so $f_j$ is the product of the leg prices.

## Results

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/accuracy-dark.svg">
  <img src="docs/figures/accuracy-light.svg" alt="Dot plot of Brier scores for the model, the traders and the unrelated-races baseline in five scenarios. In the most likely scenario the model scores 0.1609, the traders 0.1606 and the baseline 0.1667. The traders lead in every scenario, and the baseline is worst in every scenario.">
</picture>

<details>
<summary>All accuracy numbers</summary>

| Alaska and Mississippi | Combos | Model Brier | Makers Brier | Baseline Brier | Model log loss | Makers log loss | Baseline log loss | Model vs makers (Brier) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Democrats win both (most likely)** | 195 | **0.1609** | **0.1606** | 0.1667 | 0.5008 | 0.4994 | 0.5122 | **0.2% worse** |
| Democrats win AK, Republicans win MS | 195 | 0.1580 | 0.1573 | 0.1628 | 0.4950 | 0.4933 | 0.5044 | 0.4% worse |
| Combos with either race left out | 156 | 0.1580 | 0.1529 | 0.1636 | 0.4943 | 0.4831 | 0.5055 | 3.3% worse |
| Republicans win AK, Democrats win MS | 195 | 0.1881 | 0.1790 | 0.1913 | 0.5593 | 0.5402 | 0.5650 | 5.1% worse |
| Republicans win both | 195 | 0.1852 | 0.1757 | 0.1874 | 0.5536 | 0.5340 | 0.5572 | 5.4% worse |

</details>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/breakdown-dark.svg">
  <img src="docs/figures/breakdown-light.svg" alt="Dot plot of Brier scores for the model and traders by group. Both legs in Michigan, Texas, Ohio or Maine: model 0.084, traders 0.058. All other combos: model 0.167, traders 0.169. Same party on both legs: model 0.163, traders 0.165. Mixed parties: model 0.159, traders 0.155.">
</picture>

The model's forecast was closer to the outcome on 99 combos and the makers' on 96.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/prices-dark.svg">
  <img src="docs/figures/prices-light.svg" alt="Scatter plot of model price against the traders' midpoint for 195 combos, clustered close to the diagonal, with an average gap of 4.1 cents and a correlation of 0.97. Same-party combos in Michigan, Texas, Ohio and Maine sit above the diagonal.">
</picture>

## What it showed

- **One correlation number gets close.** The makers lead in every scenario, but only narrowly in the most likely one. The model beats the baseline in all of them.
- **The makers' lead comes from four states.** They priced Michigan, Texas, Ohio and Maine as moving nearly in lockstep, at $\rho \approx 0.9$, and Democrats won all four. The model was slightly better on the other 180 combos.
- **Senate control can't settle that disagreement.** A $\rho$ of 0.9 or more among those four states still fits the control price. Across all nine competitive states, it rules out anything above about 0.65 (independents not counted).
- **Looking back, 0.38 was probably low.** This wasn't part of the locked test, and it is weak evidence. It compares how likely the actual result was at each $\rho$:

```math
R(\rho) = \frac{P(\text{actual 18-race result} \mid \rho)}{P(\text{actual 18-race result} \mid \rho = 0.38)}
```

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/fit-dark.svg">
  <img src="docs/figures/fit-light.svg" alt="Line chart of how likely the actual result was at each correlation, relative to 0.38. It peaks at 1.46 times at a correlation of 0.73 and falls to 0.91 times at the traders' roughly 0.9.">
</picture>

## Limits

- **One election.** The combos share about 18 races and one national mood, so they are
  far from 195 independent tests. This can't establish skill for either side.
- **This measures accuracy, not profit.** The median bid-ask gap was 34¢ and the model
  and makers differed by 4.1¢ on average, so matching the makers' accuracy doesn't
  make a combo profitable to trade.
- **Not an independent view.** The race prices and rho both came from Kalshi.
- **One rho for every pair is crude.** Regional links would need a model with more
  factors.

## Reproduce

```
python senate_rho.py                                # rho from Senate control
python rfq_scan.py --legs 2 --rho 0.38              # dry run; --send needs an API key
python score.py --results data/rfq/assumed_ak_ms.csv   # headline scenario
python report_figures.py                            # redraw the charts
```

`data/rfq/assumed_ak_ms.csv` holds the assumed Alaska and Mississippi results. Once
those races settle, plain `python score.py` gives the final result. Until then its
output is skewed, because a combo with an open leg is only scored if its other leg
has already lost.

# Project

Everything is read-only market data except `rfq_scan.py --send`, which
creates combo markets and opens RFQs (signed with your API key from
environment variables). It never accepts a quote, and it deletes every RFQ it
opens. Nothing here places orders.

| file | what |
| --- | --- |
| `pricing.py` | `combo_prob(leg_probs, sides, rho)` |
| `calibrate.py` | `implied_rho(...)`, `calibrate_series(df)`, CLI |
| `compare.py` | one snapshot: Kalshi combo bid/ask/mid vs engine at the same leg prices (`--rho` or fitted) |
| `parse.py` | `load_snapshots()` -> `ticker, time, bid, ask, mid` |
| `collector.py` | saves `data/snapshot_*.json` |
| `kalshi_client.py` | signed API client (key from env vars) |
| `rfq_scan.py` | RFQ every Senate combo (`--legs 2/3`), record maker quotes vs engine in `data/rfq/` |
| `senate_rho.py` | cross-state rho implied by Senate control; two-tier swing-group check |
| `score.py` | after the election: Brier / log loss for engine vs maker mid and baseline |
| `report_figures.py` | draws this report's charts into `docs/figures` |

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

Snapshots are taken on demand with `python collector.py`. (An hourly Task
Scheduler job ran from Oct 28 to Nov 2 and has been removed; `--loop` still
works if a series is ever needed.)

## Model notes

- $Z_i = \sqrt{\rho}\,M + \sqrt{1-\rho}\,\varepsilon_i$; a leg resolves YES when $Z_i < \Phi^{-1}(p_i)$.
- Gauss-Hermite (64 nodes) for $\rho \le 0.8$. Above that the conditional leg
  probabilities become near-steps in M and Hermite nodes miss them (error
  reaches 0.04 at $\rho = 0.999$), so it switches to piecewise Gauss-Legendre
  with breakpoints at each step. Both agree with the exact bivariate normal
  to within $10^{-7}$ across $\rho \in [0, 0.999]$.
- $\rho \ge 0$ only (one common factor cannot make legs negatively correlated).
- `calibrate.py` prints `cell_rho`, the rho that fits each cell alone. A wide
  spread across cells means one rho can't price all four cells; widen bid/ask
  accordingly.

## Freezing prices (by Nov 1)

The one-time Task Scheduler job `KalshiComboFreeze` runs `freeze.ps1` at
2026-11-01 23:30 local (or as soon as the PC wakes after that). It commits
everything in `data/`, creates the annotated tag `prices-frozen-2026-11-01`,
and pushes `main` and the tag to GitHub (`origin`). If the tag already exists
(for example, the push failed because the PC was offline), it skips straight to
retrying the push. Output goes to `freeze.log`.

Check it afterwards with `git tag -n1`, `git ls-remote --tags origin`, or
`Get-Content freeze.log`; run it (or retry the push) by hand with:

```
powershell -ExecutionPolicy Bypass -File freeze.ps1 -Name <name> -Email <email>
```

The collector keeps adding snapshots after the freeze; the tag marks the
frozen set.
