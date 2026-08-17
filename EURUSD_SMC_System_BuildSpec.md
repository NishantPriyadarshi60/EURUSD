# EUR/USD Price-Action (SMC) Decision-Support System — Build Specification

**Audience:** an AI coding agent (Claude Code / Cursor) implementing this, plus the human reviewing before handoff.
**Data source:** MetaTrader 5 (MT5) via the `MetaTrader5` Python package.
**Nature:** engineering specification. This is not financial advice and not a recommendation to trade. The goal is a *rigorous, reproducible, honestly-evaluated* research system — not a money printer.

---

## 0. Read this first — operating principles (non-negotiable)

These override any later instruction that conflicts with them. If a feature request and a principle collide, the principle wins.

1. **Do NOT optimize toward a Sharpe target.** There is no "Sharpe 1.5" goal. Build the cleanest honest system possible and *measure* what falls out. A pre-set performance target is a curve-fitting incentive. For calibration: a genuine out-of-sample, cost-inclusive Sharpe of 0.5–1.0 on a retail intraday FX system is a good result; anything reported above ~1.5 is far more likely a bug (lookahead, missing costs, cherry-picked window, overfit) than a real edge, and must be treated as suspect until proven otherwise.

2. **The system's job is to disprove itself.** Every validation step is designed to *kill* the strategy. Only a strategy that survives genuine out-of-sample testing, realistic costs, low parameter sensitivity, and a decent trade count is worth taking seriously.

3. **Closed-bar discipline / no lookahead, everywhere.** A signal on bar `i` may use only information fully available at the *close* of bar `i` (and earlier). Any rule that peeks at the current forming bar, at the bar's own future, or at higher-timeframe data that had not yet closed is a bug. This is the #1 source of fake backtest performance. Every primitive below specifies its confirmation lag explicitly.

4. **Degrees-of-freedom budget.** Every parameter and every "which combination worked" choice is a chance to fit noise. Prefer *few robust concepts with wide stable parameter plateaus* over many fragile ones. Adding a concept must be justified by improved *out-of-sample* expectancy, not backtest curve beauty.

5. **Every concept is a hypothesis, not an article of faith.** Each is independently toggleable and independently measured. If it doesn't earn its place out-of-sample, it gets cut.

6. **Reproducibility.** Versioned data snapshots, seeded randomness, and a logged record of every signal decision with the inputs behind it. A run must be re-creatable bit-for-bit.

---

## 1. Data layer (MT5 specifics — several real traps here)

### 1.1 Retrieval
- Use `MetaTrader5.copy_rates_range` for OHLC bars and `copy_ticks_range` for tick data.
- Timeframes needed: **HTF** (H4 and D1 for bias), **entry TF** (M15 and M5), and **M1 or ticks** for intrabar backtest sequencing (see §6).
- Persist raw pulls to disk (Parquet) as immutable, dated snapshots. All research runs read from snapshots, never live-pull, so results are reproducible.

### 1.2 Volume is NOT real volume — read this
- Spot EUR/USD is OTC. There is **no consolidated traded volume.** MT5's `tick_volume` = number of price updates per bar; `real_volume` is almost always 0 for FX.
- Tick volume is a *usable proxy* for activity in a liquid pair, but it is broker-specific and is **not** contracts/notional. Any concept that leans on "volume" must be explicitly labelled as tick-volume-based and treated as fuzzy.
- If genuine volume is ever required, the only honest source is **CME EUR futures (6E)** — a different instrument with its own session structure. Out of scope for v1; note as a possible future input.

### 1.3 Time / session handling — a real gotcha
- **MT5 bar timestamps are in broker-server time**, typically GMT+2 / GMT+3 with its own DST rules. Do not assume UTC.
- On ingest, convert everything to **UTC** and store the offset used. Define all sessions/kill-zones in UTC. Getting this wrong silently shifts every session-filtered signal.
- Handle DST transitions and weekend gaps (Fri close → Sun/Mon open) explicitly.

### 1.4 Instrument metadata
- Pull `symbol_info`: `point`, `digits`, `trade_tick_size`, `trade_contract_size`, spread. For a 5-digit EUR/USD feed, `point = 0.00001` and 1 pip = `0.0001` = 10 points. Compute pip value from contract size; never hardcode.

### 1.5 Data quality gates (must pass before any backtest)
- No duplicate timestamps, monotonic time index, no zero/negative prices, gaps flagged, OHLC sanity (`low ≤ open,close ≤ high`). Log a data-quality report per snapshot.

---

## 2. Core primitives — precise, codifiable definitions

These are the atoms. Each must produce an identical result for any two implementations given the same candles. Parameters are shown with **starting defaults** — they are hypotheses to be validated (§7), not sacred numbers. Prices default to **wicks** (high/low) unless stated.

> Convention: `ATR(n)` computed on the same timeframe as the primitive unless noted. All "break" tests use the **body close**, not the wick, unless explicitly a wick/sweep test.

### 2.1 Swing points (fractal method — primary)
- **Swing high** at bar `i` with fractal width `N` (default `N = 2`): `high[i] > high[i±k]` for all `k` in `1..N`.
- **Swing low** symmetric with lows.
- **Confirmation lag:** a swing at bar `i` is only *confirmed* at bar `i+N`. Signals may reference a swing only from `i+N` onward. (This is the no-lookahead rule for swings — do not skip it.)
- Maintain an ordered list of confirmed swings with (index, price, type).
- **Two fixed-role fractal widths (see §2.2 decision):** compute swings at **`N_int`** (internal/entry swings, default `2`) and **`N_ext`** (external/bias swings, default `5`). These are *role-assigned*, not searched — do not sweep a single `N` to make structure "look right" (that is hidden discretionary DOF-spend and violates §0.4). Store each swing tagged `internal` / `external`.
- *Alternative to A/B test later:* ATR/percentage zigzag (a swing is a reversal of ≥ `m × ATR`). Keep behind a flag; do not build both into v1.

### 2.2 Market structure state — DUAL RESOLUTION (internal / external)

> **DECISION (made, with reasoning — kept togglable via `structure_mode`):** use **dual-resolution** structure, not single-resolution + HTF-gate alone.
> *Why:* single-resolution does not remove the granularity problem, it hides it inside the choice of one `N`. Whoever then tunes `N` to make M15/M5 structure "look right" is spending degrees of freedom discretionarily and invisibly — exactly what §0.4 forbids. Dual-resolution converts that hidden search into a bounded, role-assigned design choice (`N_int` for entries, `N_ext` for bias), and as a bonus stabilises the premium/discount range (§2.10) because external legs are large by construction. Cost: one extra parameter and the interaction logic below. Worth it here.
> `structure_mode = single` remains available for an ablation run, so you can *measure* whether dual-resolution actually helped rather than assuming it.

Maintain **two** running state machines over confirmed swings, one per resolution:

- **External structure** (from `N_ext` swings): defines local directional bias and the active impulse **leg**. Classify as:
  - **Bullish:** higher highs (HH) + higher lows (HL).  **Bearish:** lower highs (LH) + lower lows (LL).  **Undefined:** until enough external swings exist.
  - Track `ext_last_SH`, `ext_last_SL`, `ext_direction`, and the current external leg (low↔high) used by §2.10.
- **Internal structure** (from `N_int` swings): the finer swings *within* the current external leg, used to time entries.
  - Track `int_last_SH`, `int_last_SL`, `int_direction`.
- **Interaction rule (entry logic lives here):** an entry-side break is an **internal** BOS/CHoCH (§2.3–2.4) that occurs *in the direction of* external bias, inside the external leg. Internal breaks *against* external bias are candidate reversals only, not entries, until external structure itself flips. This is what stops M15/M5 structure from feeling noisy without touching `N`.

### 2.3 BOS (Break of Structure — continuation)
- Computed **per resolution** (external and internal) against that resolution's `last_SH`/`last_SL`.
- **Bullish BOS:** while that resolution's `direction == bullish`, a **candle close** above its `last_SH`. Confirms continuation; `last_SH` advances.
- **Bearish BOS:** while bearish, a close below its `last_SL`.
- BOS is a break *in the direction of* the existing trend. **External BOS** = bias continuation; **internal BOS** aligned with external bias = entry trigger (§2.2 interaction rule).

### 2.4 CHoCH (Change of Character — first counter-trend break)
- Computed **per resolution**.
- **Bullish CHoCH:** while that resolution's `direction == bearish` (making LH/LL), the **first candle close above its most recent `last_SH` (the prior lower high).** Flags a possible flip to bullish.
- **Bearish CHoCH:** while bullish, first close below its `last_SL`.
- Distinction that must be coded exactly: **BOS = close beyond a swing in the trend direction; CHoCH = first close beyond a swing against the trend.** A CHoCH switches that resolution's `direction`; a BOS extends it.
- **Which resolution matters:** an **external CHoCH** changes bias (and only then may internal counter-breaks become entries). An **internal CHoCH** inside an intact external leg is an entry-timing event, not a bias change.

### 2.5 Sweep / liquidity grab (distinct from a break)
- **Sell-side sweep:** `low[i]` pierces a prior swing low / EQL by more than `sweep_tol` (default `0.1 × ATR`), **but** the candle closes back *above* that level — within the same candle or within `sweep_window` candles (default 1–3).
- **Buy-side sweep** symmetric at highs.
- Hard contrast to code: **BOS = close beyond and stay; sweep = wick beyond, close back inside.** The same level poke is one or the other, never both.
- **Sweep-then-CHoCH/BOS** (sweep immediately followed by a counter break) is a first-class high-quality pattern, not noise — flag it as a distinct event.

### 2.6 Order block (OB)
- **Bullish OB:** the **last down-close candle before an up-move that causes a bullish BOS or CHoCH.** Zone = that candle's `[low, high]` (also store body `[open, close]` as an alternative zone behind a flag).
- **Bearish OB:** last up-close candle before a down-move that breaks structure.
- **Quality gate (recommended):** only register an OB if the move that broke structure qualifies as **displacement** (§2.9). This is what separates a real OB from a random candle.
- **Mitigation state:** an OB is `unmitigated` until price returns and trades into the zone; then `mitigated`. Track first-touch bar. Unmitigated OBs are the tradable ones.

### 2.7 Support/Resistance & liquidity pools
- **S/R levels:** cluster confirmed swing highs/lows within `sr_tol` (default `0.15 × ATR` or a fixed pip band). A level's strength = number of touches.
- **Liquidity pools = where stops rest:** prior swing highs/lows and EQH/EQL (§2.8). Feed these as sweep *targets*, not just as levels.

### 2.8 Equal highs / equal lows (EQH/EQL)
- Two or more swing highs within `eqh_tol` of each other (default `0.1 × ATR` or a small fixed pip value) = **EQH** = a liquidity magnet. Symmetric for EQL.
- Output as high-priority liquidity-pool objects (prime sweep targets).

### 2.9 Displacement (keep — it's one of the cleanest concepts)
- A candle is **displacement** if `body_ratio = |close − open| / (high − low) ≥ disp_body` (default 0.6) **AND** `(high − low) ≥ disp_k × ATR(n)` (default `disp_k = 1.5`).
- Consecutive displacement candles form an **impulse leg**.
- Uses: gates OB validity (§2.6), gates FVG (§2.11), and distinguishes a *real* structure break from price limping across a level.
- **Caveat for confluence math:** displacement clusters around news/volatility, so it is *correlated* with session/event context — do **not** count it as fully independent confluence (§4).

### 2.10 Premium / Discount (equilibrium)
- Define the **dealing range** = the current **external** impulse leg (§2.2), from its swing low to swing high. Using the external leg (not an internal one) makes the range stable by construction and removes the discretionary "which leg" choice.
- **Minimum-leg guard (required — closes the fresh-CHoCH edge case):** only draw a range if the leg spans at least `min_leg_atr × ATR(n)` (default `min_leg_atr = 3`). This matters right after an external CHoCH, when the newest qualifying leg can be short and noisy and would give a jumpy equilibrium line exactly when you need it stable.
  - **Fallback when the newest leg is below the guard:** retain the *last valid* range (freeze equilibrium) until a leg clears the threshold; mark premium/discount `stale` in the log so its influence is visible. Do **not** silently redraw off a sub-threshold leg — that is the discretionary patch this guard exists to prevent.
- **Equilibrium = 50%** (Fib midpoint). **Discount** = below 50%, **Premium** = above 50%.
- **Rule:** longs only from OBs/zones in **discount**; shorts only from **premium**. This prevents technically-valid entries at bad prices inside the range.

### 2.11 Fair Value Gap (FVG) — OPTIONAL, off by default in v1
- **Bullish FVG:** 3-candle pattern where `low[i+1] > high[i-1]` (unfilled gap), with middle candle `i` qualifying as **displacement**. Symmetric for bearish.
- **Recommendation (my opinion, stated plainly):** raw FVGs are mostly noise and largely don't work — your instinct is correct. Build **v1 without FVGs.** Add FVG later as a *toggleable* entry refinement, gated by displacement + HTF alignment + premium/discount, and keep it **only if it improves out-of-sample expectancy.** Otherwise cut it.

### 2.12 Mitigation / breaker block (secondary)
- **Breaker:** when an OB is violated (price *closes* fully through it), it flips role — a failed bullish OB becomes bearish resistance (breaker), and vice-versa.
- Lower priority; add after the core works.

### 2.13 Inducement (lowest priority — honestly, the fuzziest)
- Concept: a minor liquidity grab (a small opposing swing between current price and the real OB) that traps traders before the true move.
- **Warning:** this is the hardest to define without discretion. Draft definition: the nearest *minor* (sub-`N`) opposing swing sitting between price and a target OB, whose sweep precedes the OB tap. Treat as **experimental, Phase 3**, heavily out-of-sample tested before trust. Do not let it into v1.

### 2.14 Fake breakout detection
- Largely equivalent to sweep logic (§2.5) plus a failed-BOS check: a close beyond a level that *reverses* and closes back within `fakeout_window` candles without follow-through. Reuse sweep machinery; don't build a parallel concept.

---

## 3. Multi-timeframe bias & session filters

### 3.1 HTF bias (directional gate)
- Compute `structure_direction` on **H4 (and optionally D1)** using only *closed* HTF bars (no lookahead into a forming HTF candle).
- **Gate:** entry-TF long setups allowed only when HTF bias is bullish (or neutral, per config); shorts only when bearish. A price-action system with no HTF gate will fight the dominant trend constantly.

### 3.2 Session / kill zones (timing filter, all UTC)
- London ≈ 07:00–10:00, NY ≈ 12:00–15:00, London–NY overlap ≈ 12:00–16:00, Asian ≈ 00:00–06:00 (adjust to validated values; store as config).
- EUR/USD structure means different things in Asian chop vs London/overlap. Provide a session gate (trade only in configured windows) and log session as a feature for later analysis.

### 3.3 Regime filter (more important for EUR/USD than for gold)
- EUR/USD spends much of its life mean-reverting/range-bound — the exact regime where breakout/continuation systems get chopped. Add an explicit **trend-vs-range regime classifier** (e.g. ADX or an ATR/structure-based measure) and let it gate or re-weight signals. This is central here, not optional.

---

## 4. Confluence assembly (where overfitting sneaks in)

- **Two combination modes, both implemented, config-selectable:**
  1. **Hard AND-gate:** all required conditions must hold → few, high-conviction signals.
  2. **Weighted score + threshold:** each concept contributes a weight; enter above a threshold → more trades, softer. The scoring mode lets you *measure* which concepts actually add expectancy.
- **Guard against redundancy (critical).** Several listed concepts measure overlapping information: BOS / CHoCH / displacement are all "structure broke with force"; inducement / sweep / fake-breakout are cousins. Stacking correlated conditions is *not* confluence — it's counting one vote several times and it silently shrinks your sample. Before trusting a combo, compute pairwise correlation of the signal flags and flag high-overlap sets.
- **Expected high-overlap cluster — name it before you run the check, don't discover it cold.** Dual-resolution (§2.2) adds four structure flags (internal/external × BOS/CHoCH) into an *already-correlated structure family*: internal BOS, external BOS, the CHoCH flags, displacement (which gates all of them), and sweep-then-BOS all encode "structure broke with force." Treat this whole family as a presumed-overlapping set in the §4 correlation matrix. Specifically watch **internal-BOS ↔ external-BOS**, correlated by construction (breaking a large external swing usually means the smaller internal swings in that leg already broke). Note the frequency trap: internal breaks fire far more often than external ones, which can *dilute* the raw correlation number — so here a **low** reading is the surprise to investigate, not a high one, and none of these should be counted as independent votes in the weighted score.
- **Prefer independent inputs:** structure + premium/discount location + session/regime context carry more orthogonal information than three structure-flavoured flags.

---

## 5. Risk, sizing, execution

- **Volatility-aware sizing:** risk a fixed fraction of equity per trade (default 0.5–1%), position size derived from **ATR-based stop distance** and pip value — not fixed lots.
- **Stops:** structure-based (beyond the OB / sweep extreme) or ATR-scaled; config-selectable. Account for gold-like spikes → EUR/USD is calmer but still model stop slippage.
- **Targets:** fixed R multiple, ATR-scaled, or next liquidity pool; config-selectable.
- **Hard limits (coded, not discretionary):** max risk/trade, max daily loss, max weekly loss, max concurrent positions, max correlated exposure. Breaching a limit halts new entries.
- **Separation of concerns:** signal logic and execution/risk are separate modules, independently testable.

---

## 6. Backtest engine — realism requirements (silent result-inflators live here)

- **Intrabar sequencing:** on any bar where both stop and target lie inside the range, a bar-only backtest guesses which hit first and the guess flatters you. Resolve with **M1 or tick data** stepped inside each entry-TF bar, or model conservatively (assume the worse of the two). This must be explicit.
- **Lookahead audit:** automated check that no signal uses same-bar close to trigger something that would have occurred intrabar, and no HTF value from an unclosed HTF bar is used. Fail the build if violated.
- **Costs — model all of them:**
  - **Spread:** use MT5 per-tick spread where available; model **spread widening around news and rollover** (spread is not constant).
  - **Commission** per lot (config).
  - **Slippage** on stops/market fills (worse during spikes).
  - **Swap/financing** on positions held overnight.
- **Costs are broker/account-specific — tag them.** The spread series you use belongs to one named **broker + account type**. **ECN vs market-maker EUR/USD spreads differ ~2–3×**, and on tight-stop M5/M15 setups that gap is the difference between a real edge and a phantom one. Record the broker/account on every run; results do **not** transfer to another broker without re-validation (see §10).
- **Spread-as-%-of-stop — log per trade (cheap viability signal, #5):** record `spread_at_entry / stop_distance`. If this is regularly **>10–15% on M5**, that timeframe is likely non-viable for this pair *regardless of signal quality* — you want to know this **before** Phase 2, not after. Surface the per-TF median in the early feasibility report (§7.0).
- **Fill realism:** a limit touch is not a guaranteed fill; model fill probability or require trade-through for limits.
- **Output per run:** trade log (entry/exit/reason/inputs), equity curve, and full metrics (§7.4).

---

## 7. Validation protocol — the crown jewel (do this *during* development, not as a final stamp)

> **Kill this misconception up front:** Monte Carlo at the end does **not** detect overfitting from your development process. MC on the final equity curve tells you the range of outcomes *a given system* could produce — but if you data-mined many concepts and parameters to *arrive at* that system, MC blesses an already-overfit artifact. Anti-overfitting work must run throughout development. MC is a robustness check, not an overfitting detector.

### 7.0 Sample-size feasibility gate (run EARLY — before tuning anything)
The concept stack (HTF bias → session → regime → premium/discount → displacement-gated OB → sweep) is individually reasonable but **AND-gated on EUR/USD M15/M5 it may land nowhere near the 100–200-trade minimum** over a normal window. Verify feasibility before it becomes a hidden problem.

- **Gate-survival funnel (required diagnostic):** count candidate signals remaining after each filter is applied *in sequence* — raw structure events → after HTF gate → after session gate → after regime gate → after premium/discount → after displacement-gated OB → after sweep. This shows **which gate is killing the sample**, not just that it died. Produce it on in-sample data as soon as Phase 2 primitives exist.
- **Decision rule (binding — this is the important part):** if the honest trade count is short of ~100, that is **diagnostic information about the gate stack, not a signal to add confidence.** Respond by (a) switching AND-gate → **weighted-score mode** (§4) to loosen, and/or (b) **dropping the least-independent gate** (per the redundancy check in §4). **Never** respond by adding more filters — a thin sample needs *fewer* constraints, not more. "Sample too small → add filters to feel surer" is the exact inversion of what small samples require, and is a listed anti-pattern (§11).
- **TF viability pre-check:** fold in the spread-as-%-of-stop median per timeframe (§6). A TF that fails that check is out *before* you spend effort tuning it.

### 7.1 Data partitioning
- **Out-of-sample vault:** carve off a final chunk of history (e.g. most recent 20–25%) and **never touch it** until the very end. It is looked at once.
- **Walk-forward** on the rest: optimize on window `k`, test on window `k+1`, roll forward. Judge on the concatenated out-of-window results, never the in-sample fit.

### 7.2 Parameter sensitivity
- A real edge **degrades gracefully** as parameters shift. Sweep each key parameter and require broad, stable **plateaus**, not sharp peaks. If moving a lookback from 20→22 collapses performance, it's fitted noise → reject.

### 7.3 Degrees-of-freedom / multiple-testing correction
- Log **every variant tried** (parameters + concept combinations). With many trials, some look great by luck. Apply a **deflated-Sharpe / multiple-testing correction** that accounts for the number of trials. Report the deflated figure, not the naive one.

### 7.4 Metrics (judge expectancy and distribution, not win rate)
- Expectancy per trade, profit factor, out-of-sample & deflated Sharpe (state **per-trade vs annualized** and the annualization factor explicitly — this is where numbers get quietly inflated 3–4×), Sortino, max drawdown, drawdown duration, trade count, exposure.
- **Minimum sample:** treat anything under ~100–200 trades as statistically uninformative. Redundant confluence that shrinks the count is a real cost — trace *where* it shrank using the §7.0 gate-survival funnel, and apply the §7.0 decision rule rather than tightening further.

### 7.5 Monte Carlo (positioned correctly, as the *final* robustness gate)
- Resample/reshuffle the trade sequence (and optionally randomize entry timing/costs) to produce a *distribution* of equity paths, drawdowns, and Sharpe. The single backtest curve is one draw. Report percentile ranges (e.g. 5th–95th) for drawdown and final return.
- Run this **on top of** walk-forward + OOS + deflated Sharpe — not instead of them.

### 7.6 Forward test
- Before any real capital: run on a demo/live-feed forward period and compare live behaviour against the backtest. Persistent divergence ⇒ a hidden bias or data mismatch to hunt down.

---

## 8. Build phases (start minimal, add concepts as tested hypotheses)

**Phase 1 — Data + primitives + harness (no strategy yet).**
MT5 ingest, UTC/server-time conversion, data-quality gates, snapshots. Implement §2.1–2.6 including **dual-resolution structure (§2.2: `N_int` + `N_ext`, internal/external state machines and the interaction rule)** — this is now part of "Phase 1 done," not a later refinement. Unit-test every primitive against its exact definition and the no-lookahead lag. Build the backtest engine (§6) and validation harness (§7) *before* tuning anything.

**Phase 2 — Core system v1 (this alone is a complete strategy).**
Combine: HTF bias (§3.1) + dual-resolution BOS/CHoCH structure + displacement-gated OB + sweep + premium/discount (external-leg range with min-leg guard, §2.10) + session gate + regime filter + ATR risk. **No FVG, no inducement, no breaker.**
- **First run the §7.0 feasibility gate**: produce the gate-survival funnel and per-TF spread-%-of-stop before any tuning. If trade count is thin, apply the §7.0 decision rule (loosen / drop least-independent gate — never add filters). Only then run the full validation protocol and establish the honest baseline.

**Phase 3 — Add refinements, one at a time, each as a measured hypothesis.**
EQH/EQL targeting → breaker blocks → FVG (toggleable, displacement-gated) → inducement (experimental). **Keep each only if it improves out-of-sample, deflated performance.** Re-run the full protocol after every addition — a rule change is a *new system* to be re-validated.

**Phase 4 — Forward test, then (only if it survives) consider live.**

---

## 9. Engineering & project structure

- Suggested layout: `data/` (ingest + snapshots), `primitives/` (§2), `signals/` (bias, sessions, regime, confluence), `risk/`, `backtest/` (engine + costs), `validation/` (walk-forward, MC, deflated Sharpe), `configs/` (all parameters — nothing hardcoded), `reports/`.
- **Config-driven:** every parameter, threshold, session window, and toggle lives in versioned config files.
- **Reproducibility:** seed all randomness; snapshot data; log every signal decision with its inputs; changelog every rule change.
- **Testing:** unit tests per primitive (assert exact definitions + confirmation lag), an automated lookahead-bias test, and golden-file tests on a fixed candle sample.

---

## 10. Definition of done (acceptance criteria)

- [ ] MT5 ingest with correct UTC/server-time conversion and passing data-quality gates.
- [ ] Every §2 primitive unit-tested against its exact definition, including confirmation lag.
- [ ] **Dual-resolution structure implemented** (internal/external state machines + interaction rule), with a `structure_mode=single` ablation path.
- [ ] **Interaction-rule test — the negative case specifically:** a unit test confirms an internal counter-trend break (e.g. bullish internal CHoCH inside a bearish external leg) is **blocked from generating an entry until an external CHoCH flips bias.** This is the one composite/state-interaction test in §10 — every other test here checks a primitive in isolation, and both primitives can be individually correct while the entry gate between them leaks. The happy-path test (aligned break → entry) does **not** exercise the block, so it must be tested separately.
- [ ] Automated lookahead-bias check passes.
- [ ] Backtest models spread (incl. widening), commission, slippage, swap, and intrabar sequencing via M1/ticks.
- [ ] **Cost model tagged to a named broker + account type; results explicitly marked non-transferable to another broker without re-validation.**
- [ ] **Premium/discount range uses the external leg with the min-leg-size guard and stale-range fallback (§2.10).**
- [ ] **§7.0 feasibility gate produced before tuning:** gate-survival funnel + per-TF spread-%-of-stop median; §7.0 decision rule applied if the sample is thin.
- [ ] Walk-forward + untouched OOS vault implemented; results reported separately from in-sample.
- [ ] Parameter-sensitivity sweeps show plateaus, not peaks, for retained parameters.
- [ ] Deflated Sharpe reported with trial count; per-trade vs annualized stated explicitly.
- [ ] Trade count ≥ ~100–200 for any result being trusted.
- [ ] Monte Carlo distribution (drawdown/return percentiles) reported on the final system.
- [ ] Every parameter in config; every run reproducible from a seed + data snapshot.

---

## 11. Anti-patterns the implementation must NOT do

- ❌ Optimize toward any Sharpe/return target, or select the window/params that hit a target.
- ❌ Use the forming bar, a bar's own future, or an unclosed HTF bar in any signal.
- ❌ Report performance from in-sample or optimized data as if it were the result.
- ❌ Treat MT5 `tick_volume` as real traded volume.
- ❌ Assume MT5 timestamps are UTC.
- ❌ Stack correlated flags and call it confluence.
- ❌ **Respond to a thin trade count by adding more filters** — small samples need fewer constraints; loosen or drop a gate instead (§7.0).
- ❌ **Present single-broker cost results as broker-agnostic**, or reuse one broker's spread series to justify another's edge.
- ❌ **Tune a single fractal `N` to make structure "look right"** — assign roles via `N_int`/`N_ext` instead (§2.2).
- ❌ Silently redraw the premium/discount range off a sub-threshold leg (§2.10).
- ❌ Run Monte Carlo on a data-mined system and present it as proof it isn't overfit.
- ❌ Ship FVG or inducement in v1, or keep any concept that doesn't earn its place out-of-sample.
- ❌ Assume gold parameters transfer to EUR/USD — re-tune; range/mean-reversion regime matters more here.

---

## 12. Honest expectation-setting (keep this visible)

EUR/USD is the *cheapest and most forgiving pair to trade* (deep liquidity, tight spreads) — but that same efficiency makes it one of the *hardest to beat*: it's the most arbitraged, most algo-saturated market there is. Most retail price-action systems carry no durable edge once realistic costs are applied; the marginal ones get eaten by spread and slippage. Build with the prior that your edge is smaller than it looks, design the validation to disprove the system, and be most skeptical of any variant you had to search hard to find. If it survives genuine out-of-sample testing with full costs, low parameter sensitivity, and a real trade count — that's the rare result worth trading. Everything in this spec exists to tell you honestly whether you have that.

*Engineering guidance only — not financial advice.*
