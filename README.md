# PEDF carbon signal reproducibility package

This repository contains computation code for a public data driven study of average and marginal carbon signals in a lossy PEDF DC microgrid. It contains no manuscript, figures, time series, daily result tables, personal information, access keys or cached API responses.

The central question is narrow: when nodal average carbon intensity and response based marginal carbon intensity disagree, how much of an emission response belongs to marginal timing before a direct source objective, a storage carbon boundary or a larger action invalidates that interpretation? The code retains all counterexamples. It does not claim that marginal carbon intensity dominates Great Britain average intensity or direct source emission minimization.

## Frozen study contract

The study uses 28 complete local days from 2 April through 29 April 2025. Each day has 48 half hour periods. The first 21 days are the primary set. The last seven are a later window check, not external validation. The random seed is 20260802. Network, device, solver and acceptance settings are frozen in `configs/pedf8.json`.

The UK Power Networks source is a 7,021,267,198 byte monthly file. It is deliberately excluded from Git. The selection script verifies its SHA 256 digest before scanning it. It then reconstructs the frozen 1,344 row feeder slice and checks that slice against a second digest. Public API data are fetched for the same UTC window. The final joined file must match the digest in `expected_results.json` before experiments can start.

## Data sources and exact download addresses

No account number, meter identifier, postcode, API key or private credential is used.

### UK Power Networks load

Dataset: Smart Meter Consumption, LV Feeder. The source provides half hourly aggregate import consumption and contributing meter counts at feeder level.

* Dataset page: https://ukpowernetworks.opendatasoft.com/explore/dataset/ukpn-smart-meter-consumption-lv-feeder/information/
* Machine readable metadata: https://ukpowernetworks.opendatasoft.com/api/explore/v2.1/catalog/datasets/ukpn-smart-meter-consumption-lv-feeder
* Official Azure access instructions: https://ukpowernetworks.opendatasoft.com/api/explore/v2.1/catalog/datasets/ukpn-smart-meter-consumption-lv-feeder/attachments/smart_meter_consumption_data_azure_blob_storagepdf
* Portal terms: https://ukpowernetworks.opendatasoft.com/terms/terms-and-conditions.pdf
* Dataset licence: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/

The access instructions contain a public, expiring Azure SAS token. That token is not committed. `scripts/gate_d0_select_load.py` downloads the current official instructions and parses the current token at run time. The script pins the original source object `LPN/2025/LV_Feeder/scpp_ss_fw_active_reactive_may25_final_LV_LPN.csv`, checks its size and SHA 256, and selects feeder `LPN-S000000065878:1` by the frozen completeness and active meter rule. The official token and source availability were rechecked on 2 August 2026.

Suggested attribution: UK Power Networks, Smart Meter Consumption, LV Feeder, dataset URL above, portal metadata last updated 1 August 2026. Consult the live metadata before public reuse because the update date can change.

### Great Britain carbon intensity

* Service and licence statement: https://www.carbonintensity.org.uk/
* API documentation: https://api.carbonintensity.org.uk/
* Query 1: https://api.carbonintensity.org.uk/intensity/2025-04-01T23:00Z/2025-04-15T23:00Z
* Query 2: https://api.carbonintensity.org.uk/intensity/2025-04-15T23:00Z/2025-04-29T23:00Z
* Licence: CC BY 4.0, as stated by the official API.

Only the `actual` value is accepted. Forecast values never fill a missing actual value.

### Octopus Agile electricity price

* REST API guide: https://docs.octopus.energy/rest/guides/endpoints/
* Exact tariff query: https://api.octopus.energy/v1/products/AGILE-24-10-01/electricity-tariffs/E-1R-AGILE-24-10-01-C/standard-unit-rates/?period_from=2025-04-01T23%3A00%3A00Z&period_to=2025-04-29T23%3A00%3A00Z&page_size=1500

This public price endpoint does not require authentication. The developer documentation does not present these prices under the CC BY licence used by UK Power Networks and the Carbon Intensity API. This repository therefore fetches the values at run time and does not redistribute the API payload. Users remain responsible for the current Octopus terms.

### Historical weather

* Historical Weather API documentation: https://open-meteo.com/en/docs/historical-weather-api
* Exact query: https://archive-api.open-meteo.com/v1/archive?latitude=51.5074&longitude=-0.1278&start_date=2025-04-01&end_date=2025-04-30&hourly=temperature_2m%2Cshortwave_radiation&timezone=Europe%2FLondon
* API data licence: CC BY 4.0, https://open-meteo.com/en/licence

The frozen query uses the Historical Weather API default reanalysis match. It does not set a single `models` parameter. Reproduction must use the exact URL above. The code does not relabel this response as a uniquely pinned ERA5 Land product.

## Installation

Python 3.12 is required. The reported environment used the exact versions in `requirements.txt`.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

Linux or macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

## Exact reproduction sequence

Run every command from the repository root. Allow at least 10 GB of free disk space for the UK Power Networks file and generated artifacts.

1. Download and scan the frozen UK Power Networks source. If `tmp/data/ukpn_month.csv` already exists, the script verifies and reuses it.

```bash
python scripts/gate_d0_select_load.py
```

Expected selected load digest: `7380113dee7dd12c7b3739aa642bce1d510c9aa3d14fc48a54190bd724611e33`.

2. Fetch the other three sources and build the joined input.

```bash
python scripts/fetch_public_timeseries.py
```

Expected joined input: 1,344 rows with SHA 256 `a2e0993423c2f01417f43172df171b4cb603605e1bf76d558aa1647bbede8608`.

3. Run unit and physical closure tests.

```bash
python -m pytest -q
```

4. Run a one day diagnostic before the full experiment.

```bash
python scripts/run_experiment.py --stage smoke
```

5. Run the frozen 28 day main experiment. The reference machine required about 11 minutes. Hardware and solver runtimes will vary.

```bash
python scripts/run_experiment.py --stage main --days 28
```

6. Run the equal scheduling solve signal role ablation. It creates five alpha schedules and one time shuffled schedule for every day, giving 168 schedules.

```bash
python scripts/run_signal_role_ablation.py
```

7. Run the cost boundary and supporting applicability slices.

```bash
python scripts/run_analysis_campaign.py --slice pareto
python scripts/run_analysis_campaign.py --slice loss
python scripts/run_analysis_campaign.py --slice flexibility
python scripts/run_analysis_campaign.py --slice capacity
python scripts/run_analysis_campaign.py --slice stress
python scripts/run_analysis_campaign.py --slice scale
```

8. Run the direct objective and internal revision campaign in the frozen order. The temporal null slice evaluates 2,660 schedules. The mixed integer slice writes native SCIP logs for 18 representative cases.

```bash
python scripts/run_internal_revision_campaign.py oracle
python scripts/run_internal_revision_campaign.py mci-eps
python scripts/run_internal_revision_campaign.py q0
python scripts/run_internal_revision_campaign.py time-null
python scripts/run_internal_revision_campaign.py time-stat
python scripts/run_internal_revision_campaign.py mip
```

The `oracle` command name is retained for command compatibility. In the manuscript and expected results, B4 is called the direct source objective because it is only an optimization reference inside the frozen synthetic model.

9. Compare the regenerated outputs with the frozen evidence contract.

```bash
python scripts/verify_reproduction.py
```

A successful reproduction prints `"status": "PASS"`. The verifier uses a 0.001 absolute tolerance for reported effect metrics and the stricter physical tolerances stored in `expected_results.json`.

## Expected result boundaries

The primary 21 day median correction of P relative to B3 is 1.471 percentage points of B0 emissions. Circular moving block bootstrap intervals for block lengths of two to four days have a common lower bound of 0.823 points and upper bounds from 3.829 to 3.966. The seven later window days are all positive.

This is not a dominance result. At the 2% cost budget, B4 is lowest on all 28 days. B1 has median regret of 0.156 percentage points to B4 and is the closest nonreference method on 24 days. P has median regret of 1.930 points and is closest on four days. In the signal role ablation, the all day median peaks at alpha 0.75 on the frozen grid, then falls at alpha 1.

Temporal alignment survives three structured null families. Median actual order advantages are 4.110 points for all nonzero circular shifts, 4.043 for four step block permutations and 2.114 for cross day same slot substitutions. The corresponding numbers of days with empirical p at most 0.05 are 18, 21 and 22. These denominators and the 5 April counterexample must remain visible.

The MCI probe scale check is stable at 0.5, 1 and 2 times the frozen epsilon, but this does not validate arbitrary probe sizes. Storage carbon initialization changes the median effect from 1.676 points under the daily first Great Britain ACI rule to 1.767 under continuous cross day propagation and 2.552 under zero initialization. At a 10% action cost budget, P shows a median 0.667% rebound relative to B0 and rebounds on 17 of 28 days. Deterministic B3 mode recovery also diverges from SCIP in the sampled 10% cases, reaching a maximum 38.042 kW action difference.

Two supporting slices have intentional failures that remain in their denominators. The no loss slice completes six of seven cases. The fixed seed correlated stress slice completes 29 of 30. Do not drop those failures or rerun only favorable cases.

## Repository layout

* `configs/pedf8.json` freezes the network, devices, solver and tolerances.
* `src/pedf/` contains dispatch, physical flow recovery, carbon tracing and MCI computation.
* `scripts/gate_d0_select_load.py` downloads and deterministically selects the UK Power Networks input.
* `scripts/fetch_public_timeseries.py` joins load, carbon intensity, price and weather.
* `scripts/run_experiment.py` runs B0, B1, B2, B3 and P.
* `scripts/run_signal_role_ablation.py` runs the alpha and time order adversarial ablation.
* `scripts/run_analysis_campaign.py` runs cost, loss, flexibility, capacity, stress and scale slices.
* `scripts/run_internal_revision_campaign.py` runs B4, MCI probe, storage boundary, temporal null, moving block bootstrap and SCIP slices.
* `scripts/verify_reproduction.py` checks the final outputs.
* `tests/` contains physical dispatch and carbon closure tests.
* `expected_results.json` contains only hashes, counts and aggregate target metrics. It contains no time series or daily results.

## Privacy and licensing

The computation code is released under the MIT License. Dataset licences and terms remain separate and control the downloaded data. The repository contains no raw or processed data. The UK Power Networks input is feeder level aggregate data, and the selected slice contains only a feeder key, timestamps, aggregate meter counts and aggregate consumption. No household level record is used.

Before publishing a fork, run a secret scan and confirm that `data/`, `tmp/`, `artifacts/`, local virtual environments and document files remain untracked. Never commit an Azure SAS token copied from the current UK Power Networks instructions.
