# Data

## `simbench_2016_profiles.csv` (committed)

One full year (2016, leap year, 35 136 × 15-min steps) of normalized
profiles, extracted from the [SimBench](https://simbench.de) benchmark
dataset by `src/real_data.py::load_simbench_profiles()`.

| Column | SimBench profile | Meaning | Normalization |
|---|---|---|---|
| `residential` | `lv_semiurb4` (load, grid `1-MV-urban--0-sw`) | aggregated demand of a semi-urban LV grid, used for CIGRE `Load R*` | annual peak = 1.0 |
| `commercial` | `G0-A` (load, grid `1-MV-urban--0-sw`) | general commercial (BDEW G0-type) demand, used for CIGRE `Load CI*` | annual peak = 1.0 |
| `pv` | `PV3` (renewables, grid `1-MV-semiurb--0-sw`) | MV-connected PV plant feed-in | fraction of installed nameplate (**not** re-scaled) |

**Time stamps.** The raw SimBench time stamps are German local time
*with* daylight-saving shifts (27 March has 92 steps, 30 October has
100). The samples are consecutive 15-min intervals, so they are
re-labelled on a regular CET (UTC+1, no DST) grid starting at
2016-01-01 00:00.

**Key statistics** (`python src/real_data.py`):
PV full-load hours ≈ 681 h/yr (capacity factor 7.7 %), PV peak 0.62 p.u.,
feeder load factor ≈ 43 %, minimum/peak load ≈ 0.13.

**Citation** — if you use these profiles in a report, cite:

> S. Meinecke, D. Sarajlić, S. R. Drauz, A. Klettke, L.-P. Lauven,
> C. Rehtanz, A. Moser, M. Braun, "SimBench — A Benchmark Dataset of
> Electric Power Systems to Compare Innovative Solutions Based on Power
> Flow Analysis", *Energies* 13(12), 3290, 2020.
> https://doi.org/10.3390/en13123290

SimBench data is published under the Open Database License (ODbL).

To regenerate the file: delete it and run `python src/real_data.py`
(requires `pip install simbench`).

## Your own data

* **PV for any location (PVGIS):**
  `python src/run_study.py --data pvgis --lat <lat> --lon <lon> --pv-year 2020`
  downloads hourly PV output for a 1 kWp south-facing system (35° tilt,
  14 % losses) from the EU JRC PVGIS API and caches it here as
  `pvgis_<lat>_<lon>_<year>.csv`. Loads stay SimBench. Cite:
  T. Huld, R. Müller, A. Gambardella, "A new solar radiation database for
  estimating PV performance in Europe and Africa", *Solar Energy* 86,
  2012 — and the PVGIS tool itself (https://re.jrc.ec.europa.eu/pvg_tools/).
* **Any PV CSV:** `--data csv --pv-csv path.csv` with columns
  `time,pv` (pv as fraction of nameplate, hourly or 15-min).
* **Load data** (e.g. smart-meter exports): read with
  `real_data.load_csv_series(path, normalize="peak")` and put it in the
  `residential` / `commercial` columns of the frame returned by
  `load_simbench_profiles()`.
