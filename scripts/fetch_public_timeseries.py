import hashlib
import json
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
LOAD = ROOT / "data" / "raw" / "ukpn_load_selected.csv"
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed" / "timeseries.csv"
MANIFEST = ROOT / "data" / "metadata" / "public_timeseries_manifest.json"
START_UTC = pd.Timestamp("2025-04-01T23:00:00Z")
END_UTC = pd.Timestamp("2025-04-29T23:00:00Z")
EXPECTED_LOAD_SHA256 = "7380113dee7dd12c7b3739aa642bce1d510c9aa3d14fc48a54190bd724611e33"
EXPECTED_PROCESSED_SHA256 = "a2e0993423c2f01417f43172df171b4cb603605e1bf76d558aa1647bbede8608"


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def write_json(name, value):
    path = RAW / name
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fetch_carbon():
    rows = []
    point = START_UTC
    raw = []
    while point < END_UTC:
        stop = min(point + pd.Timedelta(days=14), END_UTC)
        start_text = point.strftime("%Y-%m-%dT%H:%MZ")
        stop_text = stop.strftime("%Y-%m-%dT%H:%MZ")
        url = f"https://api.carbonintensity.org.uk/intensity/{start_text}/{stop_text}"
        payload = get_json(url)
        raw.append({"url": url, "payload": payload})
        rows.extend(payload["data"])
        point = stop
    path = write_json("carbon_intensity.json", raw)
    frame = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime([x["from"] for x in rows], utc=True),
            "carbon_g_per_kwh": [x["intensity"].get("actual") for x in rows],
        }
    )
    frame = frame[(frame.timestamp_utc >= START_UTC) & (frame.timestamp_utc < END_UTC)].drop_duplicates("timestamp_utc")
    if len(frame) != 1344 or frame.carbon_g_per_kwh.isna().any():
        raise RuntimeError("Carbon Intensity API did not provide 1344 actual values")
    return frame.set_index("timestamp_utc"), path, [x["url"] for x in raw]


def fetch_prices():
    params = urllib.parse.urlencode(
        {
            "period_from": START_UTC.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "period_to": END_UTC.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "page_size": 1500,
        }
    )
    url = "https://api.octopus.energy/v1/products/AGILE-24-10-01/electricity-tariffs/E-1R-AGILE-24-10-01-C/standard-unit-rates/?" + params
    payload = get_json(url)
    path = write_json("octopus_prices.json", {"url": url, "payload": payload})
    frame = pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime([x["valid_from"] for x in payload["results"]], utc=True),
            "price_gbp_per_kwh": [x["value_inc_vat"] / 100 for x in payload["results"]],
        }
    )
    frame = frame[(frame.timestamp_utc >= START_UTC) & (frame.timestamp_utc < END_UTC)].drop_duplicates("timestamp_utc")
    if len(frame) != 1344 or frame.price_gbp_per_kwh.isna().any():
        raise RuntimeError("Octopus API did not provide 1344 prices")
    return frame.set_index("timestamp_utc"), path, url


def fetch_weather():
    params = urllib.parse.urlencode(
        {
            "latitude": 51.5074,
            "longitude": -0.1278,
            "start_date": "2025-04-01",
            "end_date": "2025-04-30",
            "hourly": "temperature_2m,shortwave_radiation",
            "timezone": "Europe/London",
        }
    )
    url = "https://archive-api.open-meteo.com/v1/archive?" + params
    payload = get_json(url)
    path = write_json("open_meteo_weather.json", {"url": url, "payload": payload})
    index = pd.DatetimeIndex(payload["hourly"]["time"]).tz_localize("Europe/London").tz_convert("UTC")
    frame = pd.DataFrame(
        {
            "temperature_c": payload["hourly"]["temperature_2m"],
            "shortwave_w_m2": payload["hourly"]["shortwave_radiation"],
        },
        index=index,
    )
    frame.index.name = "timestamp_utc"
    frame = frame.resample("30min").interpolate("time")
    frame = frame[(frame.index >= START_UTC) & (frame.index < END_UTC)]
    if len(frame) != 1344 or frame.isna().any().any():
        raise RuntimeError("Open-Meteo did not provide a complete interpolated window")
    return frame, path, url


def load_frame():
    if sha256(LOAD) != EXPECTED_LOAD_SHA256:
        raise RuntimeError("Selected UKPN feeder SHA-256 does not match the frozen input")
    frame = pd.read_csv(LOAD)
    frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True)
    frame = frame.set_index("timestamp_utc").sort_index()
    if len(frame) != 1344 or frame.index.duplicated().any():
        raise RuntimeError("Selected UKPN load is not a complete 1344-row series")
    frame["load_raw_kw"] = frame["total_consumption_wh"] / 500
    frame["load_kw"] = 100 * frame["load_raw_kw"] / frame["load_raw_kw"].max()
    return frame[["load_raw_kw", "load_kw", "active_meter_count"]]


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    carbon, carbon_path, carbon_urls = fetch_carbon()
    price, price_path, price_url = fetch_prices()
    weather, weather_path, weather_url = fetch_weather()
    frame = load_frame().join(carbon, how="inner").join(price, how="inner").join(weather, how="inner")
    if len(frame) != 1344 or frame.isna().any().any():
        raise RuntimeError("Four-source time intersection is incomplete")
    frame.insert(0, "timestamp_london", frame.index.tz_convert("Europe/London").astype(str))
    frame.to_csv(PROCESSED, lineterminator="\r\n")
    processed_sha256 = sha256(PROCESSED)
    if processed_sha256 != EXPECTED_PROCESSED_SHA256:
        raise RuntimeError("Processed time series SHA-256 does not match the frozen result")
    manifest = {
        "window_start_utc": START_UTC.isoformat(),
        "window_end_utc_exclusive": END_UTC.isoformat(),
        "rows": len(frame),
        "time_step_minutes": 30,
        "load_source": "UK Power Networks Smart Meter Consumption - LV Feeder",
        "load_license": "CC BY 4.0",
        "load_sha256": sha256(LOAD),
        "carbon_urls": carbon_urls,
        "carbon_license": "CC BY 4.0",
        "carbon_raw_sha256": sha256(carbon_path),
        "price_url": price_url,
        "price_raw_sha256": sha256(price_path),
        "weather_url": weather_url,
        "weather_product": "Open-Meteo Historical Weather API default reanalysis match",
        "weather_raw_sha256": sha256(weather_path),
        "processed_file": str(PROCESSED.relative_to(ROOT)).replace("\\", "/"),
        "processed_sha256": processed_sha256,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
