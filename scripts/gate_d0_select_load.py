import csv
import hashlib
import json
import re
import statistics
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]
PDF = ROOT / "data" / "raw" / "source_docs" / "ukpn_smart_meter_blob_instructions.pdf"
OUT = ROOT / "data" / "raw" / "ukpn_load_selected.csv"
MANIFEST = ROOT / "data" / "metadata" / "ukpn_scan_manifest.json"
TEMP = ROOT / "tmp" / "data" / "ukpn_month.csv"
BASE = "https://ukpnoppublicdata001.blob.core.windows.net/odp"
GUIDE_URL = "https://ukpowernetworks.opendatasoft.com/api/explore/v2.1/catalog/datasets/ukpn-smart-meter-consumption-lv-feeder/attachments/smart_meter_consumption_data_azure_blob_storagepdf"
SOURCE_BLOB = "LPN/2025/LV_Feeder/scpp_ss_fw_active_reactive_may25_final_LV_LPN.csv"
SOURCE_SIZE = 7021267198
SOURCE_SHA256 = "08d5bf81fe9d79876b01c7baecd0a8e4a46a35da5d8a37b7451513efd22af544"
SELECTED_FEEDER = "LPN-S000000065878:1"
SELECTED_SHA256 = "7380113dee7dd12c7b3739aa642bce1d510c9aa3d14fc48a54190bd724611e33"
LONDON = ZoneInfo("Europe/London")


def download_guide():
    PDF.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(GUIDE_URL, timeout=60) as response, PDF.open("wb") as handle:
        handle.write(response.read())


def sas_token():
    text = "\n".join((p.extract_text() or "") for p in PdfReader(PDF).pages)
    block = re.sub(r"\s+", "", text.split("SAS Token:", 1)[1].split("List Objects:", 1)[0])
    match = re.search(r"(\?sv=.*?&sig=.*?%3D)", block)
    if not match:
        raise RuntimeError("UKPN SAS token was not parsed")
    return match.group(1)


def list_blobs(sas):
    url = BASE + "?restype=container&comp=list&" + sas[1:]
    with urllib.request.urlopen(url, timeout=60) as response:
        root = ET.fromstring(response.read())
    rows = []
    for blob in root.findall(".//Blob"):
        name = blob.findtext("Name")
        if name and name.startswith("LPN/") and "/LV_Feeder/" in name:
            rows.append((name, int(blob.findtext("Properties/Content-Length") or 0)))
    return rows


def first_window(timestamps):
    stamps = sorted(datetime.fromisoformat(str(x).replace("Z", "+00:00")) for x in set(timestamps))
    available = set(stamps)
    first_local = min(x.astimezone(LONDON).date() for x in stamps)
    last_local = max(x.astimezone(LONDON).date() for x in stamps)
    day = first_local
    while day + timedelta(days=27) <= last_local:
        local_start = datetime(day.year, day.month, day.day, tzinfo=LONDON)
        local_end = local_start + timedelta(days=28)
        expected = []
        point = local_start.astimezone(timezone.utc)
        end = local_end.astimezone(timezone.utc)
        while point < end:
            expected.append(point)
            point += timedelta(minutes=30)
        if all(x in available for x in expected):
            return {x.isoformat(timespec="milliseconds").replace("+00:00", "Z") for x in expected}, local_start, local_end
        day += timedelta(days=1)
    raise RuntimeError("No complete 28-day local-time window in first feeder")


def scan_blob(name, size, sas):
    url = BASE + "/" + urllib.parse.quote(name, safe="/") + sas
    digest = hashlib.sha256()
    TEMP.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if TEMP.exists():
        if TEMP.stat().st_size != size:
            raise RuntimeError(f"Existing UKPN file size mismatch: expected {size}, got {TEMP.stat().st_size}")
        processed = 0
        with TEMP.open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
                processed += len(block)
    else:
        with urllib.request.urlopen(url, timeout=180) as response, TEMP.open("wb") as handle:
            processed = 0
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                handle.write(block)
                digest.update(block)
                processed += len(block)
                if processed // (512 * 1024 * 1024) != (processed - len(block)) // (512 * 1024 * 1024):
                    elapsed = time.time() - started
                    print(json.dumps({"download_mb": round(processed / 1048576), "percent": round(100 * processed / size, 1), "mbps": round(processed / 1048576 / elapsed, 2)}), flush=True)
    if processed != size:
        raise RuntimeError(f"UKPN blob size mismatch: expected {size}, got {processed}")
    if digest.hexdigest() != SOURCE_SHA256:
        raise RuntimeError("UKPN blob SHA-256 does not match the frozen source")

    columns = [
        "secondary_substation_id",
        "lv_feeder_id",
        "aggregated_device_count_active",
        "total_consumption_active_import",
        "data_collection_log_timestamp",
    ]
    head = pd.read_csv(TEMP, usecols=columns, dtype=str, keep_default_na=False, nrows=5000)
    head["_key"] = head["secondary_substation_id"] + ":" + head["lv_feeder_id"]
    first_key = head.iloc[0]["_key"]
    first_rows = head.loc[head["_key"] == first_key, "data_collection_log_timestamp"]
    window, local_start, local_end = first_window(first_rows)

    best = None
    invalid_rows = 0
    carry = pd.DataFrame()
    parsed_rows = 0

    def finish_group(key, group):
        nonlocal best
        if group.empty:
            return
        selected = group[group["data_collection_log_timestamp"].isin(window)]
        valid = selected[
            (selected["aggregated_device_count_active"] != "")
            & (selected["total_consumption_active_import"] != "")
        ]
        if len(valid) != len(window) or valid["data_collection_log_timestamp"].nunique() != len(window):
            return
        rows = [
            (row.data_collection_log_timestamp, int(row.aggregated_device_count_active), int(row.total_consumption_active_import))
            for row in valid.itertuples(index=False)
        ]
        score = statistics.median(r[1] for r in rows)
        candidate = (score, key, rows)
        if best is None or score > best[0] or (score == best[0] and key < best[1]):
            best = candidate

    for chunk in pd.read_csv(TEMP, usecols=columns, dtype=str, keep_default_na=False, chunksize=500000):
        invalid_rows += int(((chunk["aggregated_device_count_active"] == "") | (chunk["total_consumption_active_import"] == "") | (chunk["data_collection_log_timestamp"] == "")).sum())
        chunk["_key"] = chunk["secondary_substation_id"] + ":" + chunk["lv_feeder_id"]
        if not carry.empty:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last_key = chunk.iloc[-1]["_key"]
        complete = chunk[chunk["_key"] != last_key]
        carry = chunk[chunk["_key"] == last_key].copy()
        for key, group in complete.groupby("_key", sort=False):
            finish_group(key, group)
        parsed_rows += len(complete)
        if parsed_rows and parsed_rows % 5000000 < 500000:
            print(json.dumps({"parsed_rows": parsed_rows, "current_best_median_meters": None if best is None else best[0]}), flush=True)
    if not carry.empty:
        finish_group(carry.iloc[0]["_key"], carry)

    if best is None:
        raise RuntimeError("No feeder satisfies the locked 28-day completeness rule")
    return best, local_start, local_end, digest.hexdigest(), processed, invalid_rows


def write_outputs(blob, blob_size, best, local_start, local_end, blob_sha256, scanned_bytes, invalid_rows):
    score, key, rows = best
    if key != SELECTED_FEEDER:
        raise RuntimeError(f"Selected feeder changed: expected {SELECTED_FEEDER}, got {key}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda x: x[0])
    with OUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feeder_key", "timestamp_utc", "timestamp_london", "active_meter_count", "total_consumption_wh"])
        for stamp, count, value in rows:
            utc = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            writer.writerow([key, utc.isoformat(), utc.astimezone(LONDON).isoformat(), count, value])
    selected_sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    if selected_sha != SELECTED_SHA256:
        raise RuntimeError("Selected feeder SHA-256 does not match the frozen result")
    manifest = {
        "source": "UK Power Networks Smart Meter Consumption - LV Feeder",
        "license": "CC BY 4.0",
        "source_blob": blob,
        "source_blob_size_bytes": blob_size,
        "source_blob_sha256": blob_sha256,
        "scanned_bytes": scanned_bytes,
        "invalid_source_rows_skipped": invalid_rows,
        "selection_rule": "largest median active meter count among feeders with the earliest complete 28-day Europe/London window; lexicographic feeder-key tie break",
        "selected_feeder": key,
        "median_active_meter_count": score,
        "window_start_london": local_start.isoformat(),
        "window_end_london_exclusive": local_end.isoformat(),
        "selected_rows": len(rows),
        "selected_file": str(OUT.relative_to(ROOT)).replace("\\", "/"),
        "selected_file_sha256": selected_sha,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


def main():
    download_guide()
    sas = sas_token()
    available = dict(list_blobs(sas))
    if SOURCE_BLOB not in available:
        raise RuntimeError("Frozen UKPN source blob is not listed by the current official access token")
    blob = SOURCE_BLOB
    size = available[blob]
    if size != SOURCE_SIZE:
        raise RuntimeError(f"Frozen UKPN source size changed: expected {SOURCE_SIZE}, got {size}")
    print(json.dumps({"selected_source_blob": blob, "size_bytes": size}), flush=True)
    best, local_start, local_end, blob_sha256, scanned_bytes, invalid_rows = scan_blob(blob, size, sas)
    write_outputs(blob, size, best, local_start, local_end, blob_sha256, scanned_bytes, invalid_rows)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}), file=sys.stderr, flush=True)
        raise
