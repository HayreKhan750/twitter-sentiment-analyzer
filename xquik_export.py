import csv
import io
import json


TEXT_FIELDS = ("text", "tweet", "full_text", "content", "body")


def _first_text(record):
    for field in TEXT_FIELDS:
        value = record.get(field)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _json_records(raw_text, file_name):
    if file_name.endswith(".jsonl"):
        return [
            json.loads(line)
            for line in raw_text.splitlines()
            if line.strip()
        ]
    parsed = json.loads(raw_text)
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        for key in ("data", "tweets", "results", "items"):
            value = parsed.get(key)
            if isinstance(value, list):
                return value
        return [parsed]
    raise ValueError("Xquik export must be a JSON object or array.")


def _csv_records(raw_text):
    return list(csv.DictReader(io.StringIO(raw_text)))


def parse_xquik_export(raw_bytes, file_name):
    raw_text = raw_bytes.decode("utf-8-sig")
    lower_name = file_name.lower()
    if lower_name.endswith(".csv"):
        records = _csv_records(raw_text)
    elif lower_name.endswith((".json", ".jsonl")):
        try:
            records = _json_records(raw_text, lower_name)
        except json.JSONDecodeError as exc:
            raise ValueError("Xquik JSON export contains invalid JSON.") from exc
    else:
        raise ValueError("Xquik export must be a .json, .jsonl, or .csv file.")

    texts = []
    for record in records:
        if isinstance(record, dict):
            text = _first_text(record)
            if text:
                texts.append(text)
    if not texts:
        raise ValueError("Xquik export does not contain tweet text rows.")
    return texts
