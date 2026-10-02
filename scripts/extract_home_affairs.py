from pathlib import Path
import re
import json
import html

RAW = Path("data/raw/home_affairs/current")
OUT = Path("data/processed/home_affairs")
OUT.mkdir(parents=True, exist_ok=True)

PATTERN = re.compile(
    r'\{"title":"(?P<title>(?:\\.|[^"])*)",'
    r'"content":"(?P<content>(?:\\.|[^"])*)",'
    r'"type":"(?P<type>(?:\\.|[^"])*)",'
    r'"category":"(?P<category>(?:\\.|[^"])*)",'
    r'"urls":(?P<urls>\[[^\]]*\]),'
    r'"updateDate":"(?P<updateDate>(?:\\.|[^"])*)"\}'
)

def decode_json_string(value: str) -> str:
    return json.loads('"' + value + '"')

def clean_html(value: str) -> str:
    value = html.unescape(value)
    value = re.sub(r"<script\b[^>]*>.*?</script>", " ", value, flags=re.I | re.S)
    value = re.sub(r"<style\b[^>]*>.*?</style>", " ", value, flags=re.I | re.S)
    value = re.sub(r"<[^>]+>", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()

total_records = 0

for source in sorted(RAW.glob("*.html")):
    text = source.read_text(errors="replace")
    records = []

    for match in PATTERN.finditer(text):
        title = decode_json_string(match.group("title"))
        content = clean_html(decode_json_string(match.group("content")))
        record_type = decode_json_string(match.group("type"))
        category = decode_json_string(match.group("category"))
        urls = json.loads(match.group("urls"))
        update_date = decode_json_string(match.group("updateDate"))

        if not content:
            continue

        records.append({
            "source_file": source.name,
            "title": title,
            "content": content,
            "type": record_type,
            "category": category,
            "urls": [u.strip() for u in urls if u.strip()],
            "update_date": update_date,
            "authority": "Department of Home Affairs",
            "retrieved_date": "2026-10-02"
        })

    output = OUT / f"{source.stem}.json"
    output.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"{source.name}: {len(records)} records")
    total_records += len(records)

print()
print(f"TOTAL RECORDS: {total_records}")
print(f"OUTPUT: {OUT}")
