"""HTTP GET with retries and an on-disk cache, so every scrape can be re-run offline."""
import hashlib
import time

import requests

from fed_news import config

HEADERS = {"User-Agent": "Mozilla/5.0 (academic research; fed funds forecasting)"}


def get(url: str, binary: bool = False, retries: int = 3, pause: float = 0.3,
        cache: bool = True):
    path = config.CACHE / "http" / hashlib.sha1(url.encode()).hexdigest()
    if cache and path.exists():
        data = path.read_bytes()
    else:
        for attempt in range(retries):
            try:
                resp = requests.get(url, headers=HEADERS, timeout=60)
                if resp.status_code == 404:
                    raise FileNotFoundError(url)
                resp.raise_for_status()
                break
            except requests.RequestException:
                if attempt == retries - 1:
                    raise
                time.sleep(3 * (attempt + 1))
        data = resp.content
        if cache:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        time.sleep(pause)
    return data if binary else data.decode("utf-8-sig", errors="replace")
