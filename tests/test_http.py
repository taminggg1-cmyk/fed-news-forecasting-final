import pytest

from fed_news import config, http


class FakeResponse:
    def __init__(self, status=200, content=b"hello"):
        self.status_code = status
        self.content = content

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests
            raise requests.HTTPError(str(self.status_code))


def test_get_caches_to_disk(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(config, "CACHE", tmp_path)
    monkeypatch.setattr(http.requests, "get", lambda *a, **k: calls.append(1) or FakeResponse())
    assert http.get("https://example.org/a", pause=0) == "hello"
    assert http.get("https://example.org/a", pause=0) == "hello"
    assert len(calls) == 1


def test_get_without_cache_refetches(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(config, "CACHE", tmp_path)
    monkeypatch.setattr(http.requests, "get", lambda *a, **k: calls.append(1) or FakeResponse())
    http.get("https://example.org/b", pause=0, cache=False)
    http.get("https://example.org/b", pause=0, cache=False)
    assert len(calls) == 2


def test_get_404_raises_file_not_found_without_retry(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(config, "CACHE", tmp_path)
    monkeypatch.setattr(http.requests, "get", lambda *a, **k: calls.append(1) or FakeResponse(404))
    with pytest.raises(FileNotFoundError):
        http.get("https://example.org/missing", pause=0)
    assert len(calls) == 1


def test_get_without_cache_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CACHE", tmp_path)
    monkeypatch.setattr(http.requests, "get", lambda *a, **k: FakeResponse())
    http.get("https://example.org/c", pause=0, cache=False)
    assert not (tmp_path / "http").exists() or not any((tmp_path / "http").iterdir())
