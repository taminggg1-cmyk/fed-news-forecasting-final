from fed_news.download import __main__ as download_all


def test_steps_run_in_order_and_pass_refresh(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(download_all.config, "DATA", tmp_path)
    calls = []
    steps = [(name, "src", f"{name}.pkl", lambda refresh, name=name: calls.append((name, refresh)) or [1, 2])
             for name in ("calendar", "market", "meetings")]
    download_all.run(refresh=True, steps=steps)
    assert calls == [("calendar", True), ("market", True), ("meetings", True)]
    assert "downloading" in capsys.readouterr().out


def test_saved_files_are_loaded_not_downloaded(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(download_all.config, "DATA", tmp_path)
    (tmp_path / "calendar.pkl").write_bytes(b"x")
    (tmp_path / "alfred").mkdir()
    (tmp_path / "alfred" / "GDP.pkl").write_bytes(b"x")
    assert download_all.saved("calendar.pkl") and download_all.saved("alfred/") and not download_all.saved("labels.pkl")
    download_all.run(steps=[("calendar", "src", "calendar.pkl", lambda refresh: [])])
    assert "already saved, loading" in capsys.readouterr().out


def test_every_download_step_is_listed_once():
    targets = [t for _, _, t, _ in download_all.STEPS]
    assert len(targets) == len(set(targets)) == 8
