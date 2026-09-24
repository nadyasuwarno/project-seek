import pytest


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    """Every test gets its own empty archive."""
    monkeypatch.setenv("SEEK_DATA_DIR", str(tmp_path / "data"))
    from seek import db
    db.init_db()
    yield tmp_path / "data"
