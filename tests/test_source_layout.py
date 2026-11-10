from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_source_directories_are_under_src():
    source_root = PROJECT_ROOT / "src"

    assert (source_root / "app.py").is_file()
    assert (source_root / "data").is_dir()
    assert not (source_root / "neuro_cnn_classifier").exists()