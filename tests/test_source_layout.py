from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"


def test_source_files_are_under_src():
    for name in [
        "load_data.py",
        "cnn.py",
        "optimize.py",
        "grad_cam.py",
        "report_pdf.py",
        "utils.py",
        "main.py",
    ]:
        assert (SOURCE_ROOT / name).is_file(), name

    assert (SOURCE_ROOT / "streamlit" / "app.py").is_file()


def test_no_python_package_outside_src():
    assert not (PROJECT_ROOT / "data" / "__init__.py").exists()
    assert not (PROJECT_ROOT / "reporting").exists()
