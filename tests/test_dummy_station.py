from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gui import get_station_config_path


def test_dummy_station_config_path_is_available():
    config_path = get_station_config_path()
    assert config_path is not None
    assert Path(config_path).exists()
    assert Path(config_path).name == "dummy_station.yaml"
