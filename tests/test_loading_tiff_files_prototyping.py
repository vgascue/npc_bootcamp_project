import os

import pytest

from loading_tiff_files_prototyping import create_output_directory, generate_file_list


def _make_tseries_folder(raw_data_dir, name, channels=("Ch1", "Ch2")):
    """Create a Tseries folder under raw_data_dir with a couple of files per channel."""
    folder = raw_data_dir / name
    folder.mkdir()
    for channel in channels:
        for i in (1, 2):
            (folder / f"{name}_{channel}_{i:03d}.tif").touch()
    return folder


@pytest.fixture
def session_dir(tmp_path):
    """Mimic the project's saving structure: YYYY_MM_DD_animal_x/raw_data/Tseries.../"""
    session = tmp_path / "2026_01_15_animal_3"
    raw_data = session / "raw_data"
    raw_data.mkdir(parents=True)
    return session


def test_generate_file_list_finds_first_ch2_file_per_tseries_folder(session_dir):
    raw_data = session_dir / "raw_data"
    _make_tseries_folder(raw_data, "Tseries-001")
    _make_tseries_folder(raw_data, "Tseries-002")

    result = generate_file_list(str(raw_data))

    assert sorted(os.path.basename(f) for f in result) == [
        "Tseries-001_Ch2_001.tif",
        "Tseries-002_Ch2_001.tif",
    ]


def test_generate_file_list_ignores_non_matching_folders_and_files(session_dir):
    raw_data = session_dir / "raw_data"
    _make_tseries_folder(raw_data, "Tseries-001")
    # A folder that doesn't match the search string should be skipped.
    _make_tseries_folder(raw_data, "SingleImage-001")
    # A stray file (not a directory) should be skipped.
    (raw_data / "notes.txt").touch()

    result = generate_file_list(str(raw_data))

    assert len(result) == 1
    assert "Tseries-001" in result[0]


def test_generate_file_list_only_matches_folder_name_not_full_path(tmp_path):
    # Regression test: the base_directory path itself contains "Tseries", but
    # none of its subfolders do, so nothing should match.
    base_directory = tmp_path / "Tseries_experiment" / "raw_data"
    base_directory.mkdir(parents=True)
    _make_tseries_folder(base_directory, "SingleImage-001")

    result = generate_file_list(str(base_directory))

    assert result == []


def test_generate_file_list_skips_folder_missing_requested_channel(session_dir, capsys):
    raw_data = session_dir / "raw_data"
    _make_tseries_folder(raw_data, "Tseries-001", channels=("Ch1",))

    result = generate_file_list(str(raw_data))

    assert result == []
    assert "Warning" in capsys.readouterr().out


def test_generate_file_list_respects_channel_argument(session_dir):
    raw_data = session_dir / "raw_data"
    _make_tseries_folder(raw_data, "Tseries-001")

    result = generate_file_list(str(raw_data), channel="Ch1")

    assert result[0].endswith("Ch1_001.tif")


def test_create_output_directory_names_folder_after_session(session_dir, tmp_path):
    raw_data = session_dir / "raw_data"
    raw_data.mkdir(exist_ok=True)
    dest_directory = tmp_path / "processed"

    result = create_output_directory(str(raw_data), str(dest_directory))

    assert result == dest_directory / "2026_01_15_animal_3"
    assert result.is_dir()


def test_create_output_directory_handles_trailing_slash(session_dir, tmp_path):
    raw_data = session_dir / "raw_data"
    dest_directory = tmp_path / "processed"

    result = create_output_directory(f"{raw_data}{os.sep}", str(dest_directory))

    assert result == dest_directory / "2026_01_15_animal_3"


def test_create_output_directory_raises_without_a_session_folder(tmp_path):
    dest_directory = tmp_path / "processed"

    with pytest.raises(ValueError):
        create_output_directory("no_parent_folder", str(dest_directory))
