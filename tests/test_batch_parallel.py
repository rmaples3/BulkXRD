"""Throughput + correctness: parallel==serial, excluded handling, wavelength
storage, atomic writes, and the headless batch CLI.

Steps 1-2 need numpy/h5py; Step 3a and the batch step-3 path additionally need
pymatgen (skipped when absent).
"""
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from seriesxrd.analysis import phases as ph
from seriesxrd.analysis.background import run_background_separation
from seriesxrd.analysis.peaks import run_peak_fitting
from seriesxrd.analysis import identify as idf
from seriesxrd.analysis import batch
from seriesxrd.analysis.parallel import process_map_or_serial


def _gauss(x, c, a, w):
    return a * np.exp(-0.5 * ((x - c) / w) ** 2)


def _double(value):
    return 2 * value


def _marker_then_error(payload):
    """Record one execution per payload, then fail like a real worker bug.

    A second line in a payload's marker file would prove the workload was
    re-executed (the old serial-retry-on-worker-error behavior)."""
    marker_dir, value = payload
    with open(Path(marker_dir) / f"marker_{value}.txt", "a") as fh:
        fh.write("ran\n")
    raise ValueError(f"bad payload: {value}")


class _SemaphoreDeniedPool:
    """Stand-in for platforms that reject ProcessPool's semaphore creation."""

    def __init__(self, *args, **kwargs):
        raise PermissionError("semaphore creation denied")


class _SubmitDeniedPool:
    """Pool that constructs but fails at submission (fork/spawn denied)."""

    def __init__(self, *args, **kwargs):
        pass

    def map(self, fn, items):
        raise OSError("fork failed")

    def shutdown(self, wait=True, cancel_futures=False):
        pass


def _pool_available() -> bool:
    from concurrent.futures import ProcessPoolExecutor
    try:
        with ProcessPoolExecutor(max_workers=2) as ex:
            return list(ex.map(_double, [1])) == [2]
    except OSError:
        return False


def test_process_map_permission_error_falls_back_in_order():
    with patch(
        "seriesxrd.analysis.parallel.ProcessPoolExecutor",
        _SemaphoreDeniedPool,
    ):
        result = list(process_map_or_serial(
            _double, [3, 1, 4], max_workers=2, label="TEST"))
    assert result == [6, 2, 8]


def test_process_map_submit_time_oserror_falls_back():
    with patch(
        "seriesxrd.analysis.parallel.ProcessPoolExecutor",
        _SubmitDeniedPool,
    ):
        result = list(process_map_or_serial(
            _double, [3, 1, 4], max_workers=2, label="TEST"))
    assert result == [6, 2, 8]


def test_process_map_fallback_does_not_swallow_calculation_error():
    """A worker error from a REAL pool propagates; nothing is re-run serially."""
    import pytest
    if not _pool_available():
        pytest.skip("host cannot create a process pool")
    with tempfile.TemporaryDirectory() as td:
        payloads = [(td, 7), (td, 8)]
        try:
            list(process_map_or_serial(
                _marker_then_error, payloads, max_workers=2, label="TEST"))
        except ValueError as exc:
            assert str(exc) == "bad payload: 7"
        else:
            raise AssertionError("worker error was swallowed")
        for marker in Path(td).glob("marker_*.txt"):
            runs = marker.read_text().count("ran")
            assert runs == 1, f"{marker.name} executed {runs} times"


def test_process_map_streams_results_lazily():
    calls = []

    def worker(value):
        calls.append(value)
        return 2 * value

    gen = process_map_or_serial(worker, [1, 2, 3], max_workers=1, label="TEST")
    assert iter(gen) is gen                      # a lazy iterator, not a list
    assert calls == []                           # nothing ran before consumption
    assert next(gen) == 2
    assert calls == [1]                          # only the consumed payload ran
    assert list(gen) == [4, 6]
    assert calls == [1, 2, 3]


def _make_reduced(path, n=8, nb=1000, excluded_idx=(3,), noise=0.0):
    """Synthetic reduced HDF5 with strong peaks, a diamond spike, a PONI
    wavelength, and one excluded frame. ``noise`` adds Gaussian counts noise
    (noise-free data makes the MAD floor collapse and every fit flag
    BAD_CHI2, which some tests rely on and others must avoid)."""
    import h5py
    rng = np.random.default_rng(7)
    q = np.linspace(1.0, 7.0, nb)
    mean = np.zeros((n, nb), "f4")
    robust = np.zeros((n, nb), "f4")
    for i in range(n):
        shift = 0.01 * i
        bg = 50 + 20 * np.exp(-(q - 1) / 5.0)
        peaks = (_gauss(q, 2.5 - shift, 600, 0.02) + _gauss(q, 3.6 - shift, 500, 0.02)
                 + _gauss(q, 5.1 - shift, 550, 0.02))
        robust[i] = bg + peaks
        if noise:
            robust[i] += rng.normal(0.0, noise, nb)
        mean[i] = robust[i] + _gauss(q, 4.2, 3000, 0.02)   # diamond spike (MEAN only)
    excl = np.zeros(n, bool)
    for j in excluded_idx:
        excl[j] = True
    with h5py.File(str(path), "w") as h5:
        h5.attrs["unit"] = "q_A^-1"
        h5.attrs["poni_text"] = "Detector: Pilatus\nWavelength: 4.1300e-11\n"
        pat = h5.create_group("patterns")
        pat.create_dataset("intensity", data=mean)
        pat.create_dataset("intensity_robust", data=robust)
        pat.create_dataset("radial", data=q)
        fr = h5.create_group("frames")
        names = np.array([f"f_{i:03d}.tif" for i in range(n)], dtype=object)
        fr.create_dataset("filename", data=names, dtype=h5py.string_dtype(encoding="utf-8"))
        fr.create_dataset("excluded", data=excl)


def test_background_wavelength_excluded_and_parallel():
    import h5py
    with tempfile.TemporaryDirectory() as td:
        red = Path(td) / "reduced.h5"
        _make_reduced(red, n=8)
        a1 = Path(td) / "serial.h5"
        a2 = Path(td) / "par.h5"
        run_background_separation(red, a1, num_workers=1)
        run_background_separation(red, a2, num_workers=2)
        # No leftover temp files (atomic write).
        assert not a1.with_name(a1.name + ".tmp").exists()
        with h5py.File(str(a1), "r") as h, h5py.File(str(a2), "r") as g:
            # wavelength parsed from PONI (metres → Å) and stored.
            assert abs(float(h.attrs["wavelength"]) - 0.413) < 1e-3
            # excluded mask propagated.
            assert h["frames/excluded"][3] and not h["frames/excluded"][0]
            # parallel == serial, exactly.
            assert np.array_equal(h["background/clean"][:], g["background/clean"][:])
            assert np.allclose(h["frames/contamination"][:], g["frames/contamination"][:])


def _counts(path):
    import h5py
    with h5py.File(str(path), "r") as h:
        return np.asarray(h["peaks/counts"][:])


def test_peaks_excluded_atomic_and_parallel():
    with tempfile.TemporaryDirectory() as td:
        red = Path(td) / "reduced.h5"
        _make_reduced(red, n=8, excluded_idx=(3,))
        a = Path(td) / "a.h5"
        run_background_separation(red, a, num_workers=1)

        run_peak_fitting(a, None, num_workers=1)        # in place, atomic
        assert not a.with_name(a.name + ".tmp").exists()
        c_serial = _counts(a)
        # excluded frame fitted to zero peaks.
        assert c_serial[3] == 0
        # other frames found the 3 injected reflections.
        assert c_serial[0] >= 3

        # parallel run on a fresh copy → identical counts (strong peaks ⇒ seed-
        # independent, so chunk-boundary seed resets don't change the result).
        a2 = Path(td) / "a2.h5"
        run_background_separation(red, a2, num_workers=1)
        run_peak_fitting(a2, None, num_workers=2)
        assert np.array_equal(c_serial, _counts(a2))


def test_peak_seed_pressure_scan_order_metadata():
    import h5py
    with tempfile.TemporaryDirectory() as td:
        red = Path(td) / "reduced.h5"
        _make_reduced(red, n=6, excluded_idx=(), noise=2.0)
        a = Path(td) / "a.h5"
        run_background_separation(red, a, num_workers=1)
        with h5py.File(str(a), "r+") as h:
            names = np.array([
                "P01_scan002_020.tif", "P01_scan002_010.tif", "P01_scan002_030.tif",
                "P01_scan001_020.tif", "P01_scan001_010.tif", "P01_scan001_030.tif",
            ], dtype=object)
            h["frames/filename"][:] = names
            if "pressure" in h["frames"]:
                del h["frames/pressure"]
            h["frames"].create_dataset(
                "pressure", data=np.array([20, 10, 30, 20, 10, 30], float))

        run_peak_fitting(
            a, None, seed_tracking_axis="pressure", seed_group_by="scan",
            num_workers=2)
        with h5py.File(str(a), "r") as h:
            pk = h["peaks"]
            assert pk.attrs["seed_tracking_axis"] == "pressure"
            assert pk.attrs["seed_group_by"] == "scan"
            assert int(pk.attrs["seed_group_count"]) == 2
            frames = pk["frame"][:]
            assert np.all(frames[:-1] <= frames[1:])      # saved layout stays frame-sorted


def test_identify_excluded_and_parallel():
    if not ph.pymatgen_available():
        print("  (pymatgen not installed — skipping identify parallel/excluded)")
        return
    import h5py
    au = ph.load_bundled()["Au"]
    with tempfile.TemporaryDirectory() as td:
        red = Path(td) / "reduced.h5"
        _make_reduced(red, n=8, excluded_idx=(3,))
        a = Path(td) / "a.h5"
        run_background_separation(red, a, num_workers=1)
        run_peak_fitting(a, None, num_workers=1)

        idf.run_identification(a, [au], p_min=0.0, p_max=150.0, num_workers=1)
        assert not a.with_name(a.name + ".tmp").exists()
        with h5py.File(str(a), "r") as h:
            pr1 = np.asarray(h["identify/Au/pressure"][:])
        assert np.isnan(pr1[3])                  # excluded frame skipped
        assert np.isfinite(pr1[0])

        a2 = Path(td) / "a2.h5"
        run_background_separation(red, a2, num_workers=1)
        run_peak_fitting(a2, None, num_workers=1)
        idf.run_identification(a2, [au], p_min=0.0, p_max=150.0, num_workers=3)
        with h5py.File(str(a2), "r") as h:
            pr2 = np.asarray(h["identify/Au/pressure"][:])
        # parallel == serial on the non-excluded frames.
        ok = np.isfinite(pr1) & np.isfinite(pr2)
        assert ok.sum() >= 6 and np.allclose(pr1[ok], pr2[ok])


def test_batch_cli():
    with tempfile.TemporaryDirectory() as td:
        red = Path(td) / "reduced.h5"
        _make_reduced(red, n=6)
        out = Path(td) / "out.h5"
        steps = "123" if ph.pymatgen_available() else "12"
        argv = [str(red), "-o", str(out), "--steps", steps, "--workers", "2"]
        if ph.pymatgen_available():
            argv += ["--phases", "Au", "--workspace", td]
        rc = batch.main(argv)
        assert rc == 0 and out.is_file()
        import h5py
        with h5py.File(str(out), "r") as h:
            assert "background/clean" in h and "peaks" in h
            if ph.pymatgen_available():
                assert "identify" in h


def test_batch_cli_step2_knobs():
    """The Step-2 detection knobs are honored from the CLI: a --fit-min/--fit-max
    window excludes peaks outside it, and the other overrides parse cleanly."""
    with tempfile.TemporaryDirectory() as td:
        red = Path(td) / "reduced.h5"
        _make_reduced(red, n=4, noise=2.0)      # peaks near q = 2.5, 3.6, 5.1
        out = Path(td) / "out.h5"
        rc = batch.main([
            str(red), "-o", str(out), "--steps", "12", "--workers", "1",
            "--fit-min", "3.0", "--fit-max", "4.5",
            "--min-prominence-snr", "2", "--edge-bins", "5",
            "--min-fwhm-bins", "2", "--detrend-bins", "0",
        ])
        assert rc == 0
        import h5py
        with h5py.File(str(out), "r") as h:
            centers = h["peaks/center"][:]
            flags = h["peaks/flag"][:]
            good = centers[flags == 0]
            assert good.size > 0
            assert good.min() >= 3.0 and good.max() <= 4.5, (good.min(), good.max())
            # The 3.6 reflection is inside the window and is found in every
            # non-excluded frame. The fixture's "diamond spike" at 4.2 is also
            # in the window and is also kept: it is 8.5 bins wide, and the
            # hybrid source removes only features narrower than
            # hybrid_spike_bins (5), treating anything broader as real textured
            # signal. It used to disappear because its fit tripped the
            # chi-square gate on brightness alone — see
            # peaks.DEFAULT_MAX_REL_MISFIT — not because anything rejected it
            # as a spike.
            assert np.sum(np.abs(good - 3.6) < 0.1) == 3, sorted(good)


def main() -> None:
    test_background_wavelength_excluded_and_parallel()
    test_peaks_excluded_atomic_and_parallel()
    test_peak_seed_pressure_scan_order_metadata()
    test_identify_excluded_and_parallel()
    test_batch_cli()
    test_batch_cli_step2_knobs()
    print("BATCH/PARALLEL TEST OK")


if __name__ == "__main__":
    main()
