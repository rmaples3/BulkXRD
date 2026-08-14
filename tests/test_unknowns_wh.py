"""Step 3c unknown clustering + Williamson–Hall microstructure analysis."""
import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import h5py
from seriesxrd.analysis.unknowns import link_tracks, cluster_tracks, run_unknowns
from seriesxrd.analysis.microstructure import williamson_hall
from seriesxrd.analysis.heatmap import unknown_diagram, write_unknown_diagram_csv

TWO_PI = 2 * np.pi


def _residual_file(path, n_frames=15):
    """Analysis HDF5 whose /residual/peaks holds two coherent 'phases':
    cluster A (2 tracks, frames 0-9, drifting up in q) and cluster B
    (2 tracks, frames 5-14), plus one-off noise peaks."""
    rng = np.random.default_rng(0)
    frames, centers, amps, fwhms = [], [], [], []

    def track(f0, f1, q0, drift):
        for f in range(f0, f1 + 1):
            frames.append(f)
            centers.append(q0 + drift * (f - f0) + rng.normal(0, 5e-4))
            amps.append(50.0)
            fwhms.append(0.02)

    track(0, 9, 2.00, +0.004)      # cluster A
    track(0, 9, 3.10, +0.006)
    track(5, 14, 2.60, +0.005)     # cluster B
    track(5, 14, 4.05, +0.008)
    for f in (2, 7, 11):           # isolated noise blips
        frames.append(f); centers.append(5.0 + f * 0.1)
        amps.append(8.0); fwhms.append(0.02)

    counts = np.bincount(np.asarray(frames), minlength=n_frames)
    with h5py.File(str(path), "w") as h:
        h.attrs["unit"] = "q_A^-1"
        rg = h.create_group("residual")
        gp = rg.create_group("peaks")
        gp.create_dataset("counts", data=counts.astype("i4"))
        gp.create_dataset("frame", data=np.asarray(frames, "i4"))
        gp.create_dataset("center", data=np.asarray(centers, "f8"))
        gp.create_dataset("amplitude", data=np.asarray(amps, "f8"))
        gp.create_dataset("fwhm", data=np.asarray(fwhms, "f8"))


def _multi_scan_pressure_file(path):
    """Two interleaved pressure scans with identical pressure-dependent peaks."""
    pressures = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0])
    frames = np.arange(6, dtype=int)
    centers = 2.0 + 0.02 * pressures
    names = [
        "P01_scan001_UOTe_0GPa_000.tif",
        "P02_scan002_UOTe_0GPa_000.tif",
        "P01_scan001_UOTe_1GPa_001.tif",
        "P02_scan002_UOTe_1GPa_001.tif",
        "P01_scan001_UOTe_2GPa_002.tif",
        "P02_scan002_UOTe_2GPa_002.tif",
    ]
    with h5py.File(str(path), "w") as h:
        h.attrs["unit"] = "q_A^-1"
        gf = h.create_group("frames")
        gf.create_dataset("filename", data=np.asarray(names, dtype=object),
                          dtype=h5py.string_dtype(encoding="utf-8"))
        gf.create_dataset("pressure", data=pressures)
        gf.create_dataset("temperature", data=300.0 + pressures * 10.0)
        rg = h.create_group("residual")
        gp = rg.create_group("peaks")
        gp.create_dataset("counts", data=np.ones(6, "i4"))
        gp.create_dataset("frame", data=frames.astype("i4"))
        gp.create_dataset("center", data=centers.astype("f8"))
        gp.create_dataset("amplitude", data=np.full(6, 50.0, "f8"))
        gp.create_dataset("fwhm", data=np.full(6, 0.02, "f8"))


def test_unknown_clustering():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "an.h5"
        _residual_file(p)
        man = run_unknowns(p, min_track_frames=3, jaccard_threshold=0.6)
        assert man["n_tracks"] == 4, man["n_tracks"]      # noise blips dropped
        assert man["n_clusters"] == 2, man["n_clusters"]
        for s in man["clusters"]:
            assert s["n_tracks"] == 2
            assert len(s["d_fingerprint"]) == 2           # both tracks at ref frame
        spans = sorted((s["first_frame"], s["last_frame"]) for s in man["clusters"])
        assert spans == [(0, 9), (5, 14)]                 # transition candidates
        with h5py.File(str(p), "r") as h:
            g = h["unknowns"]
            assert int(g.attrs["n_clusters"]) == 2
            assert g["obs/track"].shape[0] == 40          # 4 tracks x 10 frames
            cl = g["tracks/cluster"][:]
            assert len(set(cl.tolist())) == 2
            # fingerprint d = 2*pi/q of member centers at the reference frame
            fp_d = g["fingerprint/d"][:]
            assert np.all((fp_d > 1.0) & (fp_d < 4.0))
        # re-run replaces the group (idempotent)
        man2 = run_unknowns(p)
        assert man2["n_clusters"] == 2


def test_unknown_diagram_and_csv_export():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "an.h5"
        _residual_file(p)
        run_unknowns(p, min_track_frames=3, jaccard_threshold=0.6)

        diag = unknown_diagram(p, x_axis="frame")
        assert diag["ok"], diag["error"]
        assert diag["n_obs"] == 40
        assert diag["n_frames_with_unknowns"] == 15
        assert len(diag["clusters"]) == 2
        assert set(diag["cluster_ids"]) == {0, 1}
        assert diag["frame"].shape == diag["x"].shape == diag["center"].shape
        assert np.allclose(diag["x"], diag["frame"])
        assert all(c["n_obs"] == 20 for c in diag["clusters"])
        assert all(c["n_frames_observed"] == 10 for c in diag["clusters"])

        out = Path(td) / "unknown_export"
        man = write_unknown_diagram_csv(p, out)
        assert man["n_obs"] == 40
        assert man["n_clusters"] == 2
        assert (out / "unknown_observations.csv").is_file()
        assert (out / "unknown_clusters.csv").is_file()
        text = (out / "unknown_clusters.csv").read_text()
        assert "d_fingerprint" in text

        filtered = Path(td) / "unknown_export_filtered"
        man2 = write_unknown_diagram_csv(p, filtered, min_frames_per_cluster=11)
        assert man2["n_total_obs"] == 40
        assert man2["n_total_clusters"] == 2
        assert man2["n_obs"] == 0
        assert man2["n_clusters"] == 0
        assert (filtered / "unknown_observations.csv").read_text().count("\n") == 1


def test_pressure_tracking_can_group_by_scan():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "an.h5"
        _multi_scan_pressure_file(p)
        no_group = run_unknowns(
            p, tracking_axis="pressure", group_by="none",
            min_track_frames=3, jaccard_threshold=0.6)
        assert no_group["tracking_axis"] == "pressure"
        assert no_group["group_by"] == "none"
        assert no_group["n_tracks"] == 1, no_group

        _multi_scan_pressure_file(p)
        grouped = run_unknowns(
            p, tracking_axis="pressure", group_by="scan",
            min_track_frames=3, jaccard_threshold=0.6)
        assert grouped["group_by"] == "scan"
        assert grouped["n_tracks"] == 2, grouped
        assert grouped["n_clusters"] == 2, grouped
        assert set(grouped["group_labels"]) == {"scan001", "scan002"}
        with h5py.File(str(p), "r") as h:
            g = h["unknowns"]
            assert str(g.attrs["tracking_axis"]) == "pressure"
            assert str(g.attrs["group_by"]) == "scan"
            labels = [x.decode() if isinstance(x, bytes) else str(x)
                      for x in g["groups/label"][:]]
            assert labels == ["scan001", "scan002"]
            assert sorted(g["tracks/group"][:].tolist()) == [0, 1]


def test_temperature_tracking_uses_temperature_axis():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "an.h5"
        _multi_scan_pressure_file(p)
        man = run_unknowns(
            p, tracking_axis="temperature", group_by="scan",
            min_track_frames=3, jaccard_threshold=0.6)
        assert man["tracking_axis"] == "temperature"
        assert man["n_tracks"] == 2
        with h5py.File(str(p), "r") as h:
            axis = h["unknowns/obs/axis"][:]
            assert np.all(np.isin(axis, [300.0, 310.0, 320.0]))


def test_track_linking_gap_tolerance():
    """A track may skip up to max_gap frames (weak peak below SNR) without
    being split in two."""
    frames = np.array([0, 1, 4, 5])         # gap of 2 frames (2, 3 missing)
    centers = np.array([2.0, 2.004, 2.016, 2.02])
    amps = np.ones(4) * 10
    fwhms = np.ones(4) * 0.02
    tr = link_tracks(frames, centers, amps, fwhms, n_frames=6,
                     max_gap=2, min_track_frames=3)
    assert len(tr) == 1 and tr[0]["frames"].size == 4
    tr2 = link_tracks(frames, centers, amps, fwhms, n_frames=6,
                      max_gap=1, min_track_frames=2)
    assert len(tr2) == 2                    # gap too wide -> split


def test_link_tracks_similarity_defaults_are_inert():
    """Without a similarity gate, linking (incl. the equal-gap tie-break) is
    the historical position-only behavior, and tracks now carry their source
    observation rows."""
    # Frame 0 opens two tracks; frame 1 has ONE peak exactly equidistant from
    # both. The stable gap-only sort makes the first-opened track win.
    frames = np.array([0, 0, 1])
    centers = np.array([1.0, 2.0, 1.5])
    amps = np.array([10.0, 5.0, 1.0])
    fwhms = np.full(3, 0.5)
    tracks = link_tracks(frames, centers, amps, fwhms, n_frames=2,
                         min_track_frames=1)
    assert len(tracks) == 2
    # Sorted by descending amplitude sum: the winner absorbed the tied peak.
    assert tracks[0]["rows"].tolist() == [0, 2]
    assert tracks[0]["centers"].tolist() == [1.0, 1.5]
    assert tracks[1]["rows"].tolist() == [1]


def test_link_tracks_similarity_gate_accepts_and_rejects():
    """The ROI-similarity gate re-decides the equal-gap tie on evidence,
    rejects non-finite scores, and keeps linking one-to-one."""
    frames = np.array([0, 0, 1])
    centers = np.array([1.0, 2.0, 1.5])
    amps = np.array([10.0, 5.0, 1.0])
    fwhms = np.full(3, 0.5)

    # Evidence says the tied peak belongs to the SECOND track.
    scores = {(0, 2): 0.05, (1, 2): 0.9}
    tracks = link_tracks(
        frames, centers, amps, fwhms, n_frames=2, min_track_frames=1,
        similarity=lambda a, b: scores.get((a, b), np.nan),
        min_similarity=0.2,
    )
    rows = sorted(t["rows"].tolist() for t in tracks)
    assert rows == [[0], [1, 2]]

    # NaN = no usable evidence: while the gate is active nothing links.
    tracks = link_tracks(
        frames, centers, amps, fwhms, n_frames=2, min_track_frames=1,
        similarity=lambda a, b: float("nan"), min_similarity=0.0,
    )
    assert sorted(t["rows"].tolist() for t in tracks) == [[0], [1], [2]]

    # One-to-one survives the gate: two frame-1 peaks, both preferring the
    # same track by similarity, cannot both join it.
    frames2 = np.array([0, 0, 1, 1])
    centers2 = np.array([1.0, 2.0, 1.4, 1.6])
    amps2 = np.ones(4)
    fwhms2 = np.full(4, 0.6)
    always_high = lambda a, b: 0.9
    tracks = link_tracks(
        frames2, centers2, amps2, fwhms2, n_frames=2, min_track_frames=1,
        similarity=always_high, min_similarity=0.2,
    )
    all_rows = sorted(r for t in tracks for r in t["rows"].tolist())
    assert all_rows == [0, 1, 2, 3]
    assert all(len(set(t["rows"].tolist())) == t["rows"].size for t in tracks)


def test_williamson_hall():
    """Recover known size/strain: dq = 2*pi*K/D + 2*eps*q."""
    K, D, eps = 0.9, 400.0, 0.002
    n_frames, qs = 3, np.linspace(1.0, 5.5, 10)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "an.h5"
        with h5py.File(str(p), "w") as h:
            h.attrs["unit"] = "q_A^-1"
            gp = h.create_group("peaks")
            frame = np.repeat(np.arange(n_frames), qs.size)
            q_all = np.tile(qs, n_frames)
            dq = TWO_PI * K / D + 2 * eps * q_all
            gp.create_dataset("counts", data=np.full(n_frames, qs.size, "i4"))
            gp.create_dataset("frame", data=frame.astype("i4"))
            gp.create_dataset("center", data=q_all)
            gp.create_dataset("fwhm", data=dq)
            gp.create_dataset("fwhm_err", data=np.full(q_all.size, 1e-4))
            gp.create_dataset("flag", data=np.zeros(q_all.size, "i4"))
        man = williamson_hall(p, k_shape=K, min_peaks=5)
        assert man["instrument_corrected"] is False and "warning" in man
        for i in range(n_frames):
            assert abs(man["size_A"][i] - D) < 0.05 * D, man["size_A"][i]
            assert abs(man["strain"][i] - eps) < 0.05 * eps, man["strain"][i]
            assert man["r2"][i] > 0.999
        with h5py.File(str(p), "r") as h:
            assert "microstructure" in h
            assert np.isfinite(h["microstructure/size_A"][:]).all()

        # instrument correction removes a constant-q broadening in quadrature
        inst = 0.008
        with h5py.File(str(p), "r+") as h:
            raw = h["peaks/fwhm"][:]
            h["peaks/fwhm"][...] = np.sqrt(raw**2 + inst**2)
        man2 = williamson_hall(p, k_shape=K, min_peaks=5, instrument_fwhm_q=inst,
                               write=False)
        assert man2["instrument_corrected"] is True
        assert abs(man2["size_A"][0] - D) < 0.05 * D
        assert abs(man2["strain"][0] - eps) < 0.06 * eps


def main() -> None:
    test_track_linking_gap_tolerance()
    test_unknown_clustering()
    test_unknown_diagram_and_csv_export()
    test_pressure_tracking_can_group_by_scan()
    test_temperature_tracking_uses_temperature_axis()
    test_williamson_hall()
    print("UNKNOWNS/WH TEST OK")


if __name__ == "__main__":
    main()
