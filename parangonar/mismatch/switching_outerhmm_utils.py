import numpy as np
import warnings
from dataclasses import dataclass
from numpy.lib.stride_tricks import sliding_window_view


def create_match_from_alignment_dict(alignment_dict, sids):
    pids = list(alignment_dict.keys())
    pids_int = [int(pid[1:]) for pid in pids]
    pids_int.sort()
    match = []
    for pid_int in pids_int:
        pid = f"n{pid_int}"
        sid = alignment_dict.get(pid)
        if sid is not None:
            match.append(
                {
                    "label": "match",
                    "score_id": sid,
                    "performance_id": pid,
                }
            )
        else:
            match.append(
                {
                    "label": "insertion",
                    "performance_id": pid,
                }
            )

    for sid in sids:
        if sid not in alignment_dict.values():
            match.append(
                {
                    "label": "deletion",
                    "score_id": sid,
                }
            )
    return match


def calc_avg_notes_per_measure(sna):
    notes_per_measure_list = []
    still_measure_beat = True
    note_count = 0

    for idx, row in enumerate(sna):
        if row["onset_beat"] < 0:
            continue

        if row["rel_onset_div"] == 0:
            if row["is_grace"]:
                note_count += 1
                continue
            if still_measure_beat:
                note_count += 1
            else:
                still_measure_beat = True
                if note_count > 0:
                    notes_per_measure_list.append(note_count)
                note_count = 1
        else:
            still_measure_beat = False
            note_count += 1
    avg_notes_per_measure = round(np.mean(notes_per_measure_list))
    return avg_notes_per_measure


def calc_avg_unique_onsets_per_measure(sna):
    unique_onsets = np.unique(sna["onset_beat"])
    onsets_per_measure_list = []
    onset_count = 0
    num_of_measures = 0

    for idx, onset in enumerate(unique_onsets):
        if onset < 0:
            continue

        row = sna[sna["onset_beat"] == onset][0]
        is_grace = row["is_grace"]
        is_measure_beat = row["rel_onset_div"] == 0

        if is_measure_beat:
            if not is_grace:
                if onset_count > 0:
                    onsets_per_measure_list.append(onset_count)
                onset_count = 1
                num_of_measures += 1
            else:
                onset_count += 1
        else:
            onset_count += 1
    avg_onsets_per_measure = round(np.mean(onsets_per_measure_list))
    return avg_onsets_per_measure, num_of_measures


def find_diagonal_runs_with_coords(matrix, min_length=100):
    matrix = np.array(matrix)
    n, m = matrix.shape
    results = []

    for offset in range(1, m):  # skips main diagonal
        diag = np.diagonal(matrix, offset=offset)

        run_coords = []

        for i, val in enumerate(diag):
            if val == 1:
                rr = i
                cc = i + offset

                # safety: explicitly exclude main diagonal
                if rr == cc:
                    continue

                run_coords.append((rr, cc))
            else:
                if len(run_coords) >= min_length:
                    results.append(run_coords.copy())
                run_coords = []

        if len(run_coords) >= min_length:
            results.append(run_coords.copy())

    return results


def remove_subset_runs(runs):
    # Sort by descending length (longest first)
    runs_sorted = sorted(runs, key=len, reverse=True)

    filtered = []
    filtered_sets = []

    for run in runs_sorted:
        run_set = set(run)

        is_subset = False
        for kept_set in filtered_sets:
            if run_set.issubset(kept_set):
                is_subset = True
                break

        if not is_subset:
            filtered.append(run)
            filtered_sets.append(run_set)

    return filtered


def remove_overlaps_with_simple_trimming(runs, min_length=20):
    runs_sorted = sorted(runs, key=len, reverse=True)

    accepted = []
    occupied = set()

    for run in runs_sorted:
        run_set = set(run)

        if not (run_set & occupied):
            # no overlap → accept whole run
            accepted.append(run)
            occupied.update(run)
            continue

        # overlap exists → trim from ends
        start = 0
        end = len(run)

        # trim prefix
        while start < end and run[start] in occupied:
            start += 1

        # trim suffix
        while end > start and run[end - 1] in occupied:
            end -= 1

        trimmed = run[start:end]

        if len(trimmed) >= min_length:
            accepted.append(trimmed)
            occupied.update(trimmed)

    return accepted


def is_subrange(sub_start, sub_end, super_start, super_end):
    return super_start <= sub_start and sub_end <= super_end


def look_for_equivalent_score_ids(sna, min_diagonal_length=20, num_diagonals_limit=50):
    s_pitches = sna["pitch"]
    score_ssm = np.zeros((len(s_pitches), len(s_pitches)))
    for i in range(len(s_pitches)):
        for j in range(len(s_pitches)):
            score_ssm[i, j] = 1 if s_pitches[i] == s_pitches[j] else 0

    diagonals = find_diagonal_runs_with_coords(
        score_ssm, min_length=min_diagonal_length
    )
    filtered_diagonals = remove_subset_runs(diagonals)
    accepted_diagonals = remove_overlaps_with_simple_trimming(
        filtered_diagonals, min_length=min_diagonal_length
    )

    if len(accepted_diagonals) > num_diagonals_limit:
        return None, None, None, None, None, None, None
    elif len(accepted_diagonals) == 0:
        return None, None, None, None, None, None, 0

    print(
        f"Found {len(accepted_diagonals) + 1} accepted diagonals with min_diagonal_length={min_diagonal_length}."
    )

    diagonal_borders_dict = dict()

    diagonal_num = 0
    expanded_accepted_diagonals = []
    diagonals_beats_to_num_dict = dict()
    for diagonal in accepted_diagonals:
        diag1_start = diagonal[0][0]
        diag1_start_beat = sna["onset_beat"][diag1_start]
        diag1_end = diagonal[-1][0]
        diag1_end_beat = sna["onset_beat"][diag1_end]
        diag2_start = diagonal[0][1]
        diag2_start_beat = sna["onset_beat"][diag2_start]
        diag2_end = diagonal[-1][1]
        diag2_end_beat = sna["onset_beat"][diag2_end]
        expanded_diagonal1 = (diag1_start_beat, diag1_end_beat)
        expanded_diagonal2 = (diag2_start_beat, diag2_end_beat)
        if expanded_diagonal1 in expanded_accepted_diagonals:
            continue
        else:
            expanded_accepted_diagonals.append(expanded_diagonal1)
            diagonals_beats_to_num_dict[expanded_diagonal1] = diagonal_num
            diagonal_borders_dict[diagonal_num] = expanded_diagonal1
            diagonal_num += 1

        if expanded_diagonal2 in expanded_accepted_diagonals:
            continue
        else:
            expanded_accepted_diagonals.append(expanded_diagonal2)
            diagonals_beats_to_num_dict[expanded_diagonal2] = diagonal_num
            diagonal_borders_dict[diagonal_num] = expanded_diagonal2
            diagonal_num += 1

    ids_association_dict = dict()
    ref_score_id_dict = dict()

    onset_beat_associations_dict = dict()

    for diagonal in accepted_diagonals:
        for coord in diagonal:
            if coord[0] < coord[1]:
                equivalent_score_id_coord = coord[1]
                ref_score_id_coord = coord[0]
            if coord[0] > coord[1]:
                equivalent_score_id_coord = coord[0]
                ref_score_id_coord = coord[1]

            ref_score_id = sna["id"][ref_score_id_coord]
            equivalent_score_id = sna["id"][equivalent_score_id_coord]
            ref_score_id_dict[equivalent_score_id] = ref_score_id

    minimum_ref_id_dict = dict()
    min_ref_onset_beat_dict = dict()
    for equiv_key in ref_score_id_dict.keys():
        temp_ref_id = ref_score_id_dict[equiv_key]
        while temp_ref_id in ref_score_id_dict.keys():
            temp_ref_id = ref_score_id_dict[temp_ref_id]
        minimum_ref_id_dict[equiv_key] = temp_ref_id
        equiv_beat = sna[sna["id"] == equiv_key]["onset_beat"][0]
        ref_beat = sna[sna["id"] == temp_ref_id]["onset_beat"][0]
        min_ref_onset_beat_dict[equiv_beat] = ref_beat

    min_ref_id_dict_keys = minimum_ref_id_dict.keys()
    for equiv_key in min_ref_id_dict_keys:
        min_ref_id = minimum_ref_id_dict[equiv_key]
        if min_ref_id not in ids_association_dict.keys():
            ids_association_dict[min_ref_id] = [equiv_key]
        else:
            if equiv_key not in ids_association_dict[min_ref_id]:
                ids_association_dict[min_ref_id].append(equiv_key)

    ids_association_dict_keys = ids_association_dict.keys()
    for min_ref_id in ids_association_dict_keys:
        if min_ref_id not in minimum_ref_id_dict.keys():
            minimum_ref_id_dict[min_ref_id] = min_ref_id
        if min_ref_id not in ids_association_dict[min_ref_id]:
            ids_association_dict[min_ref_id].append(min_ref_id)

    min_ref_onset_beat_dict_keys = min_ref_onset_beat_dict.keys()
    for equiv_beat in min_ref_onset_beat_dict_keys:
        min_ref_beat = min_ref_onset_beat_dict[equiv_beat]
        if min_ref_beat not in onset_beat_associations_dict.keys():
            onset_beat_associations_dict[min_ref_beat] = [equiv_beat]
        else:
            if equiv_beat not in onset_beat_associations_dict[min_ref_beat]:
                onset_beat_associations_dict[min_ref_beat].append(equiv_beat)

    onset_beat_associations_dict_keys = onset_beat_associations_dict.keys()
    for min_ref_beat in onset_beat_associations_dict_keys:
        if min_ref_beat not in min_ref_onset_beat_dict.keys():
            min_ref_onset_beat_dict[min_ref_beat] = min_ref_beat
        if min_ref_beat not in onset_beat_associations_dict[min_ref_beat]:
            onset_beat_associations_dict[min_ref_beat].append(min_ref_beat)

    return (
        ids_association_dict,
        minimum_ref_id_dict,
        onset_beat_associations_dict,
        min_ref_onset_beat_dict,
        diagonals_beats_to_num_dict,
        diagonal_borders_dict,
        len(accepted_diagonals),
    )

######################################################################################
# WARPING PATH EXCURSION CLEANUP (used by SwitchSnapOuterHMM.clean_excursions)
#
# Detection: each note's beat is compared with a robust local reference (the centre
# of the densest cluster of beats in a centred window, ignoring already-flagged notes,
# iterated). Stage 1 (short window) repairs short excursions outright; stage 2 (long
# window) repairs longer ones only if the score pitches fit "stayed on the path" at
# least as well as the excursion.
# Repair: 'return' -> interpolate between the clean neighbours; 'transition' -> split
# the run between "end of previous section" and "start of next section" (by pitch
# agreement); 'leading'/'trailing' -> extrapolate from the one available side.
######################################################################################

@dataclass
class CleanupParams:
    # --- detection ---
    window: int = 17                 # stage 1: removes excursions up to ~(window-1)//2 notes
    long_window: int = 81            # stage 2: candidates up to ~(long_window-1)//2 notes (0 = off)
    far_beats: float = 8.0           # deviation from the local reference that counts as "away"
    cluster_tol_beats: float = 4.0   # width of the "densest cluster" used as reference
    detect_iters: int = 4            # re-estimation passes ignoring flagged notes
    merge_gap: int = 1               # flagged runs separated by <= this many notes are merged
    pitch_margin: float = 0.1        # stage 2: remove if home pitch-match >= here - margin
    # --- repair ---
    chord_tol_sec: float = 0.05
    tempo_ratio_max: float = 3.0
    return_tol_beats: float = 4.0    # slack when deciding whether the path came back
    tempo_window_sec: float = 10.0
    default_tempo_bps: float = 2.0
    return_mode: str = "interpolate"   # "interpolate" or "snap"
    transition_default: str = "next"   # without pitch info: give the run to "next" or "prev"
    pitch_match_tol_beats: float = 1.0
    snap_to_score_onsets: bool = True


class ScoreIndex:
    """Fast lookups into the score: onsets, per-pitch onsets, and note ids."""

    def __init__(self, score_beats, score_pitches, score_ids=None):
        sb = np.asarray(score_beats, float)
        sp = np.asarray(score_pitches, int)
        ids = np.asarray(score_ids) if score_ids is not None else np.arange(len(sb))
        order = np.argsort(sb, kind="stable")
        sb, sp, ids = sb[order], sp[order], ids[order]
        self.onsets = np.unique(sb)
        self.lo, self.hi = float(sb.min()), float(sb.max())
        self.by_pitch = {int(p): (sb[sp == p], ids[sp == p]) for p in np.unique(sp)}

    @staticmethod
    def _nearest(arr, b):
        j = np.searchsorted(arr, b)
        cands = [k for k in (j - 1, j) if 0 <= k < len(arr)]
        return min(cands, key=lambda k: abs(arr[k] - b))

    def matches(self, pitches, beats, tol):
        out = np.zeros(len(beats), bool)
        for i, (p, b) in enumerate(zip(pitches, beats)):
            entry = self.by_pitch.get(int(p))
            if entry is not None and entry[0].size:
                out[i] = abs(entry[0][self._nearest(entry[0], b)] - b) <= tol
        return out

    def snap(self, beats, pitches=None, tol=1.0):
        """Snap to the nearest same-pitch onset within tol, else the nearest onset."""
        out = np.empty(len(beats))
        for i, b in enumerate(beats):
            entry = self.by_pitch.get(int(pitches[i])) if pitches is not None else None
            if entry is not None and entry[0].size:
                cand = entry[0][self._nearest(entry[0], b)]
                if abs(cand - b) <= tol:
                    out[i] = cand
                    continue
            out[i] = self.onsets[self._nearest(self.onsets, b)]
        return out

    def nearest_ids(self, beats, pitches):
        out = []
        for p, b in zip(pitches, beats):
            entry = self.by_pitch.get(int(p))
            out.append(None if entry is None or not entry[0].size
                       else entry[1][self._nearest(entry[0], b)])
        return out


# -------------------------------detection---------------------------------------------- 

def running_consensus(beats, mask, window, tol):
    """
    Robust local reference: for each note, the median of the densest cluster
    (values within +-tol of each other) in a centred window, ignoring masked notes.
    """
    n = len(beats)
    w = window if window % 2 else window + 1
    h = w // 2
    x = np.where(mask, np.nan, np.asarray(beats, float))
    W = sliding_window_view(np.pad(x, h, constant_values=np.nan), w)
    ref = np.full(n, np.nan)
    chunk = max(1, int(4e6 // (w * w)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for a in range(0, n, chunk):
            Wc = W[a:a + chunk]
            counts = (np.abs(Wc[:, :, None] - Wc[:, None, :]) <= tol).sum(axis=2)
            counts[np.isnan(Wc)] = -1
            centre = Wc[np.arange(len(Wc)), counts.argmax(axis=1)]
            near = np.abs(Wc - centre[:, None]) <= tol
            ref[a:a + chunk] = np.nanmedian(np.where(near, Wc, np.nan), axis=1)
    bad = np.isnan(ref)
    if bad.all():
        return np.asarray(beats, float).copy()
    if bad.any():
        idx = np.arange(n)
        ref[bad] = np.interp(idx[bad], idx[~bad], ref[~bad])
    return ref


def detect_excursions(beats, p, window=None, protect=None):
    """Mask of notes far from the robust local reference (iterated). Returns (flag, ref)."""
    window = window or p.window
    n = len(beats)
    flag = np.zeros(n, bool)
    ref = None
    for _ in range(p.detect_iters):
        ref = running_consensus(beats, flag, window, p.cluster_tol_beats)
        new = np.abs(beats - ref) > p.far_beats
        if protect is not None:
            new &= ~protect
        if np.array_equal(new, flag):
            break
        flag = new
    if p.merge_gap > 0 and flag.any():
        idx = np.flatnonzero(flag)
        for a, b in zip(idx[:-1], idx[1:]):
            if 1 < b - a <= p.merge_gap + 1:
                flag[a:b] = True
    return flag, ref


def _runs(mask):
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(int), 0]))
    return list(zip(e[::2], e[1::2]))


# ----------------------------------repair-------------------------------------------

def _local_tempo(beats, onsets, good, p):
    """Beats/sec at every note, estimated from un-flagged notes only."""
    b, t = beats[good], onsets[good]
    db, dt = np.diff(b), np.diff(t)
    ok = (dt > p.chord_tol_sec) & (db > 0) & (db <= p.far_beats)
    rates, rt = db[ok] / dt[ok], t[1:][ok]
    if rates.size == 0:
        return np.full(len(beats), p.default_tempo_bps)
    glob = float(np.median(rates))
    lo = np.searchsorted(rt, onsets - p.tempo_window_sec / 2)
    hi = np.searchsorted(rt, onsets + p.tempo_window_sec / 2)
    return np.array([np.median(rates[a:z]) if z - a >= 3 else glob for a, z in zip(lo, hi)])


def _choose_split(t, pitch_g, pred_a, pred_c, score, p):
    """Return k: notes [:k] continue A, notes [k:] belong to C."""
    n = len(t)
    prefer_next = p.transition_default == "next"
    if score is None or pitch_g is None:
        return 0 if prefer_next else n
    ma = score.matches(pitch_g, pred_a, p.pitch_match_tol_beats).astype(int)
    mc = score.matches(pitch_g, pred_c, p.pitch_match_tol_beats).astype(int)
    cum_a, cum_c = np.r_[0, np.cumsum(ma)], np.r_[0, np.cumsum(mc)]
    total = (cum_a + (cum_c[-1] - cum_c)).astype(float)
    valid = np.ones(n + 1, bool)
    valid[1:n] = np.diff(t) > p.chord_tol_sec      # never split inside a chord
    total[~valid] = -np.inf
    return int(np.argmax(total)) if prefer_next else int(n - np.argmax(total[::-1]))


def _repair(beats, onsets, pitches, tempo, s, e, score, p):
    n = len(beats)
    t = onsets[s:e]
    pg = pitches[s:e] if pitches is not None else None
    a_i = s - 1 if s > 0 else None
    c_i = e if e < n else None
    if a_i is None and c_i is None:
        return None, None
    pred_a = beats[a_i] + tempo[a_i] * (t - onsets[a_i]) if a_i is not None else None
    pred_c = beats[c_i] - tempo[c_i] * (onsets[c_i] - t) if c_i is not None else None

    if a_i is None:
        kind, new = "leading", pred_c
    elif c_i is None:
        kind, new = "trailing", pred_a
    else:
        b_a, b_c, t_a, t_c = beats[a_i], beats[c_i], onsets[a_i], onsets[c_i]
        reach = b_a + tempo[a_i] * (t_c - t_a) * p.tempo_ratio_max + p.return_tol_beats
        if b_a - p.return_tol_beats <= b_c <= reach:
            kind = "return"
            if p.return_mode == "snap":
                new = np.where(t - t_a <= t_c - t, b_a, b_c).astype(float)
            else:
                frac = (t - t_a) / max(t_c - t_a, 1e-9)
                new = b_a + frac * (b_c - b_a)
        else:
            kind = "transition"
            k = _choose_split(t, pg, pred_a, pred_c, score, p)
            new = np.r_[pred_a[:k], pred_c[k:]]

    if score is not None:
        new = np.clip(new, score.lo, score.hi)
        if p.snap_to_score_onsets:
            new = score.snap(new, pg, p.pitch_match_tol_beats)
    return kind, np.asarray(new, float)


# -----------------------------------main entry------------------------------------------ 

def clean_warping_path(warping_path, onsets, pitches=None, score_beats=None,
                       score_pitches=None, score_ids=None, params=None):
    """
    warping_path : score beats, one per performance note (HMM output)
    onsets       : performance onset times (s), same length, non-decreasing
    pitches      : performance MIDI pitches (optional; enables pitch-based splitting)
    score_*      : score note beats / pitches / ids (optional)

    Returns (clean_beats, changed_mask, report, new_ids).
    """
    p = params or CleanupParams()
    beats = np.asarray(warping_path, float).copy()
    onsets = np.asarray(onsets, float)
    pitches = np.asarray(pitches, int) if pitches is not None else None
    if len(beats) != len(onsets):
        raise ValueError("warping_path and onsets must have the same length")
    if np.any(np.diff(onsets) < 0):
        raise ValueError("onsets must be sorted (sort pna by onset before aligning)")
    score = (ScoreIndex(score_beats, score_pitches, score_ids)
             if score_beats is not None and score_pitches is not None else None)

    original = beats.copy()
    report = []

    def _apply(flag, stage, pitch_gate):
        tempo = _local_tempo(beats, onsets, ~flag, p)
        for s, e in _runs(flag):
            kind, new = _repair(beats, onsets, pitches, tempo, s, e, score, p)
            if kind is None:
                continue
            entry = dict(stage=stage, kind=kind, start=int(s), end=int(e),
                         t_start=float(onsets[s]), t_end=float(onsets[e - 1]),
                         old_beats=beats[s:e].copy(), new_beats=new)
            if pitch_gate:
                tol = p.pitch_match_tol_beats
                here = score.matches(pitches[s:e], beats[s:e], tol).mean()
                home = score.matches(pitches[s:e], new, tol).mean()
                entry.update(pitch_match_excursion=float(here), pitch_match_home=float(home))
                if home < here - p.pitch_margin:
                    entry["kind"] = "kept (pitches support excursion)"
                    report.append(entry)
                    continue
            report.append(entry)
            beats[s:e] = new

    # stage 1: short excursions, removed outright
    flag1, _ = detect_excursions(beats, p, p.window)
    _apply(flag1, 1, pitch_gate=False)

    # stage 2: longer excursions, removed only if pitches agree
    if p.long_window and score is not None and pitches is not None:
        flag2, _ = detect_excursions(beats, p, p.long_window)
        _apply(flag2, 2, pitch_gate=True)

    changed = ~np.isclose(beats, original)
    new_ids = None
    if score is not None and pitches is not None:
        new_ids = [None] * len(beats)
        idx = np.flatnonzero(changed)
        for i, sid in zip(idx, score.nearest_ids(beats[idx], pitches[idx])):
            new_ids[i] = sid
    return beats, changed, report, new_ids

######################################################################################