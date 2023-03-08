
from partitura.utils.music import (ensure_notearray)

from partitura.musicanalysis.performance_codec import ( 
                                    to_matched_score,
                                    get_unique_onset_idxs
                                    )

import numpy as np
import os
from scipy.interpolate import interp1d

################################### PARANGONADA EXPORT ###################################


def alignment_dicts_to_array(alignment):
    """
    create structured array from list of dicts type alignment.

    Parameters
    ----------
    alignment : list
        A list of note alignment dictionaries.

    Returns
    -------
    alignarray : structured ndarray
        Structured array containing note alignment.
    """
    fields = [
        ("idx", "i4"),
        ("matchtype", "U256"),
        ("partid", "U256"),
        ("ppartid", "U256"),
    ]

    array = []
    # for all dicts create an appropriate entry in an array:
    # match = 0, deletion  = 1, insertion = 2
    for no, i in enumerate(alignment):

        if i["label"] == "match":
            array.append((no, "0", i["score_id"], str(i["performance_id"])))
        elif i["label"] == "insertion":
            array.append((no, "2", "undefined", str(i["performance_id"])))
        elif i["label"] == "deletion":
            array.append((no, "1", i["score_id"], "undefined"))

    alignarray = np.array(array, dtype=fields)

    return alignarray


def save_parangonada_csv(
    alignment,
    performance_data,
    score_data,
    outdir = None,
    zalign = None,
    feature = None
):
    """
    Save an alignment for visualization with parangonda.

    Parameters
    ----------
    alignment : list
        A list of note alignment dictionaries.
    performance_data : Performance, PerformedPart, structured ndarray
        The performance information
    score_data : ScoreLike
        The musical score. A :class:`partitura.score.Score` object,
        a :class:`partitura.score.Part`, a :class:`partitura.score.PartGroup` or
        a list of these.
    outdir : PathLike
        A directory to save the files into.
    ppart : PerformedPart, structured ndarray
        A PerformedPart or its note_array.
    zalign : list, optional
        A second list of note alignment dictionaries.
    feature : list, optional
        A list of expressive feature dictionaries.

    Returns
    -------
    perf_note_array : np.ndarray
        The performance note array. Only returned if `outdir` is None.
    score_note_array: np.ndarray
        The note array from the score. Only returned if `outdir` is None.
    alignarray: np.ndarray
    zalignarray: np.ndarray
    featurearray: np.ndarray
    """

    score_note_array = ensure_notearray(score_data)

    perf_note_array = ensure_notearray(performance_data)

    ffields = [
        ("velocity", "<f4"),
        ("timing", "<f4"),
        ("articulation", "<f4"),
        ("id", "U256"),
    ]

    farray = []
    notes = list(score_note_array["id"])
    if feature is not None:
        # veloctiy, timing, articulation, note
        for no, i in enumerate(list(feature["id"])):
            farray.append(
                (
                    feature["velocity"][no],
                    feature["timing"][no],
                    feature["articulation"][no],
                    i,
                )
            )
    else:
        for no, i in enumerate(notes):
            farray.append((0, 0, 0, i))

    featurearray = np.array(farray, dtype=ffields)
    alignarray = alignment_dicts_to_array(alignment)

    if zalign is not None:
        zalignarray = alignment_dicts_to_array(zalign)
    else:  # if no zalign is available, save the same alignment twice
        zalignarray = alignment_dicts_to_array(alignment)

    if outdir is not None:
        np.savetxt(
            os.path.join(outdir, "ppart.csv"),
            # outdir + os.path.sep + "perf_note_array.csv",
            perf_note_array[
                [
                    "onset_sec",
                    "duration_sec",
                    "pitch",
                    "velocity",
                    "track",
                    "channel",
                    "id",
                ]
            ],
            fmt="%.20s",
            delimiter=",",
            header=",".join(
                [
                    "onset_sec",
                    "duration_sec",
                    "pitch",
                    "velocity",
                    "track",
                    "channel",
                    "id",
                ]
            ),
            comments="",
        )
        np.savetxt(
            os.path.join(outdir, "part.csv"),
            # outdir + os.path.sep + "score_note_array.csv",
            score_note_array,
            fmt="%.20s",
            delimiter=",",
            header=",".join(score_note_array.dtype.names),
            comments="",
        )
        np.savetxt(
            os.path.join(outdir, "align.csv"),
            # outdir + os.path.sep + "align.csv",
            alignarray,
            fmt="%.20s",
            delimiter=",",
            header=",".join(alignarray.dtype.names),
            comments="",
        )
        np.savetxt(
            os.path.join(outdir, "zalign.csv"),
            # outdir + os.path.sep + "zalign.csv",
            zalignarray,
            fmt="%.20s",
            delimiter=",",
            header=",".join(zalignarray.dtype.names),
            comments="",
        )
        np.savetxt(
            os.path.join(outdir, "feature.csv"),
            # outdir + os.path.sep + "feature.csv",
            featurearray,
            fmt="%.20s",
            delimiter=",",
            header=",".join(featurearray.dtype.names),
            comments="",
        )
    else:
        return (
            perf_note_array,
            score_note_array,
            alignarray,
            zalignarray,
            featurearray,
        )


################################### ANCHOR POINT GENERATION ###################################


def node_array(part, 
                ppart,
                alignment,
                tapping_noise=False,
                node_interval=1, 
                start_beat=None,
                nodes_in_beats=True):
    """
    generates an array of nodes, corresponding score time point and 
    performance time point tuples, spacing given by note_interval.
    
    Args:
        part (partitura.Part): a part object
        ppart (partitura.PerformedPart): a performedpart object
        alignment (List(Dict)): an alignment

    Returns:
        np.ndarray: a minimal, aligned 
            node note array.

    """

    ppart.sustain_pedal_threshold = 128

    matched_score, snote_ids = to_matched_score(part, ppart, alignment)

    # get unique onsets
    nid_dict = dict((n.id, i) for i, n in enumerate(part.notes_tied))
    matched_subset_idxs = np.array([nid_dict[nid] for nid in snote_ids])

    part_note_array = ensure_notearray(part,
                                        include_time_signature=True)

    score_onsets = part_note_array[matched_subset_idxs]['onset_beat']
    # changed from onset_beat
    unique_onset_idxs = get_unique_onset_idxs(score_onsets)

    smatched_score_onset = notewise_to_onsetwise(matched_score['onset'],
                                                    unique_onset_idxs,
                                                    aggregation_func=np.mean)
    pmatched_score_onset = notewise_to_onsetwise(matched_score['p_onset'],
                                                    unique_onset_idxs,
                                                    aggregation_func=np.mean)
    
    if nodes_in_beats:
        beat_times, sbeat_times = beat_times_from_matched_score(
            smatched_score_onset, 
            pmatched_score_onset,
            tapping_noise=tapping_noise,
            node_interval=node_interval,
            start_beat=start_beat)
    else: 
        beat_times, sbeat_times = measure_times_from_matched_score(
            smatched_score_onset, 
            pmatched_score_onset, 
            part, 
            tapping_noise=tapping_noise,
            measure_interval=node_interval,
            start_measure=start_beat)

    alignment_times = [*zip(sbeat_times, beat_times)]

    return alignment_times


def beat_times_from_matched_score(x, y,
                                  tapping_noise=False,
                                  node_interval=1.0,
                                  start_beat=None):

    beat_times_func = interp1d(x, y, kind="linear", fill_value="extrapolate")

    min_beat = np.ceil(x.min())  # -node_interval
    min_beat_original = min_beat
    max_beat = np.floor(x.max())  # +node_interval

    if start_beat is not None:
        print("first node at ", start_beat, " first beat at ", min_beat)
        min_beat = start_beat
        
    sbeat_times = np.arange(min_beat, max_beat, node_interval)
    # prepend and append very low and high values to make sure every beat/beat annotation falls between two sbeat_times/beat_times
    sbeat_times = np.append(np.append(min_beat_original-max(100, node_interval),sbeat_times),max_beat+max(100, node_interval))
    beat_times = beat_times_func(sbeat_times)

    if tapping_noise:
        beat_times += np.random.normal(0.0, tapping_noise,
                                       beat_times.shape)

    return beat_times, sbeat_times


def measure_times_from_matched_score(x, y, part, 
                                  tapping_noise=False,
                                  measure_interval=1,
                                  start_measure=None):

    beat_times_func = interp1d(x, y, kind="linear", fill_value="extrapolate")

    measure_times_in_part = [part.beat_map(measure.start.t) for measure in part.iter_all(partitura.score.Measure)]
    if start_measure is None:
        start_measure=0
    
    smeasure_idx = np.arange(start_measure, len(measure_times_in_part), measure_interval, dtype = int)
    smeasure_times = np.array(measure_times_in_part)[smeasure_idx]
    # prepend and append very low and high values to make sure every beat/beat annotation falls between two sbeat_times/beat_times
    smeasure_times = np.append(np.append(smeasure_times[0]-max(100, measure_interval),smeasure_times),smeasure_times[-1]+max(100, measure_interval))
    pmeasure_times = beat_times_func(smeasure_times)

    if tapping_noise:
        pmeasure_times += np.random.normal(0.0, tapping_noise, pmeasure_times.shape)

    return pmeasure_times, smeasure_times


def notewise_to_onsetwise(notewise_inputs, 
                          unique_onset_idxs, 
                          aggregation_func=np.mean):
    """Agregate onset times per score onset
    """
    
    onsetwise_inputs = np.zeros(len(unique_onset_idxs),
                                dtype=notewise_inputs.dtype)

    for i, uix in enumerate(unique_onset_idxs):
        onsetwise_inputs[i] = aggregation_func(notewise_inputs[uix])
    return onsetwise_inputs


def expand_grace_notes(note_array, backwards_time=0.2):
    """
    expand the duration of gracenotes in a note_array and reset their onset by a timespan called backwards_time
    """
    grace_note_onsets = np.unique(
        note_array["onset_beat"][note_array["duration_beat"] == 0.0])
    for gno in grace_note_onsets:
        mask = np.all([note_array["duration_beat"] == 0.0,
                       note_array["onset_beat"] == gno],
                      axis=0)
        number_of_grace_notes = mask.sum()
        note_array["onset_beat"][mask] -= np.linspace(backwards_time,0.0, number_of_grace_notes+1)[:-1]
        note_array["duration_beat"][mask] += backwards_time/(number_of_grace_notes)
    return note_array


def convert_grace_to_insertions(alignment):
    """
    relabel all ornament alignments as insertions
    """
    new_alignment = [al for al in alignment if al["label"] != "ornament"]
    new_alignment_o = [al for al in alignment if al["label"] == "ornament"]
    for al in new_alignment_o:
        new_al = {'label': 'insertion', 'performance_id': al["performance_id"]}
        new_alignment.append(new_al)
    return new_alignment

