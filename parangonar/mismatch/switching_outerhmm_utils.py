import numpy as np

def create_match_from_alignment_dict(alignment_dict, sids):
    pids = list(alignment_dict.keys())
    pids_int = [int(pid[1:]) for pid in pids]
    pids_int.sort()
    match = []
    for pid_int in pids_int:
        pid = f"n{pid_int}"
        sid = alignment_dict.get(pid)
        if sid is not None:
            match.append({
                "label": "match",
                "score_id": sid,
                "performance_id": pid,
            })
        else:
            match.append({
                "label": "insertion",
                "performance_id": pid,
            })

    for sid in sids:
        if sid not in alignment_dict.values():
            match.append({
                "label": "deletion",
                "score_id": sid,
            })
    return match

def calc_avg_notes_per_measure(sna):
    notes_per_measure_list = []
    still_measure_beat = True
    note_count = 0

    for idx, row in enumerate(sna):
        if row['onset_beat'] < 0:
            continue

        if row['rel_onset_div'] == 0:
            if row['is_grace']:
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
    unique_onsets = np.unique(sna['onset_beat'])
    onsets_per_measure_list = []
    onset_count = 0
    num_of_measures = 0

    for idx, onset in enumerate(unique_onsets):
        if onset < 0:
            continue

        row = sna[sna['onset_beat'] == onset][0]
        is_grace = row['is_grace']
        is_measure_beat = row['rel_onset_div'] == 0

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
    s_pitches = sna['pitch']
    score_ssm = np.zeros((len(s_pitches), len(s_pitches)))
    for i in range(len(s_pitches)):
        for j in range(len(s_pitches)):
            score_ssm[i, j] = 1 if s_pitches[i] == s_pitches[j] else 0

    diagonals = find_diagonal_runs_with_coords(score_ssm, min_length=min_diagonal_length)
    filtered_diagonals = remove_subset_runs(diagonals)
    accepted_diagonals = remove_overlaps_with_simple_trimming(filtered_diagonals, min_length=min_diagonal_length)

    if len(accepted_diagonals) > num_diagonals_limit:
        return None, None, None, None, None, None, None
    elif len(accepted_diagonals) == 0:
        return None, None, None, None, None, None, 0
    
    print(f"Found {len(accepted_diagonals) + 1} accepted diagonals with min_diagonal_length={min_diagonal_length}.")

    diagonal_borders_dict = dict()

    diagonal_num = 0
    expanded_accepted_diagonals = []
    diagonals_beats_to_num_dict = dict()
    for diagonal in accepted_diagonals:
        diag1_start = diagonal[0][0]
        diag1_start_beat = sna['onset_beat'][diag1_start]
        diag1_end = diagonal[-1][0]
        diag1_end_beat = sna['onset_beat'][diag1_end]
        diag2_start = diagonal[0][1]
        diag2_start_beat = sna['onset_beat'][diag2_start]
        diag2_end = diagonal[-1][1]
        diag2_end_beat = sna['onset_beat'][diag2_end]
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
            
            ref_score_id = sna['id'][ref_score_id_coord]
            equivalent_score_id = sna['id'][equivalent_score_id_coord]
            ref_score_id_dict[equivalent_score_id] = ref_score_id

    minimum_ref_id_dict = dict()
    min_ref_onset_beat_dict = dict()
    for equiv_key in ref_score_id_dict.keys():
        temp_ref_id = ref_score_id_dict[equiv_key]
        while temp_ref_id in ref_score_id_dict.keys():
            temp_ref_id = ref_score_id_dict[temp_ref_id]
        minimum_ref_id_dict[equiv_key] = temp_ref_id
        equiv_beat = sna[sna['id'] == equiv_key]['onset_beat'][0]
        ref_beat = sna[sna['id'] == temp_ref_id]['onset_beat'][0]
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
            

    return ids_association_dict, minimum_ref_id_dict, onset_beat_associations_dict, min_ref_onset_beat_dict, diagonals_beats_to_num_dict, diagonal_borders_dict, len(accepted_diagonals)
