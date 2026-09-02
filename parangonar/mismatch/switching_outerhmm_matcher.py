from typing import Any, Dict, List, Optional, Union
import numpy as np
import os

import partitura as pt
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="partitura")

from partitura.utils.music import expand_grace_notes_from_local_grace_order, remove_double_notes_from_score_note_array

from partitura.score import ScoreLike
from partitura.performance import PerformanceLike

from ..prob.switching_outerhmm import SwitchSnapOuterHMM

from ..mismatch.switching_outerhmm_utils import (
    calc_avg_notes_per_measure, 
    look_for_equivalent_score_ids
)

class SwitchingOuterHMMMatcher(object):
    def __init__(
        self,
        transitions: Optional[List[tuple[int, float]]] = None,
        pitch_error_probs: Optional[dict[str, float]] = None,
        S: Optional[np.ndarray] = None,
        r: Optional[np.ndarray] = None,
        resumption_type: str = "downbeat",
        other_prob: float = 1e-6,
        ioi_num: int = 8,
        hesitation_jump_back: int = 9,
        hesitation_jump_forward: int = 2,
        hesitation_from_avg_ioi: bool = True,
        hesitation_ioi_ratio_threshold: float = 2,
        hesitation_from_pitch_errors: bool = True,
        neighbourhood_range: int = 2,
        sigma: float = 4.0,
        gamma: float = np.log(10),
        score_metadata_folder_path: Optional[str] = None,
        score_identifier: Optional[str] = None,
        output_dir: Optional[str] = None,
        consider_parallel_sections: bool = True,
        section_omit_reason: Optional[str] = None
        ) -> None:

        '''
        Refer to parangonar/prob/switching_outerhmm.py for details on the parameters.

        score_metadata_folder: Use this to provide a path to a folder to store pre-computed score metadata.
            This is especially useful for multiple runs of the matcher on the same score. 
            If provided, the matcher will look for a pre-computed score metadata file in this folder before computing it again.
            If there is no existing score metadata file, it will compute the score metadata and save it to this folder for future use.

        score_identifier: A unique identifier for the score. This is used to name/load the pre-computed score metadata file.
            A score_identifier is required if score_metadata_folder_path is provided.

        output_dir: Optional path to a directory where the matcher can save alignment files. Provide this to store a match file that stores section and omitted section information. If not provided, the matcher will not save any alignment files.
        
        consider_parallel_sections: Boolean flag to indicate whether to align performance sections to multiple score sections that are musically identical.
        
        section_omit_reason: Optional string to provide a reason for omitting a section. This is used when saving the match file to indicate why a section was omitted. If not provided, "not_performed" will be saved.
        '''

        self.transitions = transitions
        self.pitch_error_probs = pitch_error_probs
        self.S = S
        self.r = r
        self.resumption_type = resumption_type
        self.other_prob = other_prob
        self.ioi_num = ioi_num
        self.hesitation_jump_back = hesitation_jump_back
        self.hesitation_jump_forward = hesitation_jump_forward
        self.hesitation_from_avg_ioi = hesitation_from_avg_ioi
        self.hesitation_ioi_ratio_threshold = hesitation_ioi_ratio_threshold
        self.hesitation_from_pitch_errors = hesitation_from_pitch_errors
        self.neighbourhood_range = neighbourhood_range
        self.sigma = sigma
        self.gamma = gamma

        # Score metadata folder for caching pre-computed score metadata.
        self.score_metadata_folder = score_metadata_folder_path
        self.score_identifier = score_identifier

        self.output_dir = output_dir
        self.section_omit_reason = section_omit_reason

        self.consider_parallel_sections = consider_parallel_sections

        if self.score_metadata_folder is not None and self.score_identifier is None:
            raise ValueError(
                "score_identifier is required if score_metadata_folder_path is provided."
            )

    def __call__(
        self,
        score: Union[str, ScoreLike],
        performance: Union[str, PerformanceLike],
    ) -> List[Dict[str, Any]]:

        if isinstance(score, str):
            score = pt.load_score(score)

        elif isinstance(score, np.ndarray):
            raise ValueError(
                "Score input as partiture note_array (np.ndarray) is not supported for SwitchingOuterHMMMatcher. Please provide a score file path or a Partitura ScoreLike object."
            )
            return

        score_part = score.parts[0]
        score_part = pt.score.unfold_part_maximal(score_part)
        score_measure_number_map = score_part.measure_number_map
        print("Expanding grace notes in the score...")
        sna = expand_grace_notes_from_local_grace_order(score_part, grace_offset_quarter=1/4, include_metrical_position=True)
        print("Removing double notes from the score note array...")
        sna = remove_double_notes_from_score_note_array(sna)

        if isinstance(performance, str):
            performance = pt.load_performance_midi(performance, merge_tracks=True)

        elif isinstance(performance, np.ndarray):
            raise ValueError(
                "Performance input as partiture note_array (np.ndarray) is not supported for SwitchingOuterHMMMatcher. Please provide a performance MIDI file path or a Partitura PerformanceLike object."
            )
            return

        ppq = performance.performedparts[0].ppq
        mpq = performance.performedparts[0].mpq
        pna = performance.note_array()


        avg_notes_per_measure = calc_avg_notes_per_measure(sna)
        min_diagonal_length = 8 * avg_notes_per_measure
        num_diagonals_limit = round(len(sna) / (min_diagonal_length * 2))
        optimum_diagonal_length_found = False

        if self.score_metadata_folder is not None and self.score_identifier is not None:
            if not os.path.exists(self.score_metadata_folder):
                os.makedirs(self.score_metadata_folder)

            for score_metadata_file in os.listdir(self.score_metadata_folder):
                if score_metadata_file.startswith("."):
                    continue  # Skip hidden files
                if score_metadata_file.endswith(".npz"):
                    metadata_score_identifier = score_metadata_file.split(".")[0].split("_metadata")[0]
                    if metadata_score_identifier == self.score_identifier:
                        print(f"Found existing score metadata file {score_metadata_file} for {self.score_identifier}. Loading metadata...")
                        metadata_fn = os.path.join(self.score_metadata_folder, score_metadata_file)
                        loaded_metadata = np.load(metadata_fn, allow_pickle=True)
                        ids_association_dict = loaded_metadata['ids_association_dict'].item()
                        minimum_ref_id_dict = loaded_metadata['minimum_ref_id_dict'].item()
                        onset_beat_associations_dict = loaded_metadata['onset_beat_associations_dict'].item()
                        min_ref_onset_beat_dict = loaded_metadata['min_ref_onset_beat_dict'].item()
                        diagonals_beats_to_num_dict = loaded_metadata['diagonals_beats_to_num_dict'].item()
                        diagonal_borders_dict = loaded_metadata['diagonal_borders_dict'].item()
                        num_diagonals = loaded_metadata['num_diagonals'].item()
                        optimum_diagonal_length_found = True
                        print("Metadata loaded successfully.")
                        break

        if not optimum_diagonal_length_found:
            print("No existing metadata found for this score. Starting diagonal length optimization process...")
            while not optimum_diagonal_length_found:
                ids_association_dict, minimum_ref_id_dict, onset_beat_associations_dict, min_ref_onset_beat_dict, diagonals_beats_to_num_dict, diagonal_borders_dict, num_diagonals = look_for_equivalent_score_ids(sna, min_diagonal_length=min_diagonal_length, num_diagonals_limit=num_diagonals_limit)
                if num_diagonals == None:
                    min_diagonal_length = int(min_diagonal_length * 1.5)
                elif num_diagonals == 0:
                    min_diagonal_length = int(min_diagonal_length / 1.5)
                    num_diagonals_limit = round(len(sna) / (min_diagonal_length * 2))
                else:
                    optimum_diagonal_length_found = True

        if self.score_metadata_folder is not None and self.score_identifier is not None:
            # Save the metadata for future use
            metadata_fn = os.path.join(self.score_metadata_folder, f"{self.score_identifier}_metadata.npz")
            np.savez(metadata_fn, ids_association_dict=ids_association_dict, minimum_ref_id_dict=minimum_ref_id_dict, onset_beat_associations_dict=onset_beat_associations_dict, min_ref_onset_beat_dict=min_ref_onset_beat_dict, diagonals_beats_to_num_dict=diagonals_beats_to_num_dict, diagonal_borders_dict=diagonal_borders_dict, num_diagonals=num_diagonals)
            print(f"Metadata saved to {metadata_fn}.")

        self.switchSnapOuterHMM = SwitchSnapOuterHMM(
            reference_features=sna,
            performance_note_array=pna,
            score_measure_number_map=score_measure_number_map,
            transitions=self.transitions,
            pitch_error_probs=self.pitch_error_probs,
            S=self.S,
            r=self.r,
            resumption_type=self.resumption_type,
            other_prob=self.other_prob,
            ioi_num=self.ioi_num,
            hesitation_jump_back=self.hesitation_jump_back,
            hesitation_jump_forward=self.hesitation_jump_forward,
            hesitation_from_avg_ioi=self.hesitation_from_avg_ioi,
            hesitation_ioi_ratio_threshold=self.hesitation_ioi_ratio_threshold,
            hesitation_from_pitch_errors=self.hesitation_from_pitch_errors,
            neighbourhood_range=self.neighbourhood_range,
            sigma=self.sigma,
            gamma=self.gamma,
            evaluate_post_processed_alignment=True,
            ids_association_dict=ids_association_dict,
            minimum_ref_id_dict=minimum_ref_id_dict,
            onset_beat_associations_dict=onset_beat_associations_dict,
            min_ref_onset_beat_dict=min_ref_onset_beat_dict,
            diagonals_beats_to_num_dict=diagonals_beats_to_num_dict,
            diagonal_borders_dict=diagonal_borders_dict,
            average_notes_per_measure=avg_notes_per_measure,
            section_omit_reason=self.section_omit_reason,
        )

        alignment, alignment_dict = self.switchSnapOuterHMM.run()

        print("Post-processing alignment to clean quick to-fro jumps...")
        processed_alignment, processed_alignment_dict = self.switchSnapOuterHMM.clean_quick_to_fro_jumps()
                        
        snapped_alignment, snapped_alignment_dict = self.switchSnapOuterHMM.snap_to_most_likely_diagonal()

        output_alignment = snapped_alignment

        if self.consider_parallel_sections:
            print("Creating alignment with musically identical segments...")
            self.switchSnapOuterHMM.create_parallel_alignment()
            output_alignment = self.switchSnapOuterHMM.parallel_alignment

        print("Creating sections. The section lines and omitted section lines will only be printed if you save the match file using the output_dir parameter.")
        sections = self.switchSnapOuterHMM.create_section_lines()
        omitted_sections = self.switchSnapOuterHMM.create_omitted_section_lines()

        if self.output_dir is not None:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            self.switchSnapOuterHMM.save_parangonada_csv(
                self.output_dir,     
            )

            pt.save_match(
                alignment=output_alignment,
                performance_data=performance,
                score_data=score_part,
                out=os.path.join(self.output_dir, f"{self.score_identifier}_parallel_alignment.match"),
                mpq=mpq,
                ppq=ppq,
                sections=sections,
                omitted_sections=omitted_sections,
            )

        print("Alignment complete!")

        return output_alignment