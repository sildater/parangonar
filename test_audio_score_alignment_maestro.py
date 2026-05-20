import pandas as pd
from pathlib import Path
import shutil
import os
import parangonar as pa
import partitura as pt
import scipy
import numpy as np
from scipy import signal
from joblib import Parallel, delayed
from itertools import repeat
from scipy.ndimage import median_filter, maximum_filter
from sklearn.linear_model import LinearRegression
from numba import jit
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore",module="partitura")

@jit(nopython=True)
def elastic_forward_and_backward_pitch_onset(
    onsets,
    spec,
    spikes,
    pitch_sets,
    bp_init,
    max_stretch_longer,
    max_stretch_shorter,
    stretch_cost,
    spec_cost,
    spike_cost,
    max_bp,
    min_bp,
    alpha,
    cost_threshold
):
    """
    
    Parameters
    ----------

    spikes: np.ndarray
        temporal positions spikes to fit to the activations
    
    Returns
    -------
    dtwd : np.ndarray
        Accumulated cost matrix
    path: np.ndarray
        backtracked path
    """
    # Initialize arrays and helper variables
    M = spikes.shape[0]
    N = onsets.shape[1]
    spike_period = np.diff(spikes)
    
    # accumulated cost matrix is initialized with INFINITY
    D = np.ones((M, N), dtype=float) * np.inf
    # Backtracking
    B = np.ones((M, N), dtype=np.int64) * -1
    # keep track of previous beat period
    BP = np.ones((M, N), dtype=float) * bp_init
    # 
    D[0,0] = 0  
    max_bp = float(max_bp)
    min_bp = float(min_bp)
    spec_slice_len = 7
   
    
    for i in np.arange(0, M-1, dtype=np.int64 ): # loop over spikes
        pitches = pitch_sets[i+1]
        
        for j in np.arange(0, N-1, dtype=np.int64): # loop over activations
            if D[i,j] < np.inf:
                beat_period = BP[i,j]
                lower_bound = max(min(j + np.floor(beat_period * (1-max_stretch_shorter) * spike_period[i] ), N - 1), j +  1)
                upper_bound = max(min(j + np.ceil(beat_period * (1+max_stretch_longer)) * spike_period[i]  + 1, N), lower_bound + 1)
                candidate_slice = np.arange(lower_bound,upper_bound, dtype = np.int64)
                
                # prepare the slice of the spectrongram and flux to be checked
                slice_arr = D[i, candidate_slice]
                D_vals = np.empty((len(pitches), len(slice_arr)), dtype=slice_arr.dtype)
                for pitch_rows in range(len(pitches)):
                    D_vals[pitch_rows] = slice_arr
                    
                for pitch_idx, pitch in enumerate(pitches):
                    #print("*"*30)
                    #print("candidate_idx_glob", candidate_idx_glob, "candidate_onsets", candidate_onsets, "candidate_slice", candidate_slice)
                    for candidate_idx, candidate_j in enumerate(candidate_slice):
                        
                        # compute the stretch
                        stretch = max((candidate_j - j) / (beat_period * spike_period[i]),  (beat_period * spike_period[i]) / (candidate_j - j))
                        stretch_c =  min((stretch**2 - 1), 1)
                        # print("stretch", stretch, stretch_c, "at idx", candidate_j)
                        # onset activation
                        activation = onsets[pitch,candidate_j] 
                        # spec slice
                        spec_slice = spec[pitch,candidate_j: candidate_j + spec_slice_len]
                        spec_fit = np.min(spec_slice)

                        # total cost for this candidate
                        candidate_j_cost = D[i,j] + spike_cost * activation + stretch_cost * stretch_c + spec_fit * spec_cost
                        # print("i,j",i,j,"pitch",pitch,"candidate_j",candidate_j, "candidate_j_cost",candidate_j_cost, 
                        #       "activation",activation, "stretch_c",stretch_c, "spec_fit", spec_fit)
                        D_vals[pitch_idx,candidate_idx] = candidate_j_cost
                        

                for candidate_j_idx, candidate_j in enumerate(candidate_slice):
                    candidate_j_cost = np.min(D_vals[:,candidate_j_idx])
                    if D[i+1,candidate_j] > candidate_j_cost:
                        D[i+1,candidate_j] = candidate_j_cost
                        BP[i+1,candidate_j] = (1-alpha) * beat_period + alpha * min(max(float(candidate_j - j) / spike_period[i] , min_bp), max_bp)
                        B[i+1,candidate_j] = j

        min_cost_at_idx = np.min(D[i+1,:])
        mask_large_cost = D[i+1,:] > min_cost_at_idx + cost_threshold
        D[i+1,mask_large_cost] = np.inf
                
   
    # simple backtracking
    spikes = [N - 1]
    for backwards_i in range( M - 1, 0, -1):
        prev_spike = B[backwards_i,spikes[-1]]
        spikes.append(prev_spike)
    return D, B, BP, spikes[::-1]


# use different pitched onset fluxes
@jit(nopython=True)
def elastic_forward_and_backward_pitch_onset_limit(
    onsets,
    spec,
    spikes,
    pitch_sets,
    bp_init,
    max_stretch_longer,
    max_stretch_shorter,
    stretch_cost,
    spec_cost,
    spike_cost,
    max_bp,
    min_bp,
    alpha,
    spec_slice_len = 7,
    cost_threshold = 2,
     candidate_onset_number = 3
    
):
    """
    version that only does a couple of points
    
    Parameters
    ----------

    spikes: np.ndarray
        temporal positions spikes to fit to the activations
    
    Returns
    -------
    dtwd : np.ndarray
        Accumulated cost matrix
    path: np.ndarray
        backtracked path
    """
    # Initialize arrays and helper variables
    M = spikes.shape[0]
    N = onsets.shape[1]
    spike_period = np.diff(spikes)
    
    # accumulated cost matrix is initialized with INFINITY
    D = np.ones((M, N), dtype=float) * np.inf
    # Backtracking
    B = np.ones((M, N), dtype=np.int64) * -1
    # keep track of previous beat period
    BP = np.ones((M, N), dtype=float) * bp_init
    # 
    D[0,0] = 0  
    max_bp = float(max_bp)
    min_bp = float(min_bp)
    
    
    for i in np.arange(0, M-1, dtype=np.int64 ): # loop over spikes
        pitches = pitch_sets[i+1]
        # print(pitches)
        for j in np.arange(0, N-1, dtype=np.int64): # loop over activations
            if D[i,j] < np.inf:
                beat_period = BP[i,j]
                lower_bound = max(min(j + np.floor(beat_period * (1-max_stretch_shorter) * spike_period[i] ), N - 1), j +  1)
                upper_bound = max(min(j + np.ceil(beat_period * (1+max_stretch_longer)) * spike_period[i]  + 1, N), lower_bound +  1)
                candidate_slice = np.arange(lower_bound,upper_bound, dtype = np.int64)
                
                slice_arr = D[i, candidate_slice]
                D_vals = np.empty((len(pitches), len(slice_arr)), dtype=slice_arr.dtype)
                for pitch_rows in range(len(pitches)):
                    D_vals[pitch_rows] = slice_arr
                
                for pitch_idx, pitch in enumerate(pitches):
                    for candidate_idx, candidate_j in enumerate(candidate_slice):
                        # compute the stretch
                        stretch = max((candidate_j - j) / (beat_period * spike_period[i]),  (beat_period * spike_period[i]) / (candidate_j - j))
                        stretch_c =  min((stretch**2 - 1), 1)

                        # onset activation
                        activation = onsets[pitch,candidate_j] 
                        # spec slice
                        spec_slice = spec[pitch,candidate_j: candidate_j + spec_slice_len]
                        spec_fit = np.min(spec_slice)

                        # total cost for this candidate
                        candidate_j_cost = D[i,j] + spike_cost * activation + stretch_cost * stretch_c + spec_fit * spec_cost
                        D_vals[pitch_idx,candidate_idx] = candidate_j_cost
                        
                # try to extract just the top candidates and fill it in the global mmatrix
                # softmin-like function
                D_vals_min = -np.log(np.sum(np.exp(-D_vals), axis = 0))
                sorted_idx = np.argsort(D_vals_min)
                if i + 1 != M - 1:
                    for min_idx in sorted_idx[:candidate_onset_number]:
                        min_candidate_j_cost = D_vals_min[min_idx]
                        min_candidate_j = candidate_slice[min_idx]
                        if D[i+1,min_candidate_j] > min_candidate_j_cost:
                            D[i+1,min_candidate_j] = min_candidate_j_cost
                            BP[i+1,min_candidate_j] = (1-alpha) * beat_period + alpha * min(max(float(min_candidate_j - j) / spike_period[i] , min_bp), max_bp)
                            B[i+1,min_candidate_j] = j
                else:
                    min_idx = sorted_idx[0]
                    min_candidate_j_cost = D_vals_min[min_idx]
                    min_candidate_j = N - 1
                    if D[i+1,min_candidate_j] > min_candidate_j_cost:
                        D[i+1,min_candidate_j] = min_candidate_j_cost
                        BP[i+1,min_candidate_j] = (1-alpha) * beat_period + alpha * min(max(float(min_candidate_j - j) / spike_period[i] , min_bp), max_bp)
                        B[i+1,min_candidate_j] = j


                        
                
        #min_cost_at_idx = np.min(D[i+1,:])
        #mask_large_cost = D[i+1,:] > min_cost_at_idx + cost_threshold
        #D[i+1,mask_large_cost] = np.inf

        min_cost_at_idx = np.argsort(D[i+1,:])
        D[i+1,min_cost_at_idx[cost_threshold:]] = np.inf


    # simple backtracking
    spikes = [N - 1]
    for backwards_i in range( M - 1, 0, -1):
        prev_spike = B[backwards_i,spikes[-1]]
        spikes.append(prev_spike)
    return D, B, BP, spikes[::-1]


class IIRSpect:
    """
    IIR based log freq spectrogram

    Parameters
    ----------
    sample_rate : int
        Sample rate of the input signal
    n_fft : int
        Window length for the FFT (in samples)
    hop_length : int
        Hop size for the FFT (in samples)
    f_min : float
        Lower bound of the first filter
    f_max : float
        Upper bound of the last filter
    n_bins : int
        Number of filters
    power : int
        Whether to compute the magnitudes (1) or energy (2) of the complex
        spectrogram
    log_multiplier : float
        Factor that the magnitudes are multiplied with before adding 1.0 and
        taking the logarithm
    device : str
        What device to put the module initially
    rir_prob : float
        With what probability to apply a room impulse response
    shift_prob : float
        With what probability to apply pitch shifting
    shift_max : float
        How many semitones (or fractions of semitones) to shift at most
    """
    def __init__(
        self,
        sample_rate=16000,
        n_fft=2048, # window size
        hop_length=160,
        f_min=27.5,
        f_max=4186.009,
        n_bins=88,
        power=1,
        log_multiplier=1000,
        device="cpu",
        shift_prob=0,
        shift_max=0.1,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.n_bins = n_bins
        self.hop_length = hop_length
        self.log_multiplier = log_multiplier
        self.power = power
        self.device = device
        self.view_to_the_past = hop_length - self.n_fft
        self.boundary_freqs = np.logspace(np.log2(0.5 * f_min * 2 ** (23/24)),np.log2(f_max * 2 ** (1/24) ),n_bins + 1, base = 2)
        self.center_freqs = np.logspace(np.log2(f_min),np.log2(f_max),n_bins, base = 2)
        self.filter_order = 2
        self.nyq = 0.5 * self.sample_rate
        self.filters = list()
        for i, f in enumerate(self.boundary_freqs[:-1]):
            low = f / self.nyq
            high = self.boundary_freqs[i+1] / self.nyq
            coeff_array = signal.butter(N = self.filter_order, 
                                        Wn = [low, high], 
                                        btype='band', output = 'sos')
            self.filters.append(coeff_array)



    def apply_sos_and_max_filter(self, x, coeff_array, num_windows):
        filtered_signal = signal.sosfilt(coeff_array, x)
        output_max_filt_signal = list()
        for j in range(num_windows):
            start = j * self.hop_length
            start_past = max(0,start + self.view_to_the_past)
            segment = filtered_signal[start_past:start + self.hop_length]
            output_max_filt_signal.append(np.max(np.abs(segment)))
        return np.array(output_max_filt_signal)
    
    def __call__(self, x_np):

        num_windows = len(x_np) // self.hop_length
        results = Parallel(n_jobs=-1)(
            delayed(self.apply_sos_and_max_filter)(sig, car, num_windows) 
            for sig, car, num_windows in zip(repeat(x_np), self.filters, repeat(num_windows))
        )

        
        spectrogram = np.array(results)  
        return spectrogram
    
def create_init_template(
        freq0s: np.ndarray = None, # array of f0
        freqbins: np.ndarray = None, # array of frequency bins
        number_of_harmonics: int = 12,
        harmonic_attenuation: callable = lambda x : 0.9**x,
    ):
    templates = np.zeros((len(freqbins), len(freq0s)))
    max_freq = np.max(freqbins)
    for template_id, f0 in enumerate(freq0s):
        harmonics = np.arange(1, number_of_harmonics + 1)
        f0_harmonics = harmonics * f0
        attenuated_harmonics = harmonic_attenuation(harmonics)
        for harmonic_id, f0h in enumerate(f0_harmonics):
            if f0h <= max_freq:
                fidx = np.abs(freqbins - f0h).argmin()
                templates[fidx, template_id] = attenuated_harmonics[harmonic_id]

    return templates

def nnls_batch_sklearn(Y, T):
    """
    Batched NNLS using scikit-learn's LinearRegression with positive=True.
    Solves for all columns of Y at once.
    """
    model = LinearRegression(positive=True, fit_intercept=False)
    model.fit(T, Y)  # fits all columns simultaneously
    return model.coef_.T  # shape (m, s)

def prepare_score(s_array):
    features = list()
    unique_onsets = np.unique(s_array["onset_beat"])
    # create pitch set representation
    for onset in unique_onsets:
        features.append(
            [onset, set(s_array[s_array["onset_beat"] == onset]["pitch"])]
        )

    return features, unique_onsets

def evaluate(score_note_array,
             perf_note_array,
             alignment,
             predicted_stime_to_ptime_map):
    _, groundtruth_stime_to_ptime_map = pt.musicanalysis.performance_codec.get_time_maps_from_alignment(
                    perf_note_array,
                    score_note_array,
                    alignment)
    
    unique_score_times = np.unique(score_note_array["onset_beat"])
    prediction_p = predicted_stime_to_ptime_map(unique_score_times)
    groundtruth_p = groundtruth_stime_to_ptime_map(unique_score_times)
    error = groundtruth_p - prediction_p
    print("Diff:", 
          " \n mean abs:", np.abs(error).mean(),
          " \n median:", np.median(error),
          " \n max abs:", np.abs(error).max(),
          " \n < 25 ms:", (np.abs(error) < 0.025).sum() / len(error),
          " \n < 50 ms:", (np.abs(error) < 0.05).sum() / len(error),
          " \n < 100 ms:", (np.abs(error) < 0.1).sum() / len(error),
          " \n < 200 ms:", (np.abs(error) < 0.2).sum() / len(error),
          " \n < 500 ms:", (np.abs(error) < 0.5).sum() / len(error),
          )

    return error, unique_score_times, prediction_p, groundtruth_p


def create_map_from_times(predicted_perf_times,
                        predicted_score_times):
    # Use only unique onsets
    predicted_score_unique_onsets = np.unique(predicted_score_times)

    predicted_score_unique_onset_idxs = [
        np.where(predicted_score_times == u)[0] for u in predicted_score_unique_onsets
    ]
    predicted_eq_perf_onsets = np.array(
        [np.mean(predicted_perf_times[u]) for u in predicted_score_unique_onset_idxs]
    )

    predicted_stime_to_ptime_map = interp1d(
        y=predicted_eq_perf_onsets,
        x=predicted_score_unique_onsets,
        bounds_error=False,
        fill_value="extrapolate",
    )
    return predicted_stime_to_ptime_map

def process_files(path_to_midi, path_to_score, 
                  path_to_match, path_to_audio,
                    compare_to_ddtw = False):
    freq_min = 27.5
    freq_max = 4186
    n_bins = 88
    frame_rate = 50

    # SYM PREPROCESSING
    print("SYM PREPROCESSING")
    pna = pt.load_performance_midi(path_to_midi)[0].note_array()
    score = pt.load_musicxml(path_to_score)
    sna = pt.score.unfold_part_maximal(pt.score.merge_parts(score.parts)).note_array(include_grace_notes=True)
    _, alignment = pt.load_match(path_to_match)
    s_feats, s_unique_onsets = prepare_score(sna)
    
    # AUDIO PREPROCESSING
    print("AUDIO PREPROCESSING")
    log_multiplier = 1000
    sr, audio = scipy.io.wavfile.read(path_to_audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)  # convert to mono
    w = audio / 32768
    processor = IIRSpect(sample_rate=sr,
                    n_fft=int(sr/frame_rate),
                    hop_length=int(sr/frame_rate))
    iirspec = processor(w)
    iirlspec = np.log1p(log_multiplier * iirspec)
    iirlspec3 = maximum_filter(iirlspec, size = (3, 1))
    iirlspec4 = np.maximum(0,iirlspec[:,1:] - iirlspec3[:,:-1])
    # l_frequencies = np.logspace(np.log2(freq_min), np.log2(freq_max), n_bins, base = 2)
    # init_template = create_init_template(l_frequencies,
    #                                  l_frequencies, 
    #                                 harmonic_attenuation = lambda x : 1/x)
    # scaled_init_template =  iirspec.max() * init_template
    
    # # pianoroll-like features
    # coeff = nnls_batch_sklearn(iirlspec,T = scaled_init_template)
    # coeff /= coeff.max()

    # onset features
    iirlspec4_row_max = iirlspec4.max(axis=1, keepdims=True)
    iirlspec4 /= iirlspec4_row_max
    iirlspec_row_max = iirlspec.max(axis=1, keepdims=True)
    coeff = iirlspec / iirlspec_row_max

    # ALIGNMENT
    print("ALIGNMENT")
    slice_start = int(np.min(pna["onset_sec"]) * frame_rate)
    pitch_sets =  [np.array(list(b)) - 21 for a,b in s_feats]
    slice_end = int(np.max(pna["onset_sec"]) * frame_rate) + 1
    bp_average = (slice_end - slice_start)/((s_unique_onsets[-1] - s_unique_onsets[0]))

    # D, B, BP, spikes = elastic_forward_and_backward_pitch_onset(
    #                             onsets = 1-iirlspec4[:,slice_start:slice_end],
    #                             spec = 1-coeff[:,slice_start:slice_end],
    #                             spikes = s_unique_onsets,
    #                             pitch_sets = pitch_sets,
    #                             bp_init = bp_average,
    #                             max_stretch_longer = 0.5,
    #                             max_stretch_shorter = 0.3,
    #                             spike_cost = 0.7, 
    #                             stretch_cost = 0.1,
    #                             spec_cost = 0.2,
    #                             max_bp = 8.0 * bp_average,
    #                             min_bp = 0.125 * bp_average,
    #                             alpha = 0.5, # higher = more adaptive
    #                             cost_threshold = 20) 
    D, B, BP, spikes = elastic_forward_and_backward_pitch_onset_limit(
                                onsets = 1-iirlspec4[:,slice_start:slice_end],
                                spec = 1-coeff[:,slice_start:slice_end],
                                spikes = s_unique_onsets,
                                pitch_sets = pitch_sets,
                                bp_init = bp_average,
                                max_stretch_longer = 0.5,
                                max_stretch_shorter = 0.5,
                                spike_cost = 0.7,
                                stretch_cost = 0.1,
                                spec_cost = 0.2,
                                max_bp = 8.0 * bp_average,
                                min_bp = 0.125 * bp_average,
                                alpha = 0.5,
                                spec_slice_len = 20,
                                cost_threshold = 15,
                                candidate_onset_number = 5

                                )



    # EVALUATION
    print("EVALUATION")
    predicted_stime_to_ptime_map = create_map_from_times(np.array(spikes) / frame_rate + slice_start / frame_rate ,  s_unique_onsets)
    error, unique_score_times, prediction_p, groundtruth_p = evaluate(sna,
                pna,
                alignment,
                predicted_stime_to_ptime_map)
    
    if compare_to_ddtw:
        # SYM COMPARISON
        print("SYM COMPARISON")

        
        matcher = pa.DualDTWNoteMatcher()
        pred_alignment = matcher(sna,pna) 
        _, stime_to_ptime_map_symbolic = pt.musicanalysis.performance_codec.get_time_maps_from_alignment(
                    pna,
                    sna,
                    pred_alignment)
        error_sym, _, prediction_p_sym, _ = evaluate(sna,
                    pna,
                    alignment,
                    stime_to_ptime_map_symbolic)
        plt.plot(unique_score_times, prediction_p_sym, label= "pred sym")
    
    else:
        error_sym = None

    plt.plot(unique_score_times, prediction_p, label= "pred")
    plt.plot(unique_score_times, groundtruth_p, label= "gt")
    plt.legend()
    path_to_midi_list = str(path_to_midi).split("/")
    plt.savefig("/home/silvan/TESTER/audio_score_alignment_outputs_5points/"+"_".join(path_to_midi_list[-3:])+"_shorter_window_5alpha_7spike_50framerate.png")
    plt.close()


    return error, error_sym




if __name__ == "__main__":
    import time

    meta = "/home/silvan/TESTER/nasap-dataset/metadata.csv"
    maestro_path = Path("/opt/datasets/fs/maestro-v2.0.0/audio_and_midi")
    asap_path = Path("/home/silvan/TESTER/nasap-dataset")
    data = pd.read_csv(meta)
    output_path = Path("/home/silvan/TESTER/audio_score_alignment_outputs_5points")


    index = 0
    now = time.time()
    errors = dict()
    errors_sym = dict()
    for idx, row in data.iterrows():
        if not row.isna()["maestro_audio_performance"]:

            path_to_audio = maestro_path/ Path(row.maestro_audio_performance.replace("{maestro}/",""))
            path_to_midi = asap_path / Path(row.midi_performance)
            path_to_match = asap_path / Path(row.match_file)
            path_to_score = asap_path / Path(row.xml_score)

            

            if row.isna()["start"] and row.robust_note_alignment:
                print(index, row.midi_performance)
                # if "slamey" not in row.midi_performance:
                #     continue
                try:
                    error, error_sym = process_files(path_to_midi, path_to_score, 
                                            path_to_match, path_to_audio,
                                            compare_to_ddtw = False)
                    errors[row.midi_performance] = error
                    errors_sym[row.midi_performance] = error_sym
                    print(time.time() - now)
                except:
                    print("some processing error")
                print("*"*30)
                now = time.time()
                index += 1
                # if index >= 1:
                #     break

    np.savez("alignment_errors_audio_5points.npz", **errors)  
    # np.savez("alignment_errors_sym.npz", **errors_sym) 


                

