Parangonar
==========

**Parangonar** is a Python package for note alignment and of symbolic music. 
**Parangonar** contains offline and online note alignment algorithms as well as task-agnostic dynamic programming sequence alignment algorithms.
Note alignments produced by **Parangonar** can be visualized using the web tool [Parangonda](https://sildater.github.io/parangonada/).
**Parangonar** uses [Partitura](https://github.com/CPJKU/partitura) as file I/O utility.


Installation
-------

The easiest way to install the package is via `pip` from the [PyPI (Python
Package Index)](https://pypi.org/project/parangonar/):
```shell
pip install parangonar
```
This will install the latest release of the package and will install all dependencies automatically.


Getting Started
-------

There is a `getting_started.ipynb` notebook which covers the basic note alignment functions.

To demonstrate **Parangonar** the contents of performance and score alignment file (encoded in the [match file format](https://cpjku.github.io/matchfile/)) are loaded, which returns a score object, a performance objects, and an alignment list. A new alignment is computed using different note matchers and the predicted alignment are compared to the ground truth.


Documentation: Creation of Note Alignments
-------

**Parangonar** contains implementations of note alignments algorithms:

1. Offline Note Matching: 
    - `AutomaticNoteMatcher`: 
        piano roll-based, hierarchical DTW and combinatorial optimization for pitch-wise note distribution.
        requires scores and performances in the current implementation, but not necessarily.
    - `DualDTWNoteMatcher`: 
        symbolic note set-based DTW, pitch-wise onsetDTW, separate handling of ornamentations possible.
        requires scores and performances for sequence representation.
        **Default and SOTA** for standard score to performance matching.
    - `TheGlueNoteMatcher`:
        pre-trained neural network for note similarity, useful for large mismatches between versions.
        works on any two MIDI files.
    - `AnchorPointNoteMatcher`: 
        semi-automatic version of the `AutomaticNoteMatcher`, useful if annotations can be leveraged as anchor points. 


2. Online / Real-time Note Matching: 
    - `OnlineTransformerMatcher`::
        pre-trained neural network for local alignment decisions.
        post-processing by a tempo model.
    - `OnlinePureTransformerMatcher` 
        pre-trained neural network for local alignment decisions.
        no post-processing.
    - `TempoOLTWMatcher`: 
        symbolic dynamic programming akin to Online Time Warping
        based on a tempo and pitch-based metric
    - `OLTWMatcher`:
        symbolic dynamic programming akin to Online Time Warping
        based on a pitch-based metric 

3. Mismatching (cases other than one-to-one matching): 
    - `RepeatIdentifier`::
        automatically infer the repeat structure of a MIDI performance
    - `SubPartMatcher` 
        note matcher which matches a monophonic voice from the score
        to a performance.

Documentation: Dynamic Programming Algorithms
-------

**Parangonar** contains implementations of (non-)standard dynamic programming sequence alignment algorithms:

1. DTW (multiple versions, using numpy/numba/jit)
    - vanilla DTW
    - weightedDTW: generalized directions, weights, and penalites
    - FlexDTW: flexible start and end points, Bükey at al.

2. NWTW (multiple versions, using numpy/numba/jit)
    - Needleman-Wunsch: using distances on scalars, minimizing version
    - NWDTW: Needleman-Wunsch Time Warping, Grachten et al.
    - weightedNWDTW: generalized directions, weights, and penalites
    - original Needleman-Wunsch: using binary gamma on scalars, maximizing version
    - original Smith-Waterman: using binary gamma on scalars, maximizing version
    - BoundedSmithWaterman: local sequence alignment with bounded gain

3. OLTW:
    - On-Line Time Warping: standard OLTW, Dixon et al.
    - Tempo OLTW: path-wise tempo models



Documentation: Note Alignment Utilities
-------

**Parangonar** contains several utilities around note matching:

1. Alignment Visualization:
    - parangonar.evaluate.plot_alignment 
    - parangonar.evaluate.plot_alignment_comparison
    - parangonar.evaluate.plot_alignment_mappings  

2. Alignment Evaluation
    - parangonar.evaluate.fscore_alignments
    - parangonar.evaluate.fscore_alignments
    - parangonar.evaluate.fscore_alignments

3. File I/O for note alignments

    Most I/O functions are handled by [Partitura](https://github.com/CPJKU/partitura). 

    For [Parangonada](https://sildater.github.io/parangonada/):
    - partitura.io.importparangonada.load_parangonada_alignment
    - partitura.io.importparangonada.load_parangonada_csv
    - partitura.io.exportparangonada.save_parangonada_alignment
    - partitura.io.exportparangonada.save_parangonada_csv

    For [(n)ASAP alignments](https://github.com/CPJKU/asap-dataset)
    - partitura.io.importparangonada.load_alignment_from_ASAP
    - partitura.io.exportparangonada.save_alignment_for_ASAP

    For [match files](https://cpjku.github.io/matchfile/)
    - partitura.io.importmatch.load_match
    - partitura.io.exportmatch.save_match

    and a basic interface for saving parangonada-ready csv files is also available in parangonagar:
    - parangonar.match.save_parangonada_csv

4. Aligned Data

    These note-aligned datasets are publically available:
    - [Vienna 4x22](https://github.com/CPJKU/vienna4x22)
    - [(n)ASAP note alignments](https://github.com/CPJKU/asap-dataset)
    - [Batik Dataset](https://github.com/huispaty/batik_plays_mozart)


Publications
-------

Several publications are associated with models available in **Parangonar**, you find a list in the `refs.md` file. Please to them for detailed algorithm descritions and evaluation and cite the relevant ones if you use parangonar for a publication.

Acknowledgments
-------

This work is supported by the European Research Council (ERC) under the EU’s Horizon 2020 research & innovation programme, grant agreement No. 10101937 (”Wither Music?”).

License
-------

The code in this package is licensed under the Apache 2.0 License. For details,
please see the [LICENSE](LICENSE) file.
