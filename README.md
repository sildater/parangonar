Parangonar
==========

**Parangonar** is a Python package for note alignment of symbolic music. 
**Parangonar** uses [Partitura](https://github.com/CPJKU/partitura) as file I/O utility.
Note alignments produced py **Parangonar** can be visualized using the 
web tool [Parangonda](https://sildater.github.io/parangonada/)


Installation
-------

The easiest way to install the package is via `pip` from the [PyPI (Python
Package Index)](https://pypi.python.org/pypi>):
```shell
pip install parangonar
```
This will install the latest release of the package and will install all dependencies automatically.


Quickstart Examples
==========

The following code loads the contents of a a previously aligned performance
and score alignment file (encoded in the [match file format](https://arxiv.org/abs/2206.01104)). 

A new alignment is computed using a hierarchical DTW-based note matcher and the resulting
alignment are compared to the ground truth:

1 - Automatic Note Matching: `AutomaticNoteMatcher` and `DualDTWNoteMatcher`
-----

```python
import parangonar as pa
import partitura as pt

perf_match, groundtruth_alignment, score_match = pt.load_match(
    filename= pa.EXAMPLE,
    create_score=True
)

# compute note arrays from the loaded score and performance
pna_match = perf_match.note_array()
sna_match = score_match.note_array()

# match the notes in the note arrays
sdm = pa.AutomaticNoteMatcher()
pred_alignment = sdm(sna_match, 
                     pna_match,
                     verbose_time=True)

# compute f-score and print the results
print('------------------')
types = ['match','insertion', 'deletion']
for alignment_type in types:
    precision, recall, f_score = pa.fscore_alignments(pred_alignment, 
                                                      groundtruth_alignment, 
                                                      alignment_type)
    print('Evaluate ',alignment_type)
    print('Precision: ',format(precision, '.3f'),
          'Recall ',format(recall, '.3f'),
          'F-Score ',format(f_score, '.3f'))
    print('------------------')
```

Aligning MusicXML Scores and MIDI Performances
-----

```python
import parangonar as pa
import partitura as pt

score = pt.load_score(filename= 'path/to/score_file')
performance = pt.load_performance_midi(filename= 'path/to/midi_file')

# compute note arrays from the loaded score and performance
pna = performance.note_array()
sna = score.note_array()

# match the notes in the note arrays
sdm = pa.AutomaticNoteMatcher()
pred_alignment = sdm(sna_match, pna_match)
```

2 - Anchor Point Alignment: `AnchorPointNoteMatcher` 
----

```python
import parangonar as pa
import partitura as pt

perf_match, groundtruth_alignment, score_match = pt.load_match(
    filename= pa.EXAMPLE,
    create_score=True
)

# compute note arrays from the loaded score and performance
pna_match = perf_match.note_array()
sna_match = score_match.note_array()

# compute synthetic anchor points every 4 beats
nodes = pa.match.node_array(score_match[0], 
                   perf_match[0], 
                   groundtruth_alignment,
                   node_interval=4)

# match the notes in the note arrays
apdm = pa.AnchorPointNoteMatcher()
pred_alignment = apdm(sna_match, 
                     pna_match,
                     nodes)

# compute f-score and print the results
print('------------------')
types = ['match','insertion', 'deletion']
for alignment_type in types:
    precision, recall, f_score = pa.fscore_alignments(pred_alignment, 
                                                      groundtruth_alignment, 
                                                      alignment_type)
    print('Evaluate ',alignment_type)
    print('Precision: ',format(precision, '.3f'),
          'Recall ',format(recall, '.3f'),
          'F-Score ',format(f_score, '.3f'))
    print('------------------')
```


3 - Online / Realtime Alignment: `OnlineTransformerMatcher` 
----

```python
import partitura as pt
import parangonar as pa

### TODO
```





4 - File I/O for note alignments
----

```python
import partitura as pt
import parangonar as pa

# load note alignments of the asap dataset: 
# https://github.com/CPJKU/asap-dataset/tree/note_alignments
alignment = pt.io.importparangonada.load_alignment_from_ASAP(filename= 'path/to/note_alignment.tsv')

# export a note alignment for visualization with parangonada:
# https://sildater.github.io/parangonada/
pa.match.save_parangonada_csv(alignment, 
                            performance_data,
                            score_data,
                            outdir="path/to/dir")

# import a corrected note alignment from parangonada:
# https://sildater.github.io/parangonada/
alignment = pt.io.importparangonada.load_parangonada_alignment(filename= 'path/to/note_alignment.csv')
```

5 - Visualize Alignment
----


```python
import parangonar as pa
import partitura as pt

perf_match, alignment, score_match = pt.load_match(
    filename= pa.EXAMPLE,
    create_score=True
)
pna_match = perf_match.note_array()
sna_match = score_match.note_array()

# show or save plot of note alignment
pa.plot_alignment(pna_match,
                sna_match,
                alignment,
                save_file = False)
```


Publications
=====

Two publications are associated with models available in **Parangonar**.
The anchor point-enhanced `AnchorPointNoteMatcher` and the automatic `AutomaticNoteMatcher` are this described in:

```
@article{nasap-dataset,
 title = {Automatic Note-Level Score-to-Performance Alignments in the ASAP Dataset},
 author = {Peter, Silvan David and Cancino-Chacón, Carlos Eduardo and Foscarin, Francesco and McLeod, Andrew Philip and Henkel, Florian and Karystinaios, Emmanouil and Widmer, Gerhard},
 doi = {10.5334/tismir.149},
 journal = {Transactions of the International Society for Music Information Retrieval {(TISMIR)}},
 year = {2023}
}
```

and the former is used in the creation of the [note-aligned (n)ASAP Dataset](https://github.com/CPJKU/asap-dataset).


The improved automatic `DualDTWNoteMatcher` and the online / realtime `OnlineTransformerMatcher` are described in:


```
@inproceedings{peter-2023,
  title={Online Symbolic Music Alignment with Offline
Reinforcement Learning},
  author={Peter, Silvan David},
  booktitle={International Society for Music Information Retrieval Conference {(ISMIR)}},
  year={2023}
}
```

## Acknowledgments

This work is supported by the European Research Council (ERC) under the EU’s Horizon 2020 research & innovation programme, grant agreement No. 10101937 (”Wither Music?”).

License
=======

The code in this package is licensed under the Apache 2.0 License. For details,
please see the [LICENSE](LICENSE) file.
