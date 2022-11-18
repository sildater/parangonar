**Parangonar** is a Python package for note alignment of symbolic music. 
**Parangonar** uses [Partitura](https://github.com/CPJKU/partitura) as file I/O utility.
Note alignments produced py **Parangonar** can be visualized using the 
web tool [Parangonda](https://sildater.github.io/parangonada/)


Installation
==========

The easiest way to install the package is via `pip` from the [PyPI (Python
Package Index)](https://pypi.python.org/pypi>):
```shell
pip install parangonar
```
This will install the latest release of the package and will install all dependencies automatically.


Quickstart
==========

The following code loads the contents of a a previously aligned performance
and score alignment file (encoded in the [match file format](https://arxiv.org/abs/2206.01104)). 

A new alignment is computed using a hierarchical DTW-based note matcher and the resulting
alignment are compared to the ground truth:

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


License
=======

The code in this package is licensed under the Apache 2.0 License. For details,
please see the [LICENSE](LICENSE) file.
