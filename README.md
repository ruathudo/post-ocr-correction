# POST OCR Correction

To train the model:

```
python -m src.main --data yle-all.txt --model tf_ctx_trained_full --batch 256 --epoch 3 --rand 0 --window 3
```

To resume the training, add `--resume` option.


The codes are implemented for this paper: https://aclanthology.org/2021.nodalida-main.24/.
If you are using this repo for your research purposes, please cite this as:

```
@inproceedings{duong-etal-2021-unsupervised,
    title = "An Unsupervised method for {OCR} Post-Correction and Spelling Normalisation for {F}innish",
    author = {Duong, Quan  and
      H{\"a}m{\"a}l{\"a}inen, Mika  and
      Hengchen, Simon},
    booktitle = "Proceedings of the 23rd Nordic Conference on Computational Linguistics (NoDaLiDa)",
    month = may # " 31--2 " # jun,
    year = "2021",
    address = "Reykjavik, Iceland (Online)",
    publisher = {Link{\"o}ping University Electronic Press, Sweden},
    url = "https://aclanthology.org/2021.nodalida-main.24",
    pages = "240--248",
    abstract = "Historical corpora are known to contain errors introduced by OCR (optical character recognition) methods used in the digitization process, often said to be degrading the performance of NLP systems. Correcting these errors manually is a time-consuming process and a great part of the automatic approaches have been relying on rules or supervised machine learning. We build on previous work on fully automatic unsupervised extraction of parallel data to train a character-based sequence-to-sequence NMT (neural machine translation) model to conduct OCR error correction designed for English, and adapt it to Finnish by proposing solutions that take the rich morphology of the language into account. Our new method shows increased performance while remaining fully unsupervised, with the added benefit of spelling normalisation. The source code and models are available on GitHub and Zenodo.",
}
```

Thanks!
