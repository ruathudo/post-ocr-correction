# POST OCR Correction

To train the model:

```
python -m src.main --data yle-all.txt --model tf_ctx_trained_full --batch 256 --epoch 3 --rand 0 --window 3
```

To resume the training, add `--resume` option.


The codes are implemented for this paper: https://arxiv.org/abs/2011.03502.
If you are using this repo for your research purposes, please cite this as:

```
@misc{duong2020unsupervised,
      title={An Unsupervised method for OCR Post-Correction and Spelling Normalisation for Finnish}, 
      author={Quan Duong and Mika Hämäläinen and Simon Hengchen},
      year={2020},
      eprint={2011.03502},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

Thanks!