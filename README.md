# POST OCR Correction

To train the model:

```
python -m src.main --ctx --data yle-train.txt --model tf_ctx_rand_full --batch 256 --epoch 2 --rand 0.5
```

To resume the training, add `--resume` option
