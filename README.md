# Open ALBERT Training

本项目提供了一个开源的ALBERT训练代码。

## Create Data
训练数据准备。

```python
!python create_pretraining_data.py \
  --input_file=data/some-text.txt \
  --output_file=train.tfrecord \
  --spm_model_file=$ALBERT_BASE_DIR/30k-clean.model \
  --vocab_file=$ALBERT_BASE_DIR/30k-clean.vocab \
  --do_lower_case=True \
  --max_seq_length=512 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```


## Pre-Training
执行预训练，使用lamb优化器算法。

```shell
!python run_pretraining.py \
    --input_file=train.tfrecord \
    --output_dir=/content/trained_output/ \
    --albert_config_file=$ALBERT_BASE_DIR/albert_config.json \
    --do_train \
    --do_eval \
    --train_batch_size=32 \
    --eval_batch_size=32 \
    --max_seq_length=512 \
    --max_predictions_per_seq=20 \
    --optimizer='lamb' \
    --learning_rate=.00176 \
    --num_train_steps=100000 \
    --num_warmup_steps=5000 \
    --save_checkpoints_steps=5000
```

## 训练日志

完整的训练步骤见[albert_pretraining_on_albert_tiny.ipynb](colab%2Falbert_pretraining_on_albert_tiny.ipynb)

## 致谢
项目作者： Brian Shen. Twitter@dezhou. 欢迎大家关注，谢谢。

## 关注我们
欢迎关注知乎专栏号。

[深度学习兴趣小组](https://www.zhihu.com/column/thuil)

## 问题反馈 & 贡献
如有问题，请在GitHub Issue中提交。
我们没有运营，鼓励网友互相帮助解决问题。
如果发现实现上的问题或愿意共同建设该项目，请提交Pull Request。