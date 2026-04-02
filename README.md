# <Your Algorithm Name>

Official implementation of **DiffPC: Diffusion-Based Projector Photometric Compensation**

---

## Dataset

We use the following datasets for training:

- CompenNet  
- CompenNeSt++
- Synthetic data used in CompenNeSt++

Please download the [datasets](https://github.com/BingyaoHuang/CompenNeSt-plusplus) and organize them in your local directory before training.

---

## Usage

### Training

To train the model, run:

```bash
cd cmp_sde/code/config/cmp
python train.py -opt=options/train/diffpc_train.yaml
```


Before training, you need to modify the dataset path in the config file:
```yaml
# options/train/diffpc_train.yaml
```
Replace YOUR DATA PATH with your own dataset directory.

---

### Testing

To evaluate the model, run:

```bash
cd cmp_sde/code/config/cmp
python test.py -opt=options/test/diffpc_test.yaml
```



Before testing, you need to modify the dataset path and pre-trained model in the config file:
```yaml
# options/test/diffpc_test.yaml
```
Replace YOUR DATA PATH with your own data directory, YOUR PRE-TRAINED MODEL with your pre-trained model.
---

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{your2025algorithm,
  title={DiffPC: Diffusion-Based Projector Photometric Compensation},
  author={Yuxi Wang, Haibin Ling, Bingyao Huang},
  journal={},
  year={2026}
}
```

---

## Acknowledgements

Our implementation is built upon the following project:

[Image Restoration SDE](https://github.com/LVCHENYONG/refusion)

We sincerely thank the authors for their open-source contribution.

---

