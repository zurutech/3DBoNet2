# Learning Object Bounding Boxes for 3D Instance Segmentation on Point Clouds - TensorFlow 2 version

TensorFlow 2 implementation of [3DBoNet][1] designed for being almost completely in pure TensorFlow (Only one custom op!), extendibile and modular.

---

## Requirements

- Python >= 3.8
- TensorFlow >= 2.9
- CUDA >= 10
- `pip install -r requirements.txt`

## Development

```
pip install -e .
python setup.py build # compiles the custom op
```

## Train and evalution

```
python bin/bonet2-train.py --dataset ScanNet
```

or alternatively, since the bonet2-train.py is installed as a script of this package, you have it in your PATH after the installation:

```
bonet2-train.py --dataset ScanNet
```

The built-in datasets are:

- `ScanNet`
- `S3Dis`

NOTE: The case in the CLI flag (e.g. `--dataset ScanNet`) matters.

---

[1]: https://arxiv.org/abs/1906.01140 "Learning Object Bounding Boxes for 3D Instance Segmentation on Point Clouds - Bo Yang, Jianan Wang, Ronald Clark, Qingyong Hu, Sen Wang, Andrew Markham, Niki Trigoni. 2019."
