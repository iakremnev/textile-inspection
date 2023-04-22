# Anomaly detection for textile fabric images

Base library: [openvinotoolkit/anomalib](https://github.com/openvinotoolkit/anomalib)

Anomaly detection is performed as a semantic segmentation task by the [PADIM](https://github.com/openvinotoolkit/anomalib/tree/main/src/anomalib/models/padim) model.
Trained and tested on the [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad) dataset.

## Model specs:


Input shape | 256x256
----|----
patches AUC | 0.970
pathces F1 score | 0.948
pixel AUC | 0.986
pixel F1 score | 0.600


## How to run

```
python infer.py --help
```