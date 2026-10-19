# DRA-Net: Densely Residual Attention based Low-light Image Enhancement

Official code for the paper "DRA-Net: densely residual attention based
low-light image enhancement" (ICGIP 2023, SPIE).

## Architecture

![architecture](assets/Architecture.png)

## Requirements

- python 3
- Tensorflow 1.6.0
- Keras 2.2.0
- Opencv-python 3.4.2

## Usage

#### Testing

To test your own low-light images with our model, run

```
python predict.py -m <model folder> -i <input folder> -r <output folder>
```

By default, the code loads the model from "./saved_model" and saves results in
the "./results/" folder. Please read the code to see other parameter settings.

To evaluate on a paired test set (e.g. LOL eval15):

```
python evaluate.py -m <model folder> -d <test set root>
```

#### Training:

First, prepare your dataset with the following structure (low/high paired PNG
images of the same names):

```
data_root/
  low/
  high/
```

Then, you can just run through

```
python train.py -d <data_root> -e 250 -b 16
```

By default, the code saves the trained model in the "./saved_model/" folder.
Please read the code to see other parameter settings.


## Results

Comparison with other methods on the LOL dataset:

![comparison](assets/comparison_grid.png)

Quantitative results:

![results](assets/results_table.png)

## Bibtex

If you use this code for your research, please cite our paper.


 ```

@inproceedings{DRANet2023,
  title = {{DRA-Net: densely residual attention based low-light image enhancement}},
  author = {Sami Ul Rehman and Ammar Hawbani and Xingfu Wang and Muhammad Hamza and Liang Zhao and Saeed H. Alsamhi and Majjed Al-Qatf},
  booktitle = {Fourteenth International Conference on Graphics and Image Processing (ICGIP 2022)},
  organization = {International Society for Optics and Photonics},
  volume = {12705},
  publisher = {SPIE},
  pages = {1270522},
  year = {2023},
  doi = {10.1117/12.2680479},
}
 ```


