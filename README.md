<!-- ABOUT THE PROJECT -->
## About The Project

This is a repository that contains the implementation of IEEE Access paper ["Projection-based Point Convolution for Efficient Point Cloud Segmentation"](https://ieeexplore.ieee.org/document/9687584). When using this code, please cite the paper as below.

```sh
@article{ahn2022projection,
  title={Projection-based Point Convolution for Efficient Point Cloud Segmentation},
  author={Ahn, Pyunghwan and Yang, Juyoung and Yi, Eojindl and Lee, Chanho and Kim, Junmo},
  journal={IEEE Access},
  year={2022},
  publisher={IEEE}
}
```

<!-- GETTING STARTED -->
## Getting Started

This project is based on [PVCNN](https://github.com/mit-han-lab/pvcnn) and [FPConv](https://github.com/lyqun/FPConv). If you need any more detailed information, please refer to their original repositories.

PPCNN++ trained with each dataloader can be found in [pvcnn_code](https://github.com/pahn04/PPConv/tree/master/pvcnn_code) and [fpconv_code](https://github.com/pahn04/PPConv/tree/master/fpconv_code).


All the codes in this repository are tested under Python 3.7 and PyTorch 1.6. We recommend the users to download the docker image using the following command and then use this repository.

```sh
docker pull pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
```

You need to preprocess the data before running the code. Please follow the procedure of installation and data preparation in each original repository.


<!-- PRETRAINED MODELS -->
## Pretrained models

Pretrained models can be downloaded through google drive links provided below.

* For S3DIS experiment with pvcnn code, download the following pretrained model and place it under ./pvcnn_code/pretrained/.
  ```sh
  https://drive.google.com/file/d/17WgHW_KxAbQ-II5aZZciNEAgT_obR0H1/view?usp=sharing
  ```

* For ShapeNet Part experiment with pvcnn code, download the following pretrained model and place it under ./pvcnn_code/pretrained/.
  ```sh
  https://drive.google.com/file/d/17b9T1EQeJKipHK4JBHrvhBfo5J2eZGqc/view?usp=sharing
  ```

* For S3DIS experiment with fpconv code, download the following model and place it under ./fpconv_code/pretrained/.
  ```sh
  https://drive.google.com/file/d/17bpmoFBHAXYSbnu7PCJ00HdR2m5Mjq4y/view?usp=sharing
  ```

* For ScanNet experiment with fpconv code, download the following model and place it under ./fpconv_code/pretrained/.
  ```sh
  https://drive.google.com/file/d/1SfRYRFd-TUGTt4yiATOlkHGi5Pgd7eS5/view?usp=sharing
  ```


<!-- USAGE -->
## Usage

We provide shell scripts for training and testing in each directory, and the basic usage is the same as the original code of each method.

Please look through the arguments in the shell script and modify them to match your environment.


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

Please visit and show some respect to the providers of the baseline code.

* [PVCNN](https://github.com/mit-han-lab/pvcnn) (MIT License)
* [FPConv](https://github.com/lyqun/FPConv) (MIT License)

