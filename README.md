# Semantic-Segmentation-with-Sparse-Labels
The labels and codes for [Semantic Segmentation of Remote Sensing Images with Sparse Annotations](https://arxiv.org/pdf/2101.03492.pdf).

## Data
We provided three types of sparse annotations: point, scribble, and polygon.

## Usage
1) install dependencies in ```requirements.txt```
2) download [data](https://drive.google.com/file/d/1E4bhx3H6P8jTdOQG6hS14G_gBBhvwzWU/view?usp=sharing) and unzip in the folder ```data```. The directory structure should be as follows:
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images

3) download [weights](https://drive.google.com/file/d/10BYt1lvRNBtgx76lMiuWj7J2kF-tSBV1/view?usp=sharing) and unzip it in the folder ```weights```.
4) run ```python train.py``` and ```python test.py``` for testing and training

## Citation
If you find they are useful, please kindly cite the following:
```
@article{hua2021sparse,
  title={Semantic Segmentation of Remote Sensing Images with Sparse Annotations},
  author={Hua, Yuansheng and Marcos, Diego and Mou, Lichao and Zhu, Xiao Xiang and Tuia, Devis},
  journal={IEEE Geoscience and Remote Sensing Letters},
  year={in press}
}
```
