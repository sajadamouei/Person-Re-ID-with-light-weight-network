# Person-Re-ID-with-light-weight-network

### An Improved Person Re-identification Method by light-weight convolutional neural network
---
![image123](https://user-images.githubusercontent.com/27925997/90361713-27654d80-e074-11ea-85de-f267cb42fcf3.png)

we provide our training and testing code written in Keras for the paper "An Improved Person Re-identification Method by light-weight convolutional neural network".

1. This network is composed of Siamese architecture.
2. EfficientNet was employed to obtain discriminative features and reduce the demands for data.
3. we use x*y instead of (x-y)^2 as Square Layer.
4. We ran our tests on CUHK01 dataset.

### Prerequisites
---
We used google colab to train and test the network:

- python version=  3.6.9

- keras version=  2.2.5

- GPU= Tesl P100

- GPU Memory= 16G

### Evaluation
---
We arrived Rank@1= 70.1%, Rank@5= 95.2%, Rank@10= 99.1%, Rank@15= 99.1% and Rank@20= 99.2% with EfficientNetB0.

![evalu](https://user-images.githubusercontent.com/27925997/90381698-f8aa9f80-e092-11ea-9208-fbb3a56aa525.png)

### Citation
---
Please cite this paper in your publications if it helps your research:
```
@article{amouei2020transferlearning,
  title={An Improved Person Re-identification Method by light-weight convolutional neural network},
  author={Sajad Amouei Sheshkal and  Kazim Fouladi-Ghaleh and Hossein Aghababa},
  year={2020}
}
```
