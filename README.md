# PyTorch implementation of MobileNet V4

Reproduction of MobileNet V4 architecture as described in [MobileNetV4 - Universal Models for the Mobile Ecosystem](https://arxiv.org/abs/2404.10518) by Danfeng Qin, Chas Leichner, Manolis Delakis, Marco Fornoni, Shixin Luo, Fan Yang, Weijun Wang, Colby Banbury, Chengxi Ye, Berkin Akin, Vaibhav Aggarwal, Tenghui Zhu, Daniele Moro, Andrew Howard with the [PyTorch](pytorch.org) framework.

## Models

| Architecture      | # Parameters | FLOPs @ pix | Top-1 Acc. (%) |
| ----------------- | ------------ | ------ | -------------------------- |
| MobileNetV4-S    | 4.30M | 0.306G @ 224 |  |
| MobileNetV4-M    | 9.72M | 1.080G @ 256 |  |
| MobileNetV4-L    | 32.59M | 6.376G @ 384 |  |

## Acknowledgement

```
@misc{qin2024mobilenetv4,
      title={MobileNetV4 -- Universal Models for the Mobile Ecosystem}, 
      author={Danfeng Qin and Chas Leichner and Manolis Delakis and Marco Fornoni and Shixin Luo and Fan Yang and Weijun Wang and Colby Banbury and Chengxi Ye and Berkin Akin and Vaibhav Aggarwal and Tenghui Zhu and Daniele Moro and Andrew Howard},
      year={2024},
      eprint={2404.10518},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

The official [TensorFlow implementation](https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py).
