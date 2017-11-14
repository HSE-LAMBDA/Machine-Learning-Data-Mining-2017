# Macro NN architecture

~~~
\subtitle{Machine Learning and Data Mining}
\author{Maxim Borisyak}

\institute{National Research University Higher School of Economics (HSE)}
\usepackage{amsmath}

\DeclareMathOperator*{\E}{\mathbb{E}}

\DeclareMathOperator*{\var}{\mathbb{D}}
\newcommand\D[1]{\var\left[ #1 \right]}

\DeclareMathOperator*{\argmin}{\mathrm{arg\,min}}
\DeclareMathOperator*{\argmax}{\mathrm{arg\,max}}
~~~

## Outline

### Super inspirational quotes

~~~quote
\Large Network architecture is more like an art.
~~~

`\vfill`

~~~quote
Behind every is a poorly formulated science.
~~~

### Network architecture

> Neural Network Architecture plays crucial role in Deep Learning.

`\vspace{5mm}`

Most of the non-trivial architectures:
- derived from common sense;
- explained by math;
- demonstrated on some real problems.

### Usual disclaimer

> The following examples are not aimed to be cover major architecture tricks.
> Just some examples happened to be known by the author.

## Pretraining

### Layerwise pretraining

![width=1](imgs/pretraining.jpg)

### Pretraining

- layer-wise pretraining:
  - RBM;
  - AE;

- pretraining on simpler but related task.

## Auxilary losses

### Auxilary problems

$$\mathcal{L} = \mathcal{L}_{\mathrm{main}} + \lambda_1 \mathcal{L}_1 + \lambda_2 \mathcal{L}_2 + \dots$$

- solving several objectives with one network:
  - brining more information about the solution;
- auxilary losses should share the same solution;

***

![width=1.1](imgs/Gorinich.jpg)

### Auxilary problems: examples

Are the following auxilary problems reasonable:
- even vs. odd digit for MNIST;
- reconstructing initial image for MNIST;
- producing countour of target objects for detection problems;
- predicting type of a street-sign for detection problem;
- predicting super-class for CIFAR-100;
- predicting faces properies (e.g. smile/anger/neutral, female/male etc) for dimensionality reduction?

### Reconstruction regularisation

```dot [width=1, height=0.7]
digraph G {
  ratio=1.5;
  graph[dpi=420]
  rankdir=BT;
  node[shape=rect, style="filled", fillcolor=green];

  subgraph cluster_0 {
    shape=rect;
    l1[label="layer 1"];
    l2[label="layer 2"];
    l3[label="layer 3"];
    l4[label="layer 4"];
    l5[label="layer 5"];

    l1 -> l2 -> l3 -> l4 -> l5;
  }

  node[shape=rect, style="filled", fillcolor=blue];

  r1[label="reco 1"];
  r2[label="reco 2"];
  r21[label="reco 1"];

  r3 [label="reco 3"];
  r32 [label="reco 2"];
  r321 [label="reco 1"];

  l2 -> r1;

  l3 -> r2 -> r21;

  l4 -> r3 -> r32 -> r321;

  r1 -> l1 [style=dashed,  label=MSE]

  r21 -> l1 [style=dashed,  label=MSE];

  r321 -> l1 [style=dashed,  label=MSE];
}
```

***

```dot [width=1]
digraph G {
  ratio=1.5;
  graph[dpi=420]
  rankdir=BT;
  node[shape=rect, style="filled", fillcolor=green];

  subgraph cluster_0 {
    shape=rect;
    l1[label="layer 1"];
    l2[label="layer 2"];
    l3[label="layer 3"];
    l4[label="layer 4"];
    l5[label="layer 5"];

    l1 -> l2 -> l3 -> l4 -> l5;
  }

  node[shape=rect, style="filled", fillcolor=blue];

  r1[label="reco 1"];
  r2[label="reco 2"];
  r3[label="reco 3"];

  l2 -> r1;
  l3 -> r2;
  l4 -> r3;

  edge [style=dashed]

  r1 -> l1 [label=MSE]
  r2 -> l2 [label=MSE]
  r3 -> l3 [label=MSE]
}
```

### Reconstruction regularization

```dot [width=0.8]
digraph G {
  ratio=0.75;
  graph[dpi=420]
  graph[splines=ortho]
  rankdir=BT;

  node[shape=rect, style="filled", fillcolor=green];

    l1[label="layer 1"];
    l2[label="layer 2"];
    l3[label="layer 3"];
    l4[label="layer 4"];
    l5[label="layer 5"];

    l1 -> l2 -> l3 -> l4 -> l5;


  node[shape=rect, style="filled", fillcolor=blue];
    r1[label="reco 1"];
    r2[label="reco 2"];
    r3[label="reco 3"];

  l4 ->  r3 -> r2 -> r1;

  edge [style=dashed]

  r1 -> l1 [label=MSE]
  r2 -> l2 [label=MSE]
  r3 -> l3 [label=MSE]
}
```

### Reconstruction regularization

- unsupervised loss may be in conflict with classification loss;
  - reconstruction generally require higher network capacities;
  - discriminative features might be lost as unimportant for reconstruction;
- rarely used in practice.

### Deeply supervised networks

- try to solve original problem early;
- improved gradient flow (almost impossible to make it vanish);
- quite strong regularization effect;
- no unsupervised vs. supervised conflict.

***

![width=1](imgs/dsn.png)

### Ladder Networks

- replaces reconstruction with denoising:
  $$\mathcal{L} = \| f(x + \varepsilon) - x\|^2 \to \min, \, \varepsilon \sim \mathcal{N}(0, \sigma^2)$$

---

![width=0.65](imgs/ladder.png)

## Network structure

### Tree-like networks

![width=0.65](imgs/hnet.pdf)

### VGG

![width=1](imgs/vgg16.png)

### Inception

![width=1](imgs/inception.png)

---

- blue blocks: conv;
- red blocks: pool;
- green blocks: concat;
- yellow blocks: softmax.

### Inception block

![width=1](imgs/inception_block.png)

### NIN: conv on steriods

![width=1](imgs/NIN.png)

### ResNet

![width=1](imgs/resblock.png)

### ResNet

`\vspace{3mm}`

![height=0.8](imgs/resnet-34.png)

### ResNet

![width=1](imgs/bresnet_block.png)

### Highway networks

`\vspace{3mm}`

Feed-forward networks:

~~~equation*
  y = H(x, W_H)
~~~

Residual connection:

~~~equation*
  y = H(x, W_H) + x
~~~

Highway connection:

~~~equation*
  y = T(x, W_T) H(x, W_H) + C(x, W_C) x;
~~~

- $x, y$ - input, output;
- $H(x, W_H)$ - some transformation, e.g. convolution;
- $T(x, W_T), C(x, W_C) \in [0, 1]$ - gates (*transform* and *carry*).

### Highway networks

`\vspace{5mm}`

![height=0.8](imgs/highway.png)

### Squeeze net

![width=1](imgs/fire_module.png)

### Squeeze net

`\vspace{3mm}`

![width=0.95](imgs/squeeze-net.png)

### U-net

![width=1](imgs/u-net.png)

### Exercise

Suggest an architecture for a face recognition security system:
- system should be able to grant access to any person with sufficient rights.

Describe:
- data required;
- function of the neural network (classification, regression, clusterisation);
- architecture of the network;
- training procedure.

## Summary

### Summary

- network architecture plays crucial role in Deep Learning;
- additional problems may provide additional information about solution;
- there are tons of various network architectures.

### References I

- Ronneberger O, Fischer P, Brox T. U-net: Convolutional networks for biomedical image segmentation. InInternational Conference on Medical Image Computing and Computer-Assisted Intervention 2015 Oct 5 (pp. 234-241). Springer, Cham.
- Szegedy C, Ioffe S, Vanhoucke V, Alemi AA. Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning. InAAAI 2017 (pp. 4278-4284).
- Srivastava RK, Greff K, Schmidhuber J. Highway networks. arXiv preprint arXiv:1505.00387. 2015 May 3.

### References II

- Rasmus A, Berglund M, Honkala M, Valpola H, Raiko T. Semi-supervised learning with ladder networks. InAdvances in Neural Information Processing Systems 2015 (pp. 3546-3554).
- Lee CY, Xie S, Gallagher P, Zhang Z, Tu Z. Deeply-supervised nets. InArtificial Intelligence and Statistics 2015 Feb 21 (pp. 562-570).
- Goodfellow IJ, Warde-Farley D, Mirza M, Courville A, Bengio Y. Maxout networks. arXiv preprint arXiv:1302.4389. 2013 Feb 18.
- Iandola FN, Han S, Moskewicz MW, Ashraf K, Dally WJ, Keutzer K. SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size. arXiv preprint arXiv:1602.07360. 2016 Feb 24.

### References III

- Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition. arXiv preprint arXiv:1409.1556. 2014 Sep 4.
- He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. InProceedings of the IEEE conference on computer vision and pattern recognition 2016 (pp. 770-778).
- Lin M, Chen Q, Yan S. Network in network. arXiv preprint arXiv:1312.4400. 2013 Dec 16.
