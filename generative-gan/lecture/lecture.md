# Generative Adversarial Networks

~~~
\subtitle{Machine Learning and Data Mining}
\author{Maxim Borisyak}

\institute{National Research University Higher School of Economics (HSE)}
\usepackage{amsmath}

\DeclareMathOperator*{\E}{\mathbb{E}}

\DeclareMathOperator*{\var}{\mathbb{D}}
\newcommand\D[1]{\var\left[ #1 \right]}

\newcommand\dmid{\,\|\,}

\DeclareMathOperator*{\argmin}{\mathrm{arg\,min}}
\DeclareMathOperator*{\argmax}{\mathrm{arg\,max}}
~~~

## Generative models

### Generative models

- Informally, given samples we wish to learn underlying distribution in form of sampling procedure.
- Formally, given samples of a random variable $X$, we wish to find $X'$, so that:
$$P(X) \approx P(X')$$

### Types of generative models

Fitting density function $P(X)$ (or a function $f(x)$ proportional to the density):
- partition function $Z$ for $f(x)$ might be an issue:
  - for energy models Contrastive Devergence keeps $Z$ finite (ideally constant);
- sampling might be an issue:
  - Gibbs sampling works ok if $P(x^i \mid x^{-i})$ is analytically known and simple enough.

Going deep:
- RBM is intrinsically one-layer model;
- Deep Boltzmann machines:
  - Gibbs sampling becomes less efficient than for RBM.

### Types of generative procedures

Options for defining a random variables:
- specify $P(X)$ and use general sampling algorithm (e.g. Gibbs sampling);
- **learn sampling procedure directly**, e.g.:

~~~eqnarray*
  X &=& f(Z);\\
  Z &\sim& \mathrm{SimpleDistribution};
~~~

### Fitting Distributions

> Notation: $Q$ - ground truth distribution, $P$ - model.

Maximum Likelihood:

~~~eqnarray*
  \mathcal{L} &=& \sum_i \log P(x_i) \approx \E_{x \sim Q} \log P(x) \to_P \min;\\
  \mathrm{KL}(Q \dmid P) &=& \E_{x \sim Q} \log Q(x) - \E_{x \sim Q} \log P(x) \to_P \min.
~~~

Jenson-Shenon:

~~~eqnarray*
  \mathrm{JS}(P, Q) &=& \frac{1}{2} \left[ \mathrm{KL}(P \dmid M) + \mathrm{KL}(Q \dmid M) \right] \to_P \min;\\
  M &=& \frac{1}{2}(P + Q).
~~~

### Approximating JS distance

~~~multline*
  \mathrm{JS}(P, Q) = \frac{1}{2}\left[ \E_{x \sim P} \log \frac{P(x)}{M(x)} + \E_{x \sim Q} \log \frac{Q(x)}{M(x)} \right] =\\[5mm]
    \frac{1}{2}\left[ \E_{x \sim P} \log \frac{P(x)}{P(x) + Q(x)} + \E_{x \sim Q} \log \frac{Q(x)}{P(x) + Q(x)} \right] + \log 2 =\\[5mm]
    \E_{x \sim M} \frac{P(x)}{P(x) + Q(x)} \log \frac{P(x)}{P(x) + Q(x)} + \\[3mm] \E_{x \sim M} \frac{Q(x)}{P(x) + Q(x)} \log \frac{Q(x)}{P(x) + Q(x)} + \log 2
~~~

### Approximating JS distance

~~~multline*
\mathrm{JS}(P, Q) = \E_{x \sim M} \frac{P(x)}{P(x) + Q(x)} \log \frac{P(x)}{P(x) + Q(x)} +\\[3mm] \E_{x \sim M} \frac{Q(x)}{P(x) + Q(x)} \log \frac{Q(x)}{P(x) + Q(x)} + \log 2
~~~

Let's introduce $y$: $y = 1$ if $x$ is sampled from $P$ and $y = 0$ for $Q$:

~~~multline*
\mathrm{JS}(P, Q) - \log 2 =\\[5mm]
  \E_{x \sim M} \left[ P(y = 1 \mid x) \log P(y = 1 \mid x) + P(y = 0 \mid x) \log P(y = 0 \mid x) \right] =\\[5mm]
  \E_{x \sim M} \left[ \E_y \left[y \mid x\right] \log P(y = 1 \mid x) + ( 1 - \E_y \left[y \mid x\right]) \log P(y = 0 \mid x) \right]
~~~

### Approximating JS distance

~~~multline*
\mathrm{JS}(P, Q) - \log 2 =\\[5mm]
  \E_{x \sim M} \left[ \E_y \left[y \mid x\right] \log P(y = 1 \mid x) + ( 1 - \E_y \left[y \mid x\right]) \log P(y = 0 \mid x) \right]=\\[5mm]
  \E_{x \sim M, y} y\, \log P(y = 1 \mid x) + (1 - y) \log P(y = 0 \mid x)
~~~

~~~equation*
\boxed{ \mathrm{JS}(P, Q) = \log 2 - \min_f \left[ \mathrm{cross\textendash entropy}(f \dmid P, Q)  \right] }
~~~
