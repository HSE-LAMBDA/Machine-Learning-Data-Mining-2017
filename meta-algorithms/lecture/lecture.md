#  Machine Learning and Data Mining

~~~
\subtitle{Meta-algorithms}
\author{Maxim Borisyak}

\institute{National Research University Higher School of Economics (HSE)}
\usepackage{amsmath}

\DeclareMathOperator*{\E}{\mathbb{E}}

\DeclareMathOperator*{\var}{\mathbb{D}}
\newcommand\D[1]{\var\left[ #1 \right]}
~~~

### In the last episode

`\vspace{5mm}`

```python
def data_science(problem_description,
                 domain_expertise=None,
                 *args, **kwargs):
  if problem_description is None:
    raise Exception('Learning is impossible!')

  prior_on_algorithms = \
    data_scientist.world_knowledge.infer(
      problem_description,
      domain_expertise,
      *args, **kwargs
    )

  return prior_on_algorithms
```

### Making algorithms

Constructing learning algorithms from scratch is hard:
- it is the reason people use machine learning instead of classical statistical approach.

`\vspace{5mm}`

- producing tons of simple, rude algorithms is quite easy;
- fitting all-powerfull zero-bias classifier is easy.

`\vspace{5mm}`

~~~center
\large
Can an good algorithm be assembled from a set of simple ones?
~~~

## Bootstrap

### Settings

Suppose we have a quite good learning algorithm $f(x, D)$ where:
- $D$ is a dataset,
- $x$ is a point of interest,

with **high variance** and **low bias**.`\\[5mm]`
What is the most common way of decreasing variance of mean estimate of a random variable?

### Bootstrap

Let's consider average over multiple datasets:
$$F(x) = \frac{1}{n}\sum_i f(x, D_i) \approx \E_{D \sim P^n(X, Y)} f(x, D) = \hat{F}(x)$$

If $D_i$ are i.i.d:
- $F(x)$ would reduce variance.

If $D_i$ are correlated (via $f(x, D_i)$):

$$\var\left[\frac{1}{n}\sum_i Z_i\right] = \frac{\sigma^2}{n} \left( 1 + (n - 1)\rho\right) \to_{n \to \infty} \rho$$

where:
- $\var\left[Z_i\right] = \sigma^2$, $\rho = \mathrm{corr}(Z_i, Z_j)$ ($i \neq j$).

### Non-parametric bootstrap

Let's approximate $P(X, Y)$ by $\mathbb{U}\left\{ D \right\}$:
- consider $D_i = \{(x^i_j, y^i_j)\}^{m}_{j = 1}$ drawn i.i.d from $D$ with replacement:
  $$F(x) = \sum_{D_i \sim \mathbb{U}^m\left\{ D \right\}} f(x, D_i)$$
- it will reduce variance.

`\vspace{5mm}`

~~~center
\textbf{\Large Seems like model's variance was reduced for 'free', where is the catch?}
~~~

### Parametric bootstrap

If we have a sacred knowledge then we can:
- **using D produce more accurate** $\hat{P}(X, Y)$ **than** $\mathbb{U}^n\left\{D\right\}$

`\vspace{5mm}`
E.g. for regression:
$$D_i = \{(x_i, y_i + \varepsilon)\}^N_{i = 1}$$
where:
- $\varepsilon \sim \mathcal{N}(0, \sigma_\varepsilon)$

### Parametric bootstrap

> \dots the bootstrap mean is approximately a posterior average \dots

`\vspace{5mm}`

For details:`\\`
Hastie, T., Tibshirani, R. and Friedman, J., 2001. The elements of statistical learning, ser., chapter 8

### Bootstrap: a note

Sometimes we can produce more diverse $\{f(x, D_i)\}_i$ by
training on feature subsets.

### Stacking: settings

Bayesian averaging:
- $\zeta$ - variable of our interest (e.g. f(x));
- $\mathcal{M}_m$, $m = 1, \dots, M$ - a candidate models;
- $D$ - training dataset.

~~~multline*
\E(\zeta \mid D) = \\
  \sum_m \E(\zeta \mid \mathcal{M}_m, D) P(\mathcal{M}_m \mid D) = \\
    \sum_m w_m \E(\zeta \mid \mathcal{M}_m, D)
~~~

$w_m = P(\mathcal{M}_m \mid D)$

### Stacking: BIC

$$P(\mathcal{M}_m \mid D) \sim P(\mathcal{M}_m) P(D \mid \mathcal{M}_m)$$
