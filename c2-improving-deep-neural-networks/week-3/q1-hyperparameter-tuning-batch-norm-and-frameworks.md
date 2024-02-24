# Hyperparameter Tuning, Batch Normalization and Programming Frameworks

## Graded Quiz

### Question 1

With a relatively small set of hyperparameters, it is OK to use a grid search. True/False?

- True
- False

Answer: A

Explanation: When the set of hyperparameters is small range like a range for $n_l=1,2,3$ grid search works fine.

### Question 2

In a project with limited computational resources, which three of the following hyperparameters would you choose to tune?

- $\alpha$
- mini-batch size
- The $\beta$ parameter of the momentum in gradient descent.
- $\beta_1$, $\beta_2$ in Adam.
- $\epsilon$ in Adam.

Answer: ABC

Explanation:

- This might be the hyperparameter that most impacts the results of a model.
- This can have a great impact on the results of the cost function, thus it is worth tuning it.
- This hyperparameter can increase the speed of convergence of the training, thus is worth tuning.

### Question 3

Even if enough computational power is available for hyperparameter tuning, it is always better to babysit one model ("Panda" strategy), since this will result in a more custom model. True/False?

- True
- False

Answer: B

Explanation: Although it is possible to create good models using the "Panda" strategy, obtaining better results is more likely using a "caviar" strategy due to the number of tests and the nature of the deep learning process of ideas, code, and experiment.

### Question 4

Knowing that the hyperparameter $\alpha$ should be in the range of $0.00001$ and $1.0$, which of the following is the recommended way to sample a value for $\alpha$?

-

```python
r = -4 * np.random.rand()
alpha = 10 ** r
```

-

```python
r = -5 * np.random.rand()
alpha = 10 ** r
```

-

```python
r = np.random.rand()
alpha = 0.00001 + r * 0.99999
```

-

```python
r = np.random.rand()
alpha = 10 ** r
```

Answer: B

Explanation: This will generate a random value between $10^{-5}$ and $10^0$ chosen randomly in a logarithmic scale.

### Question 5

Finding good hyperparameter values is very time-consuming. So typically you should do it once at the start of the project, and try to find very good hyperparameters so that you don’t ever have to tune them again. True or false?

- True
- False

Answer: B

### Question 6

In batch normalization as presented in the videos, if you apply it on the $l_{\text{th}}$ layer of your neural network, what are you normalizing?

- $a^{[l]}$
- $z^{[l]}$
- $W^{[l]}$
- $b^{[l]}$

Answer: B

### Question 7

When using normalization:

$z^{(i)}_{\text{norm}} = \displaystyle\frac{z^{(i)} - \mu}{\sqrt{\sigma^2 + \epsilon}}$

In case $\sigma$ is too small, the normalization of $z^{(i)}$ may fail since division by 0 may be produced due to rounding errors. True/False?

- True
- False

Answer: B

Explanation: The normalization formula uses a smoothing parameter $\epsilon$ so in $z^{(i)}_{\text{norm}} = \frac{z^{(i)} - \mu}{\sqrt{\sigma^2 + \epsilon}}$, use of the $\epsilon$ parameter prevents the denominator to be 0.

### Question 8

Which of the following statements about $\gamma$ and $\beta$ in Batch Norm are true?

- The optimal values are $\gamma = \sqrt{\sigma^2 + \epsilon}$, and $\beta=\mu$.
- $\beta$ and $\gamma$ are hyperparameters of the algorithm, which we tune via random sampling.
- There is one global value of $\gamma \in \mathbb{R}$ and one global value of $\beta \in \mathbb{R}$ for each layer, and these apply to all hidden units in that layer.
- They set the variance and mean of the linear variable $\tilde{z}^{[l]}$ of a given layer.
- They can be learned using Adam, Gradient descent with momentum, or RMSProp, not just with gradient descent.

Answer: DE

### Question 9

A neural network is trained with Batch Norm. At test time, to evaluate the neural network on a new example you should perform the normalization using $\mu$ and $\sigma^2$ estimated using an exponentially weighted average across mini-batches seen during training. True/false?

- True
- False

Answer: A

Explanation: This is a good practice to estimate the $\mu$ and $\sigma^2$ to use since at test time we might not be predicting over a batch of the same size, or it might even be a single example, thus using the $\mu$ and $\sigma^2$ of a single sample doesn't make sense.

### Question 10

If a project is open-source, it is a guarantee that it will remain open source in the long run and will never be modified to benefit only one company. True/False?

- True
- False

Answer: B

Explanation: To ensure that a project will remain open source in the long run it must have a good governance body too.
