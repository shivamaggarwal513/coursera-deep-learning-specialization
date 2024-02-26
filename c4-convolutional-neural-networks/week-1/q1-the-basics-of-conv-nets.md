# The Basics of ConvNets

## Graded Quiz

### Question 1

What do you think applying this filter to a grayscale image will do?

$$
\begin{bmatrix}
-1 & -1 &  2 \\
-1 &  2 &  1 \\
 2 &  1 &  1 \\
\end{bmatrix}
$$

- Detect $45^o$ edges
- Detect horizontal edges
- Detect vertical edges
- Detect image contrast

Answer: A

Explanation: Notice that there is a high delta between the values in the top left part and the ones in the bottom right part. When convolving this filter on a grayscale image, the edges forming a $45^o$ angle with the horizontal will be detected.

### Question 2

Suppose your input is a $128$ by $128$ color (RGB) image, and you are not using a convolutional network. If the first hidden layer has $64$ neurons, each one fully connected to the input, how many parameters does this hidden layer have (including the bias parameters)?

- $3145728$
- $3145792$
- $1048640$
- $1048576$

Answer: B

Explanation: The number of inputs for each unit is $128 \times 128 \times 3 = 49152$ since the input image is RGB, so we need $49152 \times 64 = 3145728$ parameters for the weights and $64$ parameters for the bias, thus $3145728 + 64 = 3145792$ learnable parameters.

### Question 3

Suppose your input is a $300$ by $300$ color (RGB) image, and you use a convolutional layer with $100$ filters that are each $5 \times 5$. How many parameters does this hidden layer have (including the bias parameters)?

- $2501$
- $2600$
- $7500$
- $7600$

Answer: D

Explanation: Since the input volume has $3$ channels, each filter has $5 \times 5 \times 3 + 1 = 76$ weights including the bias, thus the total is $76 \times 100 = 7600$ learnable parameters.

### Question 4

You have an input volume that is $127 \times 127 \times 16$, and convolve it with $32$ filters of $5 \times 5$, using a stride of $2$ and no padding. What is the output volume?

- $123 \times 123 \times 16$
- $62 \times 62 \times 16$
- $123 \times 123 \times 32$
- $62 \times 62 \times 32$

Answer: D

Explanation: Using the formula $n^{[l]}_H = \Big\lfloor\frac{n^{[l-1]}_H + 2p - f}{s} + 1\Big\rfloor$ with $n^{[l-1]}_H=127$, $p=0$, $f=5$, and $s=2$, we get $n^{[l]}_H=62$.

### Question 5

You have an input volume that is $15 \times 15 \times 8$, and pad it using "pad = 2". What is the dimension of the resulting volume (after padding)?

- $19 \times 19 \times 12$
- $17 \times 17 \times 10$
- $19 \times 19 \times  8$
- $17 \times 17 \times  8$

Answer: C

Explanation: Padding is applied over the height and the width of the input image. If the padding is $2$, you add $4$ to the height dimension and $4$ to the width dimension.

### Question 6

You have a volume that is $64 \times 64 \times 32$, and convolve it with $40$ filters of $9 \times 9$, and stride $1$. You want to use a "same" convolution. What is the padding?

- $0$
- $6$
- $8$
- $4$

Answer: D

Explanation: When using a padding of $4$ the output volume has $n_H=\frac{64+2\times 4-9}{1}+1 = 64$. For "same" convolution, padding is given as $p=\frac{f-1}{2}=\frac{9-1}{2}=4$.

### Question 7

You have an input volume that is $66 \times 66 \times 21$, and apply max pooling with a stride of $3$ and a filter size of $3$. What is the output volume?

- $66 \times 66 \times  7$
- $22 \times 22 \times 21$
- $22 \times 22 \times  7$
- $21 \times 21 \times 21$

Answer: B

Explanation: Using the formula $n^{[l]}_H = \Big\lfloor\frac{n^{[l-1]}_H + 2p - f}{s} + 1\Big\rfloor$ with $n^{[l-1]}_H=66$, $p=0$, $f=3$, and $s=3$, we get $n^{[l]}_H=22$.

### Question 8

Because pooling layers do not have parameters, they do not affect the backpropagation (derivatives) calculation. True/False?

- True
- False

Answer: B

Explanation: Everything that influences the loss should appear in the backpropagation because we are computing derivatives. In fact, pooling layers modify the input by choosing one value out of several values in their input volume. Also, to compute derivatives for the layers that have parameters (Convolutions, Fully-Connected), we still need to backpropagate the gradient through the Pooling layers.

### Question 9

In lecture we talked about "parameter sharing" as a benefit of using convolutional networks. Which of the following statements about parameter sharing in ConvNets are true?

- It allows a feature detector to be used in multiple locations throughout the whole input image/input volume.
- It reduces the total number of parameters, thus reducing overfitting.
- It allows parameters learned for one task to be shared even for a different task (transfer learning).
- It allows gradient descent to set many of the parameters to zero, thus making the connections sparse.

Answer: AB

Explanation:

- By sliding a filter of parameters over the entire input volume, we make sure a feature detector can be used in multiple locations.
- A convolutional layer uses parameter sharing and usually has a lot less parameters than a fully-connected layer.

### Question 10

In lecture we talked about "sparsity of connections" as a benefit of using convolutional layers. What does this mean?

- Regularization causes gradient descent to set many of the parameters to zero.
- Each layer in a convolutional network is connected only to two other layers.
- Each activation in the next layer depends on only a small number of activations from the previous layer.
- Each filter is connected to every channel in the previous layer.

Answer: C

Explanation: Weight sharing reduces significantly the number of parameters in a neural network, and sparsity of connections allows us to use a smaller number of inputs thus reducing even further the number of parameters. It makes possible to train a network with smaller training sets without having overfitting.
