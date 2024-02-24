# Practical Aspects of Deep Learning

## Graded Quiz

### Question 1

If you have 20,000,000 examples, how would you split the train/dev/test set?

- 99% train, 0.5% dev, 0.5% test
- 60% train, 20% dev, 20% test
- 90% train, 5% dev, 5% test

Answer: A

Explanation: Given the size of the dataset, 0.5% of the samples are enough to get a good estimate of how well the model is doing.

### Question 2

When designing a neural network to detect if a house cat is present in the picture, 500,000 pictures of cats were taken by their owners. These are used to make the training, dev and test sets. It is decided that to increase the size of the test set, 10,000 new images of cats taken from security cameras are going to be used in the test set. Which of the following is true?

- This will increase the bias of the model so the new images shouldn't be used.
- This will reduce the bias of the model and help improve it.
- This will be harmful to the project since now dev and test sets have different distributions.

Answer: C

Explanation: The quality and type of images are quite different thus we can't consider that the dev and the test sets came from the same distribution.

### Question 3

If your Neural Network model seems to have high variance, what of the following would be promising things to try?

- Get more training data
- Get more test data
- Add regularization
- Make the Neural Network deeper
- Increase the number of units in each hidden layer

Answer: AC

Explanation: High variance means overfitting. It can be reduced by getting more training data and by adding regularization.

### Question 4

You are working on an automated check-out kiosk for a supermarket and are building a classifier for apples, bananas, and oranges. Suppose your classifier obtains a training set error of 19% and a dev set error of 21%. Which of the following are promising things to try to improve your classifier? (Suppose the human error is approximately 0%)

- Get more training data
- Increase the regularization parameter `lambda`
- Use a bigger network.

Answer: C

Explanation: This is a case of high bias (underfitting). Using a bigger network can be helpful to reduce the bias of the model, and then we can start trying to reduce the high variance if this happens.

### Question 5

What is weight decay?

- The process of gradually decreasing the learning rate during training.
- Gradual corruption of the weights in the neural network if it is trained on noisy data.
- A technique to avoid vanishing gradient by imposing a ceiling on the values of the weights.
- A regularization technique (such as L2 regularization) that results in gradient descent shrinking the weights on every iteration.

Answer: D

### Question 6

The regularization hyperparameter must be set to zero during testing to avoid getting random results. True/False?

- True
- False

Answer: B

Explanation: The regularization parameter affects how the weights change during training, this means during backpropagation. It has no effect during the forward propagation that is when predictions for the test are made.

### Question 7

With the inverted dropout technique, at test time:

- You apply dropout (randomly eliminating units) but keep the `1 / keep_prob` factor in the calculations used in training.
- You do not apply dropout (do not randomly eliminate units) but keep the `1 / keep_prob` factor in the calculations used in training.
- You do not apply dropout (do not randomly eliminate units) and do not keep the `1 / keep_prob` factor in the calculations used in training.
- You apply dropout (randomly eliminating units) and do not keep the `1 / keep_prob` factor in the calculations used in training.

Answer: C

### Question 8

Decreasing the parameter `keep_prob` from (say) 0.6 to 0.4 will likely cause the following:

- Reducing the regularization effect.
- Increasing the regularization effect.
- Causing the neural network to have a higher variance.

Answer: B

Explanation: This will make the dropout have a higher probability of eliminating a node in the neural network, increasing the regularization effect.

### Question 9

Which of the following actions increase the regularization of a model?

- Use Xavier initialization
- Decrease the value of `keep_prob` in dropout.
- Increase the value of the hyperparameter `lambda`.
- Decrease the value of the hyperparameter `lambda`.
- Increase the value of `keep_prob` in dropout.

Answer: BC

Explanation:

- When decreasing the `keep_prob` value, the probability that a node gets discarded during training is higher, thus increasing the regularization effect.
- When increasing the hyperparameter `lambda`, we increase the effect of the L2 penalization.

### Question 10

Suppose that a model uses, as one feature, the total number of kilometers walked by a person during a year, and another feature is the height of the person in meters. What is the most likely effect of normalization of the input data?

- It will make the data easier to visualize.
- It will increase the variance of the model.
- It won't have any positive or negative effects.
- It will make the training faster.

Answer: D

Explanation: Since the difference between the ranges of the features is very different, this will likely cause the process of gradient descent to oscillate, making the optimization process longer. Normalization can prevent that.
