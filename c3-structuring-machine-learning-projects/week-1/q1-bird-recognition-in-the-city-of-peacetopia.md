# Bird Recognition in the City of Peacetopia

## Graded Quiz

### Problem Statement

This example is adapted from a real production application, but with details disguised to protect confidentiality.

![Peacetopia](./images/q1-peacetopia.png)

You are a famous researcher in the City of Peacetopia. The people of Peacetopia have a common characteristic: they are afraid of birds. To save them, you have to build an algorithm that will detect any bird flying over Peacetopia and alert the population.

The City Council gives you a dataset of 10,000,000 images of the sky above Peacetopia, taken from the city’s security cameras. They are labeled:

- $y = 0$: There is no bird on the image
- $y = 1$: There is a bird on the image

Your goal is to build an algorithm able to classify new images taken by security cameras from Peacetopia.

There are a lot of decisions to make:

- What is the evaluation metric?
- How do you structure your data into train/dev/test sets?

#### Metric of success

The City Council tells you the following that they want an algorithm that

1. Has high accuracy.
2. Runs quickly and takes only a short time to classify a new image.
3. Can fit in a small amount of memory, so that it can run in a small processor that the city will attach to many different security cameras.

#### Question 1

Having three evaluation metrics makes it harder for you to quickly choose between two different algorithms, and will slow down the speed with which your team can iterate. True/False?

- True
- False

Answer: A

Explanation: More than one metric expands the choices and tradeoffs you have to decide for each with unknown effects on the other two.

#### Question 2

The city asks for your help in further defining the criteria for accuracy, runtime, and memory. How would you suggest they identify the criteria?

- Suggest that they purchase more infrastructure to ensure the model runs quickly and accurately.
- Suggest to them that they define which criterion is most important. Then, set thresholds for the other two.
- Suggest to them that they focus on whichever criterion is important and then eliminate the other two.

Answer: B

Explanation: The thresholds provide a way to evaluate models head to head.

#### Question 3

The essential difference between an optimizing metric and satisficing metrics is the priority assigned by the stakeholders. True/False?

- True
- False

Answer: B

Explanation: Satisficing metrics have thresholds for measurement and an optimizing metric is unbounded.

#### Question 4

With 10,000,000 data points, what is the best option for train/dev/test splits?

- Train: 60%, Dev: 10%, Test: 30%
- Train: 95%, Dev: 2.5%, Test: 2.5%
- Train: 60%, Dev: 30%, Test: 10%
- Train: 33.3%, Dev: 33.3%, Test: 33.3%

Answer: B

Explanation: The size of the data set allows for bias and variance evaluation with smaller data sets.

#### Question 5

After setting up your train/dev/test sets, the City Council comes across another 1,000,000 images, called the "citizens' data". Apparently the citizens of Peacetopia are so scared of birds that they volunteered to take pictures of the sky and label them, thus contributing these additional 1,000,000 images. These images are different from the distribution of images the City Council had originally given you, but you think it could help your algorithm.

Notice that adding this additional data to the training set will make the distribution of the training set different from the distributions of the dev and test sets.

Is the following statement true or false?

"You should not add the citizens' data to the training set, because if the training distribution is different from the dev and test sets, then this will not allow the model to perform well on the test set."

- True
- False

Answer: B

Explanation: Sometimes we'll need to train the model on the data that is available, and its distribution may not be the same as the data that will occur in production. Also, adding training data that differs from the dev set may still help the model improve performance on the dev set. What matters is that the dev and test set have the same distribution.

#### Question 6

One member of the City Council knows a little about machine learning and thinks you should add the 1,000,000 citizens' data images to the dev set. You object because:

- A bigger test set will slow down the speed of iterating because of the computational expense of evaluating models on the test set.
- The dev set no longer reflects the distribution of data (security cameras) you most care about.
- This would cause the dev and test set distributions to become different. This is a bad idea because you're not aiming where you want to hit.
- The 1,000,000 citizens' data images do not have a consistent $x \to y$ mapping as the rest of the data.

Answer: BC

Explanation:

- The performance of the model should be evaluated on the same distribution of images it will see in production.
- Adding a different distribution to the dev set will skew bias.

#### Question 7

You train a system, and the train/dev set errors are 3.5% and 4.0% respectively. You decide to try regularization to close the train/dev accuracy gap. Do you agree?

- No, because this shows your variance is higher than your bias.
- Yes, because having a 4.0% training error shows you have a high bias.
- Yes, because this shows your bias is higher than your variance.
- No, because you do not know what the human performance level is.

Answer: D

Explanation: You need to know what the human performance level is to estimate avoidable bias.

#### Question 8

You want to define what human-level performance is to the city council. Which of the following is the best answer?

- The average of regular citizens of Peacetopia (1.2%).
- The average performance of all their ornithologists (0.5%).
- The performance of their best ornithologist (0.3%).
- The average of all the numbers above (0.66%).

Answer: C

Explanation: The best human performance is closest to Bayes' error.

#### Question 9

Assuming best case scenario, arrange the learning algorithm's performance with human-level performance and Bayes error in correct order from highest to lowest error.

- Human level performance, learning algorithm, Bayes error.
- Learning algorithm, Human level performance, Bayes error.
- Human level performance, Bayes error, learning algorithm.
- Bayes error, learning algorithm, Human level performance.

Answer: A

Explanation: A learning algorithm's performance can be better than human-level performance but it can never be better than Bayes error.

#### Question 10

You find that a team of ornithologists debating and discussing an image gets an even better 0.1% performance, so you define that as "human-level performance". After working further on your algorithm, you end up with the following:

|                         |      |
| ----------------------- | ---- |
| Human-level performance | 0.1% |
| Training set error      | 2.0% |
| Dev set error           | 2.1% |

Based on the evidence you have, which two of the following four options seem the most promising to try?

- Train a bigger model to try to do better on the training set.
- Try decreasing regularization.
- Get a bigger training set to reduce variance.
- Try increasing regularization

Answer: AB

Explanation: This is the case of high bias or underfitting.

#### Question 11

After running your model with the test set you find it is a 7.0% error compared to a 2.1% error for the dev set and 2.0% for the training set. What can you conclude?

- You have overfitted to the dev set.
- You have underfitted to the dev set.
- Try decreasing regularization for better generalization with the dev set.
- You should try to get a bigger dev set.

Answer: AD

Explanation: The dev set performance versus the test set indicates it is overfitting.

#### Question 12

After working on this project for a year, you finally achieve:

|                         |       |
| ----------------------- | ----- |
| Human-level performance | 0.10% |
| Training set error      | 0.05% |
| Dev set error           | 0.05% |

Which of the following are likely?

- There is still avoidable bias.
- The model has recognized emergent features that humans cannot. (Chess and Go for example)
- This is a statistical anomaly (or must be the result of statistical noise) since it should not be possible to surpass human-level performance.
- Pushing to even higher accuracy will be slow because you will not be able to easily identify sources of bias.

Answer: BD

Explanation:

- When Google beat the world Go champion, it was recognized that it was making deeper moves than humans.
- Exceeding human performance means you are close to Bayes error.

#### Question 13

Your system is now very accurate but has a higher false negative rate than the City Council of Peacetopia would like. What is your best next step?

- Expand your model size to account for more corner cases.
- Pick false negative rate as the new metric, and use this new metric to drive all further development.
- Reset your "target" (metric) for the team and tune to it.
- Look at all the models you've developed during the development process and find the one with the lowest false negative error rate.

Answer: C

Explanation: The target has shifted so an updated metric is required.

#### Question 14

Over the last few months, a new species of bird has been slowly migrating into the area, so the performance of your system slowly degrades because your data is being tested on a new type of data. There are only 1,000 images of the new species. The city expects a better system from you within the next 3 months. Which of these should you do first?

- Augment your data to increase the images of the new bird.
- Put the new species' images in training data to learn their features.
- Add pooling layers to downsample features to accommodate the new species.
- Split them between dev and test and re-tune.

Answer: A

Explanation: A sufficient number of images is necessary to account for the new species.

#### Question 15

The City Council thinks that having more Cats in the city would help scare off birds. They are so happy with your work on the Bird detector that they also hire you to build a Cat detector. You have a huge dataset of 100,000,000 cat images. Training on this data takes about two weeks. Which of the statements do you agree with?

- Reducing the model complexity will allow the use of the larger data set but preserve accuracy.
- This significantly impacts iteration speed.
- Lowering the number of images will reduce training time and likely allow for an acceptable trade-off between iteration speed and accuracy.
- Having built a good Bird detector, you should be able to take the same model and hyperparameters and just apply it to the Cat dataset, so there is no need to iterate.

Answer: BC

Explanation:

- This training time is an absolute constraint on iteration.
- There is a sweet spot that allows development at a reasonable rate without significant accuracy loss.
