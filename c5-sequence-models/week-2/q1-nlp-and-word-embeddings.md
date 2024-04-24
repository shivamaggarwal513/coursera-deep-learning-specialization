# Natural Language Processing and Word Embeddings

## Graded Quiz

### Question 1

Suppose you learn a word embedding for a vocabulary of 60,000 words. Then the embedding vectors could be 60,000 dimensional, so as to capture the full range of variation and meaning in those words. True/False?

- True
- False

Answer: B

Explanation: No, the dimension of word vectors is usually smaller than the size of the vocabulary. Most common sizes for word vectors range between 50 and 1000.

### Question 2

What is t-SNE?

- A linear transformation that allows us to solve analogies on word vectors.
- A non-linear dimensionality reduction technique.
- A supervised learning algorithm for learning word embeddings.
- An open source sequence modelling library.

Answer: B

Explanation: t-SNE is a non-linear dimensionality reduction technique.

### Question 3

Suppose you download a pre-trained word embedding which has been trained on a huge corpus of text. You then use this word embedding to train an RNN for a language task of recognizing if someone is happy from a short snippet of text, using a small training set.

| x (input text)        | y (happy?) |
| --------------------- | ---------- |
| Having a great time!  | 1          |
| I'm sad it's raining. | 0          |
| I'm feeling awesome!  | 1          |

Even if the word "wonderful" does not appear in your small training set, what label might be reasonably expected for the input text "I feel wonderful!"?

- $y=1$
- $y=0$

Answer: A

Explanation: Word vectors empower your model with an incredible ability to generalize. The vector for "wonderful" would contain a negative/unhappy connotation which will probably make your model classify the sentence as a "1".

### Question 4

Which of these equations do you think should hold for a good word embedding?

- $e_\text{boy} - e_\text{brother} \approx e_\text{girl} - e_\text{sister}$
- $e_\text{boy} - e_\text{girl} \approx e_\text{sister} - e_\text{brother}$
- $e_\text{boy} - e_\text{girl} \approx e_\text{brother} - e_\text{sister}$
- $e_\text{boy} - e_\text{brother} \approx e_\text{sister} - e_\text{girl}$

Answer: AC

### Question 5

Let $E$ be an embedding matrix, and let $o_{1234}$ be a one-hot vector corresponding to word 1234. Then to get the embedding of word 1234, why don't we call $E * o_{1234}$ in Python?

- It is computationally wasteful.
- The correct formula is $E^T * o_{1234}$.
- This doesn't handle unknown words `<UNK>`.
- None of the above: calling the Python snippet as described above is fine.

Answer: A

Explanation: The element-wise multiplication will be extremely inefficient.

### Question 6

When learning word embeddings, we pick a given word and try to predict its surrounding words or vice versa. True/False?

- True
- False

Answer: A

Explanation: Word embeddings are learned by picking a given word and trying to predict its surrounding words or vice versa.

### Question 7

In the word2vec algorithm, you estimate $P(t \vert c)$, where $t$ is the target word and $c$ is a context word. How are $t$ and $c$ chosen from the training set?

- $c$ is the sequence of all the words in the sentence before $t$.
- $c$ and $t$ are chosen to be nearby words.
- $c$ is the one word that comes immediately before $t$.
- $c$ is the sequence of several words immediately before $t$.

Answer: B

Explanation: $c$ is chosen from the training set to be nearby words.

### Question 8

Suppose you have a 10,000 word vocabulary, and are learning 100-dimensional word embeddings. The word2vec model uses the following softmax function:

$$P(t \vert c) = \frac{e^{\theta^T_t e_c}}{\sum_{t'=1}^{10000} e^{\theta^T_{t'} e_c}}$$

Which of these statements are correct?

- $\theta_t$ and $e_c$ are both 100 dimensional vectors.
- $\theta_t$ and $e_c$ are both 10,000 dimensional vectors.
- $\theta_t$ and $e_c$ are both trained with an optimization algorithm such as Adam or gradient descent.
- After training, we should expect $\theta_t$ to be very close to $e_c$ when $t$ and $c$ are the same word.

Answer: AC

### Question 9

Suppose you have a 10,000 word vocabulary, and are learning 500-dimensional word embeddings. The GloVe model minimizes this objective:

$$\min \sum_{i=1}^{10000} \sum_{j=1}^{10000} f(X_{ij})(\theta^T_i e_j + b_i + b_j' - \log X_{ij})^2$$

Which of these statements are correct?

- $\theta_i$ and $e_j$ should be initialized to 0 at the beginning of training.
- $\theta_i$ and $e_j$ should be initialized randomly at the beginning of training.
- $X_{ij}$ is the number of times word $i$ appears in the context of word $j$.
- The weighting function $f(X_{ij})$ must satisfy $f(0) = 0$.

Answer: BCD

Explanation: The weighting function helps prevent learning only from extremely common word pairs.

### Question 10

You have trained word embeddings using a text dataset of $t_1$ words. You are considering using these word embeddings for a language task, for which you have a separate labeled dataset of $t_2$ words. Keeping in mind that using word embeddings is a form of transfer learning, under which of these circumstances would you expect the word embeddings to be helpful?

- When $t_1$ is equal to $t_2$.
- When $t_1$ is larger than $t_2$.
- When $t_1$ is smaller than $t_2$.

Answer: B

Explanation: Transfer embeddings to new tasks with smaller training sets.
