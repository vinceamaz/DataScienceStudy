# Neural Network Crash Course

## 1 A single neuron

A neuron is a generic computational unit that takes n inputs and produces a single output. One of
the most popular choices for neurons is the "sigmoid"  unit.
$$
a = \frac{1}{1+exp(-[w^T\ b] \cdot [x\ 1])}
$$
where

* $a$ is the scalar activation output
* $x$ is the input
* $w$ is the weight matrix
* $b$ is the bias term

The image below captures how in a sigmoid neuron, the input vector $x$ is first scaled, summed, added to a bias unit, and then passed to the squashing sigmoid function.

<img src="C:\Users\vince\AppData\Roaming\Typora\typora-user-images\image-20200117192211219.png" alt="image-20200117192211219" style="zoom:50%;" />

## 2 A single layer of neurons

This image captures how multiple sigmoid units are stacked on the right, all of which receive the same input $x$.

<img src="C:\Users\vince\AppData\Roaming\Typora\typora-user-images\image-20200117193513095.png" alt="image-20200117193513095" style="zoom:50%;" />

The representation of the input $x$ remains unchanged from the previous single neuron example. However, the weight matrix has become
$$
W=\left[\begin{array}{cc}
{-} & {w^{(1) T}} & {-} \\
{} & {\cdots} \\
{-} & {w^{(m) T}} & {-} 
\end{array}\right] \in \mathbb{R}^{m \times n}
$$
The bias term has become
$$
b=\left[\begin{array}{c}
{b_{1}} \\
{\vdots} \\
{b_{m}}
\end{array}\right] \in \mathbb{R}^{m}
$$
The activation output can be written as
$$
\sigma(z)=\left[\begin{array}{c}
{\frac{1}{1+\exp \left(z_{1}\right)}} \\
{\vdots} \\
{\frac{1}{1+\exp \left(z_{m}\right)}}
\end{array}\right]
$$
We can now write the output of scaling and biases as
$$
z = Wx+b
$$
The activations of the sigmoid function can then be written as
$$
a=\left[\begin{array}{c}
{a^{(1)}} \\
{\vdots} \\
{a^{(m)}}
\end{array}\right] = \sigma(z) = \sigma(Wx+b)
$$

## 3 Intuition of hidden layer

Let us consider the following named entity recognition (NER) problem in NLP as an example:
$$
\text{Museums in Paris are amazing}
$$
Here, we want to classify whether or not the center word "Paris" is a named-entity. In such cases, it is very likely that we would not just want to capture the presence of words in the window of word vectors but some other interactions between the words in order to make the classification.

For instance, maybe it should matter that "Museums" is the first word only if "in" is the second word. Such non-linear decisions can often not be captured by inputs fed directly to a Softmax function but instead require the scoring of the intermediate layer. We can thus use another matrix $U \in R^{m \times 1}$ to generate an unnormalized score for a classification task from the activations
$$
s=U^Ta=U^Tf(Wx+b)
$$
where f is the activation function.

<img src="C:\Users\vince\AppData\Roaming\Typora\typora-user-images\image-20200117195345880.png" alt="image-20200117195345880" style="zoom: 33%;" />

**Dimensions for a single hidden layer neural network:**

If we represent each word using a 4-dimensional word vector and we use a 5-word window as input, then the input $x \in \mathbb{R}^{20}$. If we use 8 sigmoid units in the hidden layer and generate 1 score output from the activations, then
$$
W \in \mathbb{R}^{8 \times 20},\ b \in \mathbb{R}^{8},\ z \in \mathbb{R}^{8},\ U \in \mathbb{R}^{8}\ s \in \mathbb{R}
$$
and
$$
z = Wx + b \\
a = \sigma(z) \\
s = U^Ta
$$

## 4 Objective Function

In this example, we will discuss a popular error metric known as the `maximum margin objective`. The
idea behind using this objective is to ensure that the score computed for "true" labeled data points is higher than the score computed for "false" labeled data points.

Using the previous example, if we call the score computed for the "true" labeled window 
$$
\text{Museums in Paris are amazing}
$$
as $s$ and the score computed for the "false" labeled window 
$$
\text{Not all museums in Paris}
$$
as $s_c$ (subscripted as $c$ to signify that the window is "corrupt").

Then, our objective function would be to maximize $(s − s_c)$ or to minimize $(s_c − s)$. However, we modify our objective to ensure that error is only computed if $s_c > s ⇒ (s_c − s) > 0$. The intuition behind doing this is that we only care the the "true" data point have a higher score than the "false" data point and that the rest does not matter. Thus, we want our error to be $(s_c − s)$ if $s_c > s$ else 0. Thus, our optimization objective is now:
$$
\text{minimize}\ J = \text{max}(s_c - s, 0)
$$
However, the above optimization objective is risky in the sense that it does not attempt to create a margin of safety. We would want the "true" labeled data point to score higher than the "false" labeled data point by some positive margin ∆ (more than just 0). In other words, we would want error to be calculated if $(s − s_c < \Delta)$ and not just when $(s − s_c < 0)$. Thus, we modify the optimization objective:
$$
\text{minimize}\ J = \text{max}(\Delta +s_c - s, 0)
$$
and the margin can be 1. Therefore, we can re-write it as:
$$
\text{minimize}\ J = \text{max}(1 +s_c - s, 0)
$$
where

* $s_c = U^Tf(Wx_c+b)$
* $s = U^Tf(Wx+b)$

## 5 Training with Backpropagation – Elemental

In this section we discuss how we train the different parameters in the model when the cost $J = \text{max}(1 +s_c - s, 0)$ is positive.  No parameter updates are necessary if the cost is 0.

Since we typically update parameters using gradient descent (or a variant such as SGD),
we typically need the gradient information for any parameter as required in the update equation:
$$
\theta^{(t+1)}=\theta^{(t)}-\alpha \nabla_{\theta^{(t)}} J
$$
Backpropagation is technique that allows us to use the chain rule of differentiation to calculate loss gradients for any parameter used in the feed-forward computation on the model.

<img src="C:\Users\vince\AppData\Roaming\Typora\typora-user-images\image-20200117202659589.png" alt="image-20200117202659589" style="zoom:50%;" />

The figure above is a 4-2-1 neural network where neuron $j$ on layer $k$ receives input $z^{(k)}_j$and produces activation output $\alpha^{(k)}_j$. Here, we use a neural network with a single hidden layer and a single unit output.

* $x_i$ is an input to the neural network
* $s$ is the output of the neural network
* Each layer (including the input and output layers) has neurons which receive an input and produce an output. The $j^{-th}$ neuron of layer $k$ receives the scalar input $z^{(k)}_j$ and produces the scalar activation output $a^{(k)}_j$

* $\delta ^{(k)}_j$ denotes the backpropagated error calculated at $z^{(k)}_j$
* Layer 1 refers to the input layer and not the first hidden layer. For the input layer, $x_j = z^{(1)}_j = a^{(1)_j}$
* $W^{(k)}$ is the transfer matrix that maps the output from the $k^{-th}$ layer to the input to the $(k_1)^{(-th)}$ layer. Thus, $W^{(1)} = W$ and $W^{(2)} = U$ to put this new generalized notation in perspective of Section 1.3.

### 5.1 Backpropagation with chain rule

<img src="C:\Users\vince\AppData\Roaming\Typora\typora-user-images\image-20200117203844370.png" alt="image-20200117203844370" style="zoom:50%;" />

Suppose the cost $J = (1 +s_c - s)$ is positive and we want to perform the update of parameter $W^{(1)}_{14}$, we must realize that $W^{(1)}_{14}$ only contributes to $z^{(2)}_1$ and thus $a^{(2)}_1$. This fact is crucial to understanding backpropagation - backpropagated gradients are only affected by values they contribute to. $a^{(2)}_1$ is consequently used in the forward computation of score by multiplication with $W^{(2)}_1$. 

We can see from the max-margin loss that:
$$
\frac{\partial J}{\partial s}=-\frac{\partial J}{\partial s_{c}}=-1
$$
Therefore we will work with $\frac{\partial s}{\partial W^{(1)}_{ij}}$ here for simplicity. Thus,
$$
\begin{aligned}
\frac{\partial s}{\partial W_{i j}^{(1)}} &=\frac{\partial W^{(2)} a^{(2)}}{\partial W_{i j}^{(1)}}=\frac{\partial W_{i}^{(2)} a_{i}^{(2)}}{\partial W_{i j}^{(1)}}=W_{i}^{(2)} \frac{\partial a_{i}^{(2)}}{\partial W_{i j}^{(1)}} \\
\\
\Rightarrow W_{i}^{(2)} \frac{\partial a_{i}^{(2)}}{\partial W_{i j}^{(1)}} &=W_{i}^{(2)} \frac{\partial a_{i}^{(2)}}{\partial z_{i}^{(2)}} \frac{\partial z_{i}^{(2)}}{\partial W_{i j}^{(1)}} \\
&=W_{i}^{(2)} \frac{f\left(z_{i}^{(2)}\right)}{\partial z_{i}^{(2)}} \frac{\partial z_{i}^{(2)}}{\partial W_{i j}^{(1)}} \\
&=W_{i}^{(2)} f^{\prime}\left(z_{i}^{(2)}\right) \frac{\partial z_{i}^{(2)}}{\partial W_{i j}^{(1)}} \\
&=W_{i}^{(2)} f^{\prime}\left(z_{i}^{(2)}\right) \frac{\partial}{\partial W_{i j}^{(1)}}\left(b_{i}^{(1)}+a_{1}^{(1)} W_{i 1}^{(1)}+a_{2}^{(1)} W_{i 2}^{(1)}+a_{3}^{(1)} W_{i 3}^{(1)}+a_{4}^{(1)} W_{i 4}^{(1)}\right)\\
&=W_{i}^{(2)} f^{\prime}\left(z_{i}^{(2)}\right) \frac{\partial}{\partial W_{i j}^{(1)}}\left(b_{i}^{(1)}+\sum_{k} a_{k}^{(1)} W_{i k}^{(1)}\right)\\
&=W_{i}^{(2)} f^{\prime}\left(z_{i}^{(2)}\right) a_{j}^{(1)}\\
&=\delta_{i}^{(2)} \cdot a_{j}^{(1)}
\end{aligned}
$$
We can see that the gradient reduces to the product $\delta^{(2)}_i \cdot a^{(1)}_j$ 

* $\delta^{(2)}_i$ is essentially the error propagating backwards from the $i^{-th}$ neuron in layer 2 
* The result of $a^{(1)}_j$ multiplied by $W_{ij}$ is an input fed to $i^{-th}$ neuron in layer 2. For example, the result of $a^{(1)}_4$ multiplied by $W^{(1)}_{14}$ is the input fed into the $1^{-th}$ neuron in layer 2.

### 5.2 Error sharing/distribution interpretation of backpropagation

<img src="C:\Users\vince\AppData\Roaming\Typora\typora-user-images\image-20200117203844370.png" alt="image-20200117203844370" style="zoom:50%;" />

 Say we were to update $W^{(1)}_{14}$:

1. We start with an error signal of 1 propagating backwards from $a^{(3)}_1$.
2. We then multiply this error by the local gradient of the neuron which maps $z^{(3)}_1$ to $a^{(3)}_1$. This happens to be 1 in this case and thus, the error is still 1. This is now known as $\delta^{(3)}_1 = 1$.
3. At this point, the error signal of 1 has reached $z^{(3)}_1$. We now need to distribute the error signal so that the "fair share" of the error reaches to $a^{(2)}_1$.
4. This amount is the $(\text{error signal at}\ z^{(3)}_1 = \delta^{(3)}_1) \times W^{(2)}_1 = 1 \times W^{(2)}_1 = W^{(2)}_1$. Thus, the error at $a^{(2)}_1 = W^{(2)}_1$. This is because $\frac{\partial z^{(3)}_1}{\partial a^{(2)}_1} = W^{(2)}_1$.
5. As we did in step 2, we need to move the error across the neuron which maps $z^{(2)}_1$ to $a^{(2)}_1$. We do this by multiplying the error signal at $a^{(2)}_1$ by the local gradient of the neuron which happens to be $f'(z^{(2)}_1)$. This is because $\frac{\partial a^{(2)}_1}{\partial z^{(2)}_1}=f'(z^{(2)}_1)$.
6. Thus, the error signal at $z^{(2)}_1$ is $f'(z^{(2)}_1) W^{(2)}_1$. This is known as $\delta^{(2)}_1$.
7. Finally, we need to distribute the "fair share" of the error to $W^{(1)}_{14}$ by simply multiplying it by the input it was responsible for forwarding, which happens to be $a^{(1)}_4$. This is because $\frac{\partial z^{(2)}_1}{\partial W^{(1)}_{14}} = a^{(2)}_4$.
8. Thus, the gradient of the loss with respect to $W^{(1)}_{14}$ is calculated to be $a^{(1)}_4 f'(z^{(2)}_1) W^{(2)}_1$.



We can calculate error gradients with respect to a parameter in the network using either the chain rule of differentiation or using an error sharing and distributed flow approach – both of these approaches happen to do the exact same thing but it might be helpful to think about them one way or another.

### 5.3 Bias Updates

Bias terms (such as $b^{(1)}_1$) are mathematically equivalent to other weights contributing to the neuron input ($z^{(2)}_1$) as long as the input being forwarded is 1. As such, the bias gradients for neuron $i$ on layer $k$ is simply $\delta^{(k)}_i$. For instance, if we were updating $b^{(1)}_1$ instead of $W^{(1)}_{14}$ above, the gradient would simply be $f'(z^{(2)}_1) W^{(2)}_1$. This is because $\frac{\partial z^{(2)}_1}{\partial b^{(1)}_1} = 1$. 

### 5.4 Generalized steps to propagate $\delta^{(k)}$ to $\delta^{(k-1)}$

1. We have error $\delta^{(k)}$ propagating backwards from $z^{(k)}_i$, i.e. neuron $i$ at layer $k$.

<img src="C:\Users\vince\AppData\Roaming\Typora\typora-user-images\image-20200117221700039.png" alt="image-20200117221700039" style="zoom: 50%;" />

2. We propagate this error backwards to $a^{(k-1)}_j$ by multiplying $\delta^{(k)}$ by the path weight $W^{(k-1)}_{ij}$.

3. Thus, the error received at $a^{(k-1)}_j$ is $\delta^{(k)}_i W ^{(k-1)}_{ij}$.

4. However, $a^{(k-1)}_j$ may have been forwarded to multiple nodes in the next layer. It should receive responsibility for errors propagating backward from node $m$ in layer $k$ too, using the exact same mechanism.

   <img src="C:\Users\vince\AppData\Roaming\Typora\typora-user-images\image-20200117222020109.png" alt="image-20200117222020109" style="zoom:50%;" />

5. Thus, error received at $a^{(k-1)}_j$ is $\delta^{(k)}_i W ^{(k-1)}_{ij} + \delta^{(k)}_m W ^{(k-1)}_{mj}$.
6. In fact, we can generalize this to be $\sum_i \delta^{(k)}_i W ^{(k-1)}_{ij}$.
7. Now that we have the correct error at $a^{(k-1)}_j$, we move it across neuron $j$ at layer $k-1$ by multiplying with the local gradient $f'(z^{(k-1)}_j)$.
8. Thus, the error that reaches $z^{(k-1)}_j$, called $\delta^{(k-1)}_j$ is $f'(z^{(k-1)}_j) \sum_i \delta^{(k)}_i W ^{(k-1)}_{ij}$.



## 6 Training with Backpropagation - Vectorized

So far, we discussed how to calculate gradients for a given parameter in the model. Here we will generalize the approach above so that we update weight matrices and bias vectors all at once. Note that these are simply extensions of the above model that will help build intuition for the way error propagation can be done at a matrix vector level.

For a given parameter $W^{(k)}_{ij}$, we identified that the error gradient is simply $\delta^{(k+1)}_i \cdot a^{(k)}_j$. As a reminder, $W^{(k)}$ is the matrix that maps $a^{(k)}$ to $z^{(k+1)}$. We can thus establish that the error gradient for the entire matrix $W^{(k)}$ is:
$$
\nabla_{W^{(k)}}=\left[\begin{array}{ccc}
{\delta_{1}^{(k+1)} a_{1}^{(k)}} & {\delta_{1}^{(k+1)} a_{2}^{(k)}} & {\cdots} \\
{\delta_{2}^{(k+1)} a_{1}^{(k)}} & {\delta_{2}^{(k+1)} a_{2}^{(k)}} & {\cdots} \\
{\vdots} & {\vdots} & {\ddots}
\end{array}\right]=\delta^{(k+1)} a^{(k) T}
$$
Thus, we can write an entire matrix gradient using the outer product of the error vector propagating into the matrix and the activations forwarded by the matrix.

Now, we will see how we can calculate the error vector $\delta^{(k)}$. We establish earlier that $\delta^{(k)}_j = f'(z^{(k)}_j) \sum_i \delta^{(k+1)}_i W ^{(k)}_{ij}$. This can easily generalize to matrices such that:
$$
\delta^{(k)}=f^{\prime}\left(z^{(k)}\right) \circ\left(W^{(k) T} \delta^{(k+1)}\right)
$$
In the above formulation, the $\circ$ operator corresponds to an element wise product between elements of vectors ($\circ:\ \mathbb{R}^{N} \times \mathbb{R}^{N} → \mathbb{R}^{N}$).

### 6.1 Computational efficiency

Having explored element-wise updates as well as vector-wise updates, we must realize that the vectorized implementations run substantially faster in scientific computing environments such as MATLAB or Python (using NumPy/SciPy packages). Thus, we should use vectorized implementation in practice. 

Furthermore, we should also reduce redundant calculations in backpropagation - for instance, notice that $\delta^{(k)}$  depends directly on $\delta^{(k+1)}$. Thus, we should ensure that when we update $W^{(k)}$ using $\delta^{(k+1)}$, we save $\delta^{(k+1)}$ to later derive $\delta^{(k)}$ – and we then repeat this for $(k − 1). . .(1)$. Such a recursive procedure is what makes backpropagation a computationally affordable procedure.

## 7 Neural network tricks 

### 7.1 Activation functions

#### 7.1.1 Sigmoid

$$
\sigma(z) = \frac{1}{1+exp(-z)}
$$

where $\sigma(z) \in (0,1).$

<img src="C:\Users\vince\AppData\Roaming\Typora\typora-user-images\image-20200118205449689.png" alt="image-20200118205449689" style="zoom:50%;" />

The gradient of $\sigma(z)$ is
$$
\sigma'(z) = \frac{-exp(-z)}{1+exp(-z)} = \sigma(z)(1-\sigma(z))
$$

#### 7.1.2 Tanh

The tanh function is an alternative to the sigmoid function that is often found to converge faster in practice. The primary difference between tanh and sigmoid is that tanh output ranges from −1 to 1 while the sigmoid ranges from 0 to 1.
$$
tanh(z) = \frac{exp(z)-exp(-z)}{exp(z)+exp(-z)} = 2\sigma(2z)-1
$$
where $tanh(z) \in (-1, 1)$.

<img src="C:\Users\vince\AppData\Roaming\Typora\typora-user-images\image-20200118205646708.png" alt="image-20200118205646708" style="zoom:50%;" />

The gradient of tanh(z) is
$$
tanh'(z) = 1 - (\frac{exp(z)-exp(-z)}{exp(z)+exp(-z)})^2 = 1-tanh^2(z)
$$

#### 7.1.3 Hard tanh

The hard tanh function is sometimes preferred over the tanh function since it is **computationally cheaper**. It does however saturate for magnitudes of z greater than 1. The activation of the hard tanh is:
$$
\operatorname{hardtanh}(z)=\left\{\begin{aligned}
-1 &: z<-1 \\
z &:-1 \leq z \leq 1 \\
1 &: z>1
\end{aligned}\right.
$$
<img src="C:\Users\vince\AppData\Roaming\Typora\typora-user-images\image-20200118205940597.png" alt="image-20200118205940597" style="zoom:50%;" />

The derivative can also be expressed in a piecewise functional form:
$$
\operatorname{hardtanh}'(z)=\left\{\begin{aligned}
1 &:-1 \leq z \leq -1 \\
0 &:\text{otherwise}
\end{aligned}\right.
$$

#### 7.1.4 Soft sign

The soft sign function is another nonlinearity which can be considered an alternative to tanh since it too does not saturate as easily as hard clipped functions:
$$
\text{softsign}(z) = \frac{z}{1+|z|}
$$
<img src="C:\Users\vince\AppData\Roaming\Typora\typora-user-images\image-20200118210246497.png" alt="image-20200118210246497" style="zoom:50%;" />

The derivative is expressed as:
$$
\text{softsign}'(z) = \frac{sgn(z)}{(1+z)^2}
$$
where sgn is the signum function which returns ±1 depending on the sign of $z$.

#### 7.1.5 ReLU

The ReLU (Rectified Linear Unit) function is a popular choice of activation since it does not saturate even for larger values of z and has found much success in computer vision applications: 
$$
\text{rect}(z) = max(z,0)
$$
<img src="C:\Users\vince\AppData\Roaming\Typora\typora-user-images\image-20200118210525213.png" alt="image-20200118210525213" style="zoom:50%;" />

The derivative is then the piecewise function:
$$
\operatorname{rect}'(z)=\left\{\begin{aligned}
1 &:z > 0 \\
0 &:\text{otherwise}
\end{aligned}\right.
$$

#### 7.1.6 Leaky ReLU

Traditional ReLU units by design do not propagate any error for non-positive z – the leaky ReLU modifies this such that a small error is allowed to propagate backwards even when z is negative:
$$
\text{leaky}(z) = max(z, k \cdot z)
$$
where $0<k<1$.

<img src="C:\Users\vince\AppData\Roaming\Typora\typora-user-images\image-20200118210741311.png" alt="image-20200118210741311" style="zoom:50%;" />

This way, the derivative is representable as:
$$
\operatorname{leaky}'(z)=\left\{\begin{aligned}
1 &:z > 0 \\
k &:\text{otherwise}
\end{aligned}\right.
$$

### 7.2 Data preprocessing

#### 7.2.1 Mean Subtraction

Given a set of input data X, it is customary to zero-center the data by subtracting the mean feature vector of X from X. 

An important point is that in practice, the mean is calculated only across the training set, and this mean is subtracted from the training, validation, and testing sets. This way we can prevent data leakage.

#### 7.2.2 Normalization

Another frequently used technique (though perhaps less so than mean subtraction) is to scale every input feature dimension to have similar ranges of magnitudes. This is useful since input features are often measured in different “units”, but we often want to initially consider all features as equally important. 

The way we accomplish this is by simply dividing the features by their respective standard deviation calculated across the training set.

#### 7.2.3 Whitening

Not as commonly used as mean-subtraction + normalization, whitening essentially converts the data to a have an identity covariance matrix – that is, features become uncorrelated and have a variance
of 1. 

This is done by first mean-subtracting the data, as usual, to get $X'$.  We can then take the Singular Value Decomposition $(SVD)$ of $X'$ to get matrices $U, S, V$. We then compute $UX'$ to project $X'$ into the
basis defined by the columns of $U$. We finally divide each dimension of the result by the corresponding singular value in $S$ to scale our data appropriately (if a singular value is zero, we can just divide by a
small number instead).

### 7.3 Parameter Initialization (Xavier)

A good starting strategy is to initialize the weights to small random numbers normally distributed around 0 – and in practice, this often words acceptably well.

However, in `Understanding the difficulty of training deep feedforward neural networks (2010)`, Xavier et al study the effect of different weight and bias initialization schemes on training dynamics. The empirical findings suggest that for **sigmoid and tanh** activation units, faster convergence and lower error rates are achieved when the weights of a matrix $W \in \mathbb{R}^{n^{(l+1)} \times n^{(l)}}$ are initialized randomly with a uniform distribution as follows:

<img src="C:\Users\vince\AppData\Roaming\Typora\typora-user-images\image-20200118212938893.png" alt="image-20200118212938893" style="zoom:50%;" />

where

* $n^{(l)}$ is the number of input units to $W$ (fan-in)
* $n^{(l+1)}$ is the number of output units from $W$ (fan-out)

In this parameter initialization scheme, bias units are initialized to 0.

This approach attempts to maintain activation variances as well as backpropagated gradient variances across layers. Without such initialization, the gradient variances (which are a proxy for information) generally decrease with backpropagation across layers.

### 7.4 Learning Strategies

#### 7.4.1 Hand-set learning rates

The rate/magnitude of model parameter updates during training can be controlled using the learning rate. In the following naïve Gradient Descent formulation, $\alpha$ is the learning rate:
$$
\theta^{new} = \theta^{old} - \alpha \nabla_{\theta}J_t(\theta)
$$
When learning rate  $\alpha$ is too large, we might experience that the loss function actually diverges because the
parameters update causes the model to overshoot the convex minima. In non-convex models (most of those we work with), the outcome of a large learning rate is unpredictable, but the chances of diverging loss functions are very high.

<img src="C:\Users\vince\AppData\Roaming\Typora\typora-user-images\image-20200118213759945.png" alt="image-20200118213759945" style="zoom:33%;" />

When learning rate  $\alpha$ is too small, we might not converge in a reasonable amount of time, or might get caught in local minima.

Since training is the most expensive phase in a deep learning system, some research has attempted to improve this naïve approach to setting learning learning rates. For instance, Ronan Collobert scales the learning rate of a weight $W_{ij}$ (where $W \in \mathbb{R}^{n^{(l+1)} \times n^{(l)}}$) by the inverse square root of the fan-in of the neuron ($n^{(l)}$).

There are several other techniques that have proven to be effective as well – one such method is **annealing**, where, after several iterations, the learning rate is reduced in some way – this method ensures that we start off with a high learning rate and approach a minimum quickly; as we get closer to the minimum, we start lowering our learning rate so that we can find the optimum under a more fine-grained scope.

* A common way to perform annealing is to reduce the learning rate $\alpha$ by a factor $x$ after every $n$ iterations of learning. 
* Exponential decay is also common, where, the learning rate $\alpha$ at iteration $t$ is given by $\alpha(t) = \alpha_0 e^{-kt}$, where $\alpha_0$ is the initial learning rate, and $k$ is a hyperparameter. 
* Another approach is to allow the learning rate to decrease over time such that $\alpha(t)=\frac{\alpha_{0} \tau}{\max (t, \tau)}$ where $\alpha_0$ is a tunable parameter and represents the starting learning rate. $\tau$ is also a tunable parameter and represents the time at which the learning rate should start reducing. In practice, this method has been found to work quite well.

  #### 7.4.2 Momentum updates

Momentum methods enables adaptive gradient descent without the need of hand-set learning rates. It is  a variant of gradient descent inspired by the study of dynamics and motion in physics, attempting to use the "velocity" of updates as a more effective update scheme.  Pseudocode for momentum updates is shown below:

<img src="C:\Users\vince\AppData\Roaming\Typora\typora-user-images\image-20200118215054142.png" alt="image-20200118215054142" style="zoom:50%;" />

### 7.5 Adaptive Optimization Methods

#### 7.5.1 AdaGrad

AdaGrad is an implementation of standard stochastic gradient descent (SGD) with one key difference: the learning rate can vary for each parameter.

The learning rate for each parameter depends on the history of gradient updates of that parameter in a way such that parameters with a scarce history of updates are updated faster using a larger learning rate. In other words, parameters that have not been updated much in the past are likelier to have higher learning rates now. Formally:
$$
\theta_{t, i}=\theta_{t-1, i}-\frac{\alpha}{\sqrt{\sum_{\tau=1}^{t} g_{\tau, i}^{2}}} g_{t, i} \text { where } g_{t, i}=\frac{\partial}{\partial \theta_{i}^{t}} J_{t}(\theta)
$$
In this technique, we see that if the RMS of the history of gradients is extremely low, the learning rate is very high. A simple implementation of this technique is:

<img src="C:\Users\vince\AppData\Roaming\Typora\typora-user-images\image-20200118215359134.png" alt="image-20200118215359134" style="zoom:50%;" />

#### 7.5.2 RMSProp and Adam

**`RMSProp`:**

<img src="C:\Users\vince\AppData\Roaming\Typora\typora-user-images\image-20200118215534829.png" alt="image-20200118215534829" style="zoom:50%;" />

**`Adam`:**

<img src="C:\Users\vince\AppData\Roaming\Typora\typora-user-images\image-20200118215618925.png" alt="image-20200118215618925" style="zoom:50%;" />

RMSProp is a variant of AdaGrad that utilizes a moving average of squared gradients – in particular, unlike AdaGrad, its updates do not become monotonically smaller. The Adam update rule is in turn a variant of RMSProp, but with the addition of momentum like updates

.