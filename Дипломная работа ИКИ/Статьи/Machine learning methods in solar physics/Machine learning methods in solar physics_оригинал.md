---
title: "Machine learning methods in solar physics"
source_article: "E.A. Illarionov, 2025 - Machine learning methods in solar physics (Uspekhi Fizicheskikh Nauk)"
tags:
  - machine-learning
  - solar-physics
  - paper/original
  - review
authors: "E.A. Illarionov"
year: 2025
journal: "Uspekhi Fizicheskikh Nauk"
doi: "10.3367/UFNe.2025.02.039872"
---

# Machine learning methods in solar physics

**Author:** E.A. Illarionov  
**Journal:** Uspekhi Fizicheskikh Nauk (2025)  
**DOI:** [10.3367/UFNe.2025.02.039872](https://doi.org/10.3367/UFNe.2025.02.039872)

---

## Contents

### [[#1. Introduction|1. Introduction]]
### [[#2. Main ideas of machine learning|2. Main ideas of machine learning]]
- [[#2.1 Data model|2.1 Data model]]
- [[#2.2 Training problem and loss function|2.2 Training problem and loss function]]
  - [[#2.2.1 Classification and regression|2.2.1 Classification and regression]]
  - [[#2.2.2 Physically informed neural networks|2.2.2 Physically informed neural networks]]
  - [[#2.2.3 Neural-differential equations|2.2.3 Neural-differential equations]]
  - [[#2.2.4 Autoencoders|2.2.4 Autoencoders]]
  - [[#2.2.5 Generative models|2.2.5 Generative models]]
- [[#2.3 Minimizing loss function|2.3 Minimizing loss function]]

---

### 1. Introduction

Machine learning methods are becoming an increasingly popular and effective tool in data processing. Perhaps one of the most notable results is provided by natural language processing models, used, for example, in machine translation. Interestingly, natural language models are currently one of the main driving forces in machine learning. Many modern models for processing images, videos, and other data formats were first proposed for text processing. A similar situation can also be seen in the past. As is known, the first striking application of Markov chains was the statistical analysis of a literary text, but then this model turned out to be in demand in a much wider range of applications. A modern example is the architecture of transformer-type neural networks (Transformer). The model was proposed in the context of machine translation and formed the basis of language models that have become the most popular and advanced: BERT (Bidirectional Encoder Representations from Transformers), GPT (Generative Pretrained Transformer), and a number of others. Recently, adaptations have been proposed for working with images (VisualTransformer) and video (VideoGPT), as well as other tasks. The above language models are closely related to another important concept that has been shaped mainly in the context of natural language processing models: the concept of foundation models of machine learning. The essence is that, instead of training individual models for each specific task, a single and, in a sense, universal model is trained for many tasks at once. Today, this approach is actively being developed in a variety of applications.

Without a doubt, the results that machine learning methods show in mundane tasks inspire the quest for applications in more specialized fields. The purpose of this paper is to review applications for problems related to solar activity studies. The emphasis is on demonstrating the diversity of problems for which machine learning models can be useful and on providing examples of solutions to those problems. A detailed discussion of differences among different solutions to the same problem and a comparison of the quality of approaches is beyond the scope of this paper, because it would require a more in-depth analysis of subtleties of the nature of data, the structure of the models, and the quality assessment criteria.

The idea to use machine learning methods in solar physics is not unreasonable. One of the most important premises suggesting that the resulting model would be viable is the availability of a large array of data containing examples of the desired solution to the problem. In this sense, solar physics is a rich source of both diverse problems and a wealth of observational data. The appearance of the first systematic data on solar activity is usually associated with a series of records and sketches of the solar disk and positions of sunspots made by Galileo Galilei in 1612, although several earlier series of sketches by Thomas Harriot in late 1610 to early 1613 have recently been reconstructed. In any case, the historical corpus of data is currently more than 400 years old, which is a unique duration for a scientific measurement streak in general. Of course, historical data are not without their own uncertainties and irregularities, which complicates their analysis, but they help us to understand the long-term features of the dynamics of solar activity. Modern ground-based and space telescopes together provide virtually continuous monitoring of the solar disk with high spatial resolution (up to several tens of kilometers for the DKIST telescope) and in different spectral ranges (white light, ultraviolet (UV) and X-ray ranges, the radio range, and magnetograms). All this allows the formation and development of active regions on the Sun to be comprehensively studied.

The large time span and high level of space-time details achieved to date invite various statistical studies. In practice, however, the analysis of observational data is significantly complicated by a number of factors. One of them is the heterogeneity of the data. Over the 400 years, observers have changed, technology has improved, and ideas about what and how to observe have evolved. As a result, the question of reconciling earlier and later observations arises, entailing various discussions on this matter. For example, the Wolf number series, compiled from individual series of observations of sunspot groups, has been revised more than once, resulting in significant modifications.

Another major problem is the availability and completeness of data. A hefty stratum of historical data is still in paper archives, and much of what has been scanned into electronic form awaits digitization of handwritten text and sketches. Many details of how observations were conducted in the distant past are unknown to us, and hence the digitization stage is often also a stage of reconstructing the observation methods.

Modern observational data, on the one hand, are free of many problems that arise when working with historical data: they are immediately presented in digital form, are much more homogeneous (the known problem of degradation of measuring instruments notwithstanding), and are taken regularly. Still, we cannot say that they have been studied systematically. The main stumbling block is the large physical volume of data, replenished daily with streams of new data.

As an example, we can recall that just one SDO (Solar Dynamics Observatory) satellite, which is operational since 2010, observes the solar disk every 12 seconds.

We can conclude that extracting useful information from large amounts of data is a challenge for modern science and computing technologies. Machine learning methods, as practice shows, constitute one effective way to achieve this goal.

In what follows, we recall the main ideas of machine learning methods and then proceed to examples of their use in solar physics.

### 2. Main ideas of machine learning

#### 2.1 Data model

A typical example of the emergence of a machine learning task is the problem of describing some observed dependence in functional form (building a data model). If it is also desirable to work out a general approach to solving a whole class of similar problems, then the nature and details of the observed dependence at a particular moment must be unimportant. Instead, a parametric family of functions (often also called a model) must be chosen so as to be broad enough to cover any possible dependence. An example of such a family is given by polynomials, which, as is known, allow approximating continuous functions on an interval with any accuracy.

While mathematically perfect, many constructions become computationally complex when working in high-dimensional spaces (for example, the resulting products of a large number of polynomials can quickly overcome the limits set by machine precision). We note that high-dimensional spaces are a typical, rather than a special, case in modern data processing tasks. An example is given by the problem of classifying images, which are essentially numerical matrices and are interpreted as objects in a space with a dimension equal to the number of matrix entries. For high-resolution images, this number can be several million. Thus, from a computational standpoint, the question arises of finding a more convenient and universal method for approximating functions.

An important concept of functions in arbitrary-dimension spaces was worked out by Kolmogorov. He showed that any continuous function in the interior of a unit n-dimensional cube can be represented by a finite composition of functions of one argument,

$$f(x_1, x_2, ..., x_n) = \sum_{k=1}^{2n+1} \phi_k \left( \sum_{i=1}^n \psi_{i,k}(x_i) \right) \quad (1)$$

where the functions $\psi_{i,k}$ are independent of $f$ and depend only on the dimension $n$. We can say that they form a universal basis. The functions $\phi_k$, of course, do depend on $f$.

![[Figure 1.png|Figure 1]]
**Figure 1:** Representation of a function by a superposition of functions of one variable, in accordance with Kolmogorov's theorem.

In essence, functions of $n$ arguments can be represented as a relatively simple computational graph, as shown in Fig. 1, where the number of nodes depends on the space dimension only linearly. Thus, relying on this theorem, we can hope to find an approximation of a function using a finite (and quite reasonable) number of simple functions (simple in the sense that they depend on one argument). A practical implementation of this theorem was proposed quite recently, but it is still not without its shortcomings, and it is therefore too early to speak about its widespread use.

From a practical standpoint, a somewhat later result in Cybenko's theorem was of great importance (it is worth noting that, in the same year, the same result was independently obtained by others). Cybenko's theorem states that functions that are continuous inside a unit n-dimensional cube can be approximated with any predefined accuracy using a composition of sigmoid functions,

$$f(x_1, x_2, ..., x_n) \approx \sum_k \alpha_k \sigma \left( \sum_{i=1}^n \gamma_{i,k} x_i + \beta_k \right) \quad (2)$$

where $\gamma_{i,k}$, $\alpha_k$, and $\beta_k$ are constants depending on $f$, and $\sigma$ is some predefined continuous nondecreasing bounded function on the entire line. Such a function is often taken in the form

$$\sigma(x) = \frac{1}{1 + \exp(-x)} \quad (3)$$


known as the logistic function or sigmoid. The corresponding computational graph is shown in Fig. 2. Superficially, it is very similar to the one in Fig. 1, but an essential difference is that the number of nodes in the second case is not specified in advance and can be large enough for an accurate approximation, but unknown functions are replaced with unknown scalar parameters, which are much easier to work with.

![[Figure 2.png|Figure 2]]
**Figure 2:** Approximation of a function by a superposition of sigmoid functions, in accordance with Cybenko's theorem.

Today, this computational graph is usually called the fully connected neural network model with one hidden layer and the activation function $\sigma$. Accordingly, Cybenko's theorem is referred to as the theorem of universal approximation by neural networks. Graph nodes are called neurons, and the specific structure of the graph (the number of neurons, layers, and connections) is called the neural network architecture.

Somewhat anticipating the subsequent discussion, we note that modern architectures have a large number of layers and a sophisticated system of connections. We refer the reader to specialized literature for further discussion of theoretical issues.

In this review, we limit ourselves to considering applications of machine learning models based on neural networks, although they by no means exhaust the toolbox of machine learning methods. A systematic review can be found, for example, in specialized books. We only note that the general principles remain the same.

#### 2.2 Training problem and loss function

A data model based on a neural network defines a whole family of functions. The choice of a particular function is determined by the choice of specific model parameter values. To select such values, we must formulate the criteria that the desired function must satisfy. Mathematically, this is implemented by defining a functional (often called the loss function or the risk function) whose minimum is achieved on the desired function, and the optimization problem is then solved. The functional in question can include observational data, physical relations (for example, differential equations), various types of constraints on the function values, and other terms. We consider some of the most important examples.

##### 2.2.1 Classification and regression

A typical problem is as follows: given the data in the form of pairs of observations $(x_i, y_i)$, where the subscript $i$ is the observation number, find a function $f(x)$ that approximates the observed values of the second variable $y$. Such a formulation is possible for the binary classification problem (where $y$ takes only two values, for example, 0 and 1), a multiclass classification problem (with $y$ having a discrete nature and more than two possible values), or regression (where $y$ is continuous). We let $f(x, \theta)$ denote the data model, with $\theta$ being the set of model parameters. We can then propose the loss function as the average over all observations of:
- the squared difference $l(\theta) = (y_i - f(x_i, \theta))^2$ for the regression problem;
- the binary cross-entropy $l(\theta) = -y_i \log f(x_i, \theta) - (1 - y_i) \log(1 - f(x_i, \theta))$ for the binary classification problem (where the values of $f(x_i, \theta)$ are assumed to be nonnegative and not exceeding 1);
- the cross-entropy $l(\theta) = -\sum_{k=1}^K y_{i,k} \log f_k(x_i, \theta)$ for the K-class classification problem, where $y_{i,k} = 1$ if $y_i = k$ and 0 otherwise (and the $f_k(x_i, \theta)$ are also assumed to be nonnegative numbers such that their sum over $k$ is 1).

The above functions are by no means the only possible ones. They can be called canonical in the sense that they arise from maximum likelihood estimates. For example, the mean-square loss function arises under the assumption that the observations $y_i$ are independent and distributed in accordance with the normal (Gaussian) law, in which the mean (mathematical expectation) is a function of $x_i$ and the parameters $\theta$. It is also useful to recall that all three cases (regression, binary, and multiclass classification), despite their superficial differences, can be combined into one model and studied in the framework of a single theory of a generalized linear model using the tools of probability theory and mathematical statistics ('linearity' in no way limits the consideration of nonlinear dependences). For more information on the probabilistic context of machine learning models, we refer the reader to specialized books.

It is worth noting that, in a number of problems formally related to classification or regression, the choice of the functions considered above may not fully correspond to the nature of the problem. As an example, we consider the situation where $y_i$ are images (say, any photograph). If we mentally shift the entire image one pixel to the right, then both images would be virtually indistinguishable from the standpoint of perception, but the mean square distance between them can be large (for example, larger than between two images that are visually dissimilar). In such cases, more specific metrics have to be used. For example, in the context of images, they can be metrics calculated not directly between pixel values but between some more complex characteristics of two images (for example, perceptual loss).

##### 2.2.2 Physically informed neural networks

Classification and regression problems are basic in the sense that other problems, whose original formulation can differ greatly from the problem of predicting $y$ from $x$ values, can be reduced to them in one way or another. As an example, we consider the problem of numerically solving differential equations. For definiteness, we take an ordinary differential equation $F(x, y, y', ..., y^{(n)}) = 0$ for a function $y(x)$ with the initial condition $y(x_0) = y_0$. This problem can be solved using the finite-difference method and its variations. However, it can also be approached from the machine learning standpoint. Because neural networks are a universal approximator, we can define the desired function $y(x)$ in the form of a neural network and choose its parameters such that it satisfies the chosen differential equation as accurately as possible. It is useful to note here that a function represented by a neural network is by construction differentiable with respect to both its parameters and the input variables. Moreover, all derivatives are calculated using automatic differentiation methods. Thus, if we define $y(x)$ by a neural network $y(x, \theta)$, then all derivatives $y^{(n)}$ are also defined. Next, to compile the loss function, one usually takes a set of random points $x_i$ from the domain of $y(x)$ and calculates the mean square of the discrepancy between the left- and right-hand sides of the equation at these points: $l(\theta) = F^2(x_i, y(x_i, \theta), y'(x_i, \theta), ..., y^{(n)}(x_i, \theta))$ (recall that there is zero on the right-hand side of the equation, hence the square of $F$). To take the initial condition into account, a term of the form $(y(x_0, \theta) - y_0)^2$ is added to the error function with a coefficient that controls the contribution of this term.

It is easy to see that solving a differential equation is thus reduced to a regression problem. However, a more interesting conclusion is obtained if we try to combine the problems of regression on observed pairs $(x_i, y_i)$ and of solving a differential equation. In other words, we assume that the observed dependence of $y$ on $x$ satisfies a given differential equation with some accuracy. Accordingly, the dependence $y(x)$ can be found by minimizing the composite loss function:

$$l(\theta) = \frac{1}{n} \sum_{i=1}^n (y(x_i) - y_i)^2 + \alpha \frac{1}{m} \sum_{j=1}^m F^2(x_j, y(x_j), y'(x_j), ..., y^{(n)}(x_j)) \quad (4)$$

Here, the $\theta$ argument is omitted for simplicity, $n$ is the sample size of observational data, $m$ is the number of points at which the accuracy of the solution of the differential equation is estimated, and the $\alpha$ coefficient is responsible for the balance of the two terms. Such a construction is known as a physics-informed neural network (PINN). We note that the a priori physical model of the data in the form of a differential equation acts as an extra constraint (often called a regularizer) on the spectrum of possible dependences $y(x)$ underlying the observed data. In particular, this approach can help overcome the model overfitting problem (when the training data error is significantly less than the test data error) and increase the stability of prediction for new data.

As an example, we consider a problem where the training sample is obtained from a sinusoid superimposed by random Gaussian noise (Fig. 3). We take two identical fully connected neural network models and train the first one with the loss function given by the sum of the mean-square error in the training sample and the error from solving the differential equation $y'(x) = \cos x$; the second model is trained based only on the mean-square error in the training sample. Figure 3 shows that the second model fits the observational data better, while the first model balances between the physical model and the observational data. The difference is pronounced in the region where no observations are available.

![[Figure 3.png|Figure 3]]
**Figure 3:** Regression on training set (black dots) using PINN (red line) and regular neural network (NN, blue line) models. Model architecture is the same in both cases. PINN is trained using equation $y'(x) = \cos x$.

Other methods of model regularization also exist. For example, additional terms characterizing the amplitude of the model parameters can be added to the loss function: the sum of the squares of the model parameters (the so-called L2, or ridge, or Tikhonov regularization) or the sum of their moduli (the L1, or LASSO regularization). More complex constructions are described in specialized literature.

##### 2.2.3 Neural-differential equations

From the problem of solving differential equations, we proceed to the inverse problem: given observations of the process at certain time instants, reconstruct the differential equation that models this process. The mathematical formulation of the problem again reduces to finding the loss function. We assume that the observed process satisfies the equation

$$\frac{df(t)}{dt} = g(f(t), t) \quad (5)$$

with an unknown function $g(\cdot, \cdot)$, which we define by a neural network $g(\cdot, \cdot, \theta)$. We then solve the equation using standard numerical schemes and vary the neural network parameters such that the solution result agrees with the observed data. To construct the loss function, we let $\text{ODESolve}(g, t_0, f_0, t)$ denote the numerical scheme whose input is given by the function on the right-hand side of (5), the initial data $(t_0, f_0)$, and the time instant $t$ for which the solution is sought. Let $\{(t_i, f_i)\}_{i=1}^n$ be a set of observed values of the process. Then, the mean-square error can again be used as the loss function:

$$l(\theta) = \frac{1}{n} \sum_{i=1}^n (\text{ODESolve}(g, t_0, f_0, t_i) - f_i)^2 \quad (6)$$

This approach is known as neural-differential equations. We note that the key is the procedure for calculating the gradient of the loss function $l(\theta)$ with respect to the parameters $\theta$ without having to directly differentiate the ODESolve function. For this, an auxiliary (adjoint) differential equation is constructed, whose solution yields the value of $dl/d\theta$.

##### 2.2.4 Autoencoders

Many types of observational data, for example, images and spectral line profiles, are objects in high-dimensional spaces (images are specified by numerical matrices, and lines, by a set of values at each wavelength). To analyze such objects, their description in a more compact form is desirable, using a smaller number of parameters but capturing the most significant features. Mathematically, such a problem is akin to that of reducing the dimensionality of data. To make this problem meaningful, one can require that the map of an object to a lower-dimension space be almost invertible, i.e., that the reconstruction of the original image be possible. Then, one can assert that the lower-dimensional representation preserves information about the original object. A common standard method for solving this problem is the principal component analysis (PCA) method proposed by Pearson in 1901. However, this method is linear, and we can therefore assume its efficiency to be insufficient with complex data. Finding nonlinear methods then becomes of interest, the autoencoder model being a possible solution. The model consists of two functions defined by neural networks. One of them, denoted by $\text{Enc}(x, \theta)$, is called the encoder and maps $x$ to a lower-dimensional vector $z = \text{Enc}(x)$. The vector $z$ is called the hidden or latent representation of $x$. The second function, the decoder $\text{Dec}(z, \tilde{\theta})$, maps vectors from the latent space to the original one: $\tilde{x} = \text{Dec}(z)$. In the simplest implementation, model training requires that the composition of the decoder and encoder $\text{Dec}(\text{Enc}(x))$ be as close to the identity transformation as possible. Accordingly, the loss function arises in the form

$$l(\theta, \tilde{\theta}) = \frac{1}{n} \sum_{i=1}^n (\text{Dec}(\text{Enc}(x_i)) - x_i)^2 \quad (7)$$

where $\{x_i\}_{i=1}^n$ is the training dataset. There are various modifications of the autoencoder model, which, for example, impose additional constraints on the distribution of vectors in the latent space. As an example, let us recall the variational autoencoder. It involves an additional term in the loss function, which is responsible for the closeness of the distribution of latent vectors to the standard multivariate Gaussian distribution. An extended review can be found in specialized literature.

In the context of variational autoencoder-type models, the models called normalizing flows are also worth mentioning. Their main purpose is to select a rigorously invertible transformation that converts the distribution generated by the sample into a standard Gaussian one. Accordingly, the dimensions of the hidden and original spaces are set equal, and the neural network is constructed such that the inverse transformation can easily be written. For a more detailed review, we refer the reader to specialized literature. One of the applications of normalizing flows is the generation of synthetic data. For this, a sample from a Gaussian distribution is modeled, which is then mapped into the original space by the inverse of the normalizing flow, thereby creating a new synthetic sample. The same technique allows modeling synthetic data using the decoder of a pretrained variational autoencoder.

##### 2.2.5 Generative models

Let us discuss generative models in more detail. Their purpose is to create new data samples that are indistinguishable in properties from a real data sample, but are not literally the same. Mathematically, this means that the sample obtained from the model has the same distribution as the actual data sample. The resulting synthetic samples can be useful, for example, as initial data in simulations of dynamic systems.

A generative model (generative-adversarial network, GAN) usually consists of two parts implemented in terms of neural networks. One part is called the generator $G(z, \theta)$: it maps the space in which the standard Gaussian sample $z_i$ is modeled to the space in which the real sample $x_i$ is defined. The second model is called the discriminator $D(x, \tilde{\theta})$: it is a binary classifier that must distinguish a synthetic example from the real one. The generator and the discriminator have competing tasks. The generator must learn to produce examples that are classified as real from the discriminator's standpoint, while the discriminator, on the contrary, must learn to discriminate between real and synthetic data as well as possible. This game gives rise to a loss function for the generator

$$l(\theta) = -\frac{1}{n} \sum_{i=1}^n \log[1 - D(G(z_i))] \quad (8)$$

which becomes smaller (noting the minus sign before the expression) the more often the discriminator erroneously classifies synthetic data as real (it is assumed that real data correspond to class zero, and synthetic data, to class one). The loss function for the discriminator is

$$l(\tilde{\theta}) = \frac{1}{n} \sum_{i=1}^n [\log D(x_i) + \log(1 - D(G(z_i)))] \quad (9)$$

The first term becomes smaller the more accurately the discriminator classifies real data as real, and the second term becomes smaller the more accurately it classifies synthetic examples as synthetic. This loss function was proposed in foundational GAN literature; modern implementations use more complex constructions.

The above examples of problems and corresponding loss functions by no means exhaust the range of possibilities, but they often serve as a starting point for discussions of more complex constructions (as an example, we note the problem of approximating operators in infinite-dimensional spaces by neural networks). Another motivation for discussing these examples was to demonstrate the diversity of problem statements for machine learning methods. We now proceed to the next stage: numerical solution of the loss function minimization problem.

#### 2.3 Minimizing loss function

On the one hand, the problem of finding a minimum point of a function is classical in numerical methods, but a number of features that arise in the context of machine learning give rise to new challenges for methods as well as computing devices.

The first feature is associated with the high dimension of the space in which the optimization problem is defined. Modern neural networks have about 10⁶–10⁸ parameters and are trained on samples of comparable size. Accordingly, the question arises about efficient parallelization of computations and the use of processors that support such technologies. In particular, graphic processing units (GPUs) and tensor processors (TPUs), designed especially for machine learning tasks, are used.

Another important feature of the problem is that the loss function is almost always nonconvex and has many local minima. Accordingly, the classical gradient descent method is ineffective (it directs to the nearest local minimum, which can be much worse than the global one) and, moreover, the result is strongly dependent on the choice of the starting point (in the context of neural networks, we are talking about the initialization of the model parameters, which are usually set randomly). Machine learning problems have stimulated a new round of development of numerical optimization methods, as a result of which a number of new algorithms have appeared that use different ideas on how to ensure convergence to a more optimal local minimum. A review of these methods is beyond the scope of this paper, and we only mention some algorithms that have appeared over the past decade and have entered the standard set of libraries for working with neural networks:
- adaptive learning rate method (Adadelta);
- adaptive moment estimation (Adam);
- adaptive subgradient method (Adagrad);
- evolved sign momentum (Lion);
- follow the regularized leader (FTRL).

More classical algorithms, also included in standard libraries, are:
- stochastic gradient descent (SGD);
- averaged stochastic gradient descent (ASGD);
- limited memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) algorithm;
- resilient backpropagation (RPROP).

As an example, Fig. 4 shows the trajectories traced in searching for a minimum by some algorithms for the same function and starting point.

![[Figure 4.png|Figure 4]]
**Figure 4:** Trajectories converging to a minimum obtained by different optimization algorithms: black line is Adam, orange line is SGD, and purple line is RPROP. Red dot shows location of global minimum of the function.

The above optimization methods use gradient descent in one form or another. This raises the question of how to calculate the derivatives (gradient) of the loss function with respect to the model parameters. Standard approaches—numerical or symbolic (computer algebra) differentiation—are not entirely suitable for such purposes for a number of reasons. Symbolic differentiation is slow due to implementation features and is ineffective for deep networks with complex architecture. Numerical differentiation, due to its discrete nature, introduces errors that lead to instability and is ineffective for a large number of model parameters. These problems were solved with a method called automatic differentiation. We discuss one of the implementations of automatic differentiation, which is called the backpropagation algorithm and is currently the main method for training neural networks.

The computational graph of a neural network is built from nodes that represent simple arithmetic operations: addition, multiplication, raising to a power, exponential, logarithm, etc. In each node, besides the operation itself, a function is written that implements the rule for calculating the derivative of this operation. With the example of the simplest computational graph shown in Fig. 5 and implementing a neural network with one neuron, we explain the procedure for calculating the gradient based on model parameters $w$ and $b$.

The procedure consists of two stages. In the first stage, called the forward pass, the values are calculated at all nodes of the graph, from the input to the output, and are stored in memory. The second stage, called the backward pass, consists of applying the chain rule of differentiation of a composite function to sequentially find derivatives when going from the output node to the nodes containing the model parameters:

$$\frac{\partial l}{\partial b} = \frac{\partial l}{\partial z} \frac{\partial z}{\partial b}, \quad (10)$$

$$\frac{\partial l}{\partial w} = \frac{\partial l}{\partial z} \frac{\partial z}{\partial w}. \quad (11)$$

The rules for calculating all derivatives on the right-hand side of (10) and (11) are written in the graph nodes: $\partial l/\partial z = 2(z - y)$, $\partial z/\partial b = 1$, and $\partial z/\partial w = x$. It then remains to substitute the values of $x$, $y$, and $z$ saved after the forward pass. We note that the use of dynamical programming principles allows avoiding the duplicating of calculations in (10) and (11) and provides an efficient procedure for more complex neural network architectures.

![[Figure 5.png|Figure 5]]
**Figure 5:** Computational graph illustrating error backpropagation algorithm. Green: trained model parameters, blue: arithmetic operations (multiplication, addition, calculating the norm), orange: other numerical variables.

Next, the iterative training procedure is launched. At each iteration, a subsample (called a batch in machine learning) is taken from the training data set, the loss function gradient is calculated on this subsample based on the model parameters, and the model parameters are updated in accordance with the optimization algorithm. There are different criteria for stopping the iterations: for example, when the loss function value stops decreasing significantly on the training sample or the error on the validation sample starts increasing. The last-iteration model parameters are fixed, and a trained model is said to have been obtained.

In the next section, we consider examples of models and their training results in solar physics problems.

### 3. Data sets for machine learning

The basis of machine learning is the training data set. The quality of a model depends on the properties of the set:
- size (the number of objects);
- variability (the diversity of examples);
- unbiasedness (the statistical properties of the training set must match the properties of the test set);
- annotation accuracy (well-defined classes, object boundaries, etc.).

Yet another important factor is the homogeneity of the data format. For example, if the data are represented by images of different sizes (in pixels), then time-efficient training of the model would most likely require a reduction of all images to the same size. For example, efficient training of neural networks is based on matrix operations, and hence the set of images should be represented by a single multidimensional array.

In practice, collecting and preparing a training dataset is one of the most labor-intensive and time-consuming stages in machine learning, and at the same time one of the most resource-intensive stages, requiring large data warehouses and a high speed of access to data (reading and writing). This can only be fully implemented in large computing centers. As a result, large volumes of data prepared for machine learning are of exceptional value today. In Table 1, we collect examples of solar activity data sources and datasets prepared for machine learning tasks. We note that such databases are also relevant for more general problems of studying solar activity.

| Title | Brief description | Address | Literature |
|---|---|---|---|
| HEK | Consolidated database from various sources on solar activity events and objects | https://lmsal.com/hek/ | |
| Helioviewer | Visualization of solar observation catalogues from various instruments and solar activity maps | https://www.helioviewer.org/ | |
| Interactive Multi-Instrument Database of Solar Flares | Consolidated catalogue of solar flares based on GOES, RHESSI, SDO/AIA, and a number of other instruments for 2002–2022 | https://solarflare.njit.edu/ | |
| Catalogue of SDO/AIA 193 Angstrom synoptic maps and coronal holes | Catalogue of synoptic maps and coronal hole maps based on SDO/AIA observations on the 193-Å line from 2010 to the present | https://sun.njit.edu/coronal_holes | |
| Large-Scale Dataset of Three-Dimensional Solar Magnetic Fields Extrapolated by Nonlinear Force-Free Method | Set of 73,000 examples of coronal magnetic field reconstructions over active regions from the SHARP catalogue for 2010–2019 | https://nlfff.dataset.deepsolar.space/en | |
| Large-scale Solar Dynamics Observatory image dataset for computer vision applications | Set of 260,000 SDO/AIA images corresponding to 270,000 events from the Heliophysics Event Knowledge (HEK) base | https://dataverse.harvard.edu/dataverse/lsdo | |
| Machine Learning Data Set for NASA's Solar Dynamics Observatory | SDO/AIA and SDO/HMI images from 2010–2018, calibrated and downsampled to a 512 × 512 resolution | https://purl.stanford.edu/... | |
| ObserveTheSun | Catalogue of solar activity maps compiled by the Kislovodsk Mountain Astronomical Station | https://observethesun.ru | |
| SDOBenchmark | Set of 8336 training and 886 test data on solar flares based on SDO/AIA and SDO/HMI images 12 h, 5 h, 1.5 h, and 10 min before the flare | https://i4ds.github.io/SDOBenchmark/ | |
| SEP 3 | Consolidated catalogue of SPEs, CMEs, and solar flares and their parameters based on GOES, SOHO/LASCO, SDO/HMI, and a number of other instruments | https://sun.njit.edu/SEP3 | |
| SERPENTINE | Catalogue of solar cosmic ray data based on Solar Orbiter, Parker Solar Probe, and BepiColombo observations | https://serpentine-h2020.eu/ | |
| SHARP | Vector magnetograms of active regions and their parameters based on SDO/HMI observations in 2010–2020 | http://jsoc.stanford.edu/doc/data/hmi/sharp/sharp.htm | |
| S®ARP | Magnetograms of active regions based on SOHO/MDI observations in 1996–2010 | http://jsoc.stanford.edu/ | |
| SOHO/EIT Flare Catalog | Catalogue of solar flares and their parameters based on SOHO/EIT data in 1997–2010 | https://doi.org/10.7910/DVN/C9H34R | |
| Solar active region magnetogram image dataset for space weather studies | Catalogue of magnetograms of active regions, reduced to a single size of 600 × 600 pixels (and 224 × 224 in a reduced version) for 2010–2018 | https://doi.org/10.5061/dryad... | |
| SolarMonitor | Visualization of solar observation catalogues from different instruments and solar activity maps | https://www.solarmonitor.org/ | |
| SunInTime | Visualization of solar disk images from the SDO satellite and the catalogue of active HEK events | https://suntoday.lmsal.com/ | |
| Sunspot groups | Catalogue of sunspot group images and their parameters based on data from the Kislovodsk Mountain Astronomical Station for 2010–2022 | https://github.com/observethesun/sunspot_groups | |
| SWAN-SF | Catalogue of solar flares and associated parameters obtained simultaneously from different instruments in 2010–2018 | https://doi.org/10.7910/DVN/EBCFKM | |

### 4. Software for working with data

The use of machine learning methods involves the use of tools for both creating and launching models, for data loading and preprocessing, and for implementing the training procedure.

Tools for working with machine learning models are universal and independent of the nature of the data. For example, PyTorch, Keras, and Tensorflow libraries are popular for working with neural networks. Data processing tools, on the contrary, depend significantly on the data storage format and their nature. In most solar activity studies, these steps are quite standard and are repeated from one task to another, and a reasonable strategy is therefore not to create a new program code for each new task but to use unified tools. The idea arises of creating libraries focused on solving standard data processing problems in a certain area of research.

In solar physics, one of the most famous is the SolarSoft library, which has existed for more than 30 years. Written in the IDL language, it continues to be developed and is currently supported. Most of its functionality is currently implemented in the SunPy library (Table 2) written in Python, which is the main language for machine learning. At the same time, a number of other libraries are emerging that solve the problems of creating homogeneous datasets from various sources of observational data, preprocessing and analyzing them (see examples in Table 2).

We note that the modern practice is to make the source code publicly available from repositories. This allows independently verifying and reproducing the stages of scientific research and using it as a starting point for further work.

| Name | Brief description | Address |
|---|---|---|
| aiapy | Library for calibration and processing of SDO/AIA data | https://aiapy.readthedocs.io/ |
| FlareNet | Code for preparing data and training a solar flare forecast model | https://github.com/nasa-fdl/flarenet |
| Mission ML Data Ready | Library for creating datasets from SOHO and SDO catalogues | https://github.com/cshneider/soho-ml-data-ready |
| observethesun | Collection of machine learning programs and models in solar physics | https://github.com/observethesun |
| SERPENTINE | Tools for loading and analyzing data on solar cosmic rays | https://serpentine-h2020.eu/ |
| SunPy | Library for loading, processing, and analyzing observational data on solar activity | https://sunpy.org/ |
| SpaceML | Collection of programs and models on machine learning in solar physics | https://spaceml.org/ |

### 5. Machine learning models in solar physics

#### 5.1 Segmentation of solar disk images

One of the main types of observational data on the Sun and solar activity is solar disk images. These images can be obtained in white light as well as at specific wavelengths. Observers start processing such images by identifying active regions of the Sun. Sunspots are a well-known example, but they do not exhaust solar activity. In particular, there are areas known as coronal holes. They appear as dark areas in X-ray images of the solar disk (the 193- or 195-Å line is usually chosen). The problem is that not only coronal holes but also some other objects, including solar filaments, appear dark in such images. Thus, the problem of identifying (outlining) coronal holes is nontrivial and ambiguous.

On the other hand, the results of many years of processing, accumulated in catalogues of coronal hole maps, allow creating a machine learning model tasked with reproducing the work of expert observers. A recent proposal was to use a U-Net-type convolutional neural network model for coronal hole segmentation.

The U-Net model consists of a sequence of convolution operations applied to the input image of the solar disk. Perhaps the best-known example of convolution is with a Gaussian kernel, which smoothens the image. The Gaussian kernel contains predetermined values, while kernel values in the convolutional neural network model are trainable parameters. We discuss the advantages of this approach to working with images by comparing it with fully connected models (such as the one shown in Fig. 2).

First, the input to a fully connected model is a vector of values but not a matrix of values. Of course, a matrix can be represented as a vector, but this erases the two-dimensional structure and the concept of locality (nearby pixels in the image being connected to each other). A two-dimensional convolution appears to be a more natural model in this case.

Another issue is related to the number of model parameters. Let us assume that the input matrix (image) is 1000 × 1000 (a typical resolution of existing solar disk images), and the next layer of the fully connected network has at least 100 neurons. Then, the model already has about 10⁸ parameters (the number of parameters is of the order of the number of neurons in the preceding layer times the number of neurons in the next layer). In a convolutional network, a single convolution kernel (which is typically 3 × 3, 5 × 5, or 7 × 7, and rarely bigger) has the number of parameters of the order of 10, and a smaller total number of parameters can even be obtained when using multiple kernels.

In addition, the convolutional model, unlike the fully connected one, is invariant under shifts and to some extent insensitive to the image size. Indeed, the convolution operation is defined on images of any size, and a shift of objects in the input image leads only to a shift in the output image. In a fully connected model, the number of inputs is fixed, and the result of shifting objects in the input image gives a poorly predictable result.

Figure 6 shows some of the images obtained by convolutions with a set of kernels in the internal layers of a trained U-Net model. We note that the model consists of two parts, implementing the principle of an encoder and a decoder. When passing through the encoder branch, the spatial dimensions of the image decrease, but the number of channels increases. It is assumed that, on this path, the features significant for solving the target problem are extracted from the image. On the decoder branch, the features are unfolded into a new image, which in the case of a segmentation problem is a binary matrix with ones corresponding to pixels belonging to the desired objects (for example, coronal holes).

![[Figure 6.png|Figure 6]]
**Figure 6:** Architecture of a U-Net convolutional neural network. Blue arrows: convolution and downsampling operations. Green arrows: transposed convolution (convolution and upsampling). Orange arrows: array concatenation operations. Block captions indicate array sizes (in terms of images: height, width, and number of channels). Images under diagram show individual slices of the arrays outlined with red ellipses.

The model was trained on a long-term record of coronal hole maps of the Kislovodsk Mountain Astronomical Station of the Main (Pulkovo) Astronomical Observatory of the Russian Academy of Sciences. During training, a standard metric was used for the segmentation task in the form of binary cross-entropy (recall that the same metric is standardly used in classification problems; the segmentation problem is considered as a pixel-by-pixel classification problem).

Figure 7 shows an example of segmentation using a trained U-Net model in comparison with some other algorithms. The observed differences, sometimes significant, are the subject of ongoing discussions, which will apparently start coming to definitive conclusions as experience with the use of the obtained data accumulates.

![[Figure 7.png|Figure 7]]
**Figure 7:** Comparison of coronal hole segmentation results using different algorithms: (a) U-Net model, (b) SPoCA algorithm, and (c) HIMERA algorithm.

In this regard, one of the fundamental machine learning problems can be noted once again: to master the methodology of expert data processing and reproduce it on new (or, conversely, historical) data. This will create a long-term homogeneous data series, which is relevant for many problems in solar physics research.

Interestingly, the feature of convolutional models mentioned above—independence from the input image size—was confirmed in a study where a U-Net model trained on solar disk images showed a stable result in segmenting coronal holes on synoptic maps (recall that a synoptic map represents the entire surface of the Sun obtained by gluing together daily solar disk observations).

The idea of using a U-Net-type model for segmenting solar disk images was developed further: for segmenting filaments, active regions on vector magnetograms, regions in the solar corona, and granules on the photosphere.

The study of various architectures continues: a convolutional model used in one study had the input given by a set of images in different spectral lines and a magnetogram, and solved the coronal hole segmentation problem; other researchers considered the detection problem (drawing a bounding box around an object) for identification of coronal holes, sunspots, and prominences; in more complex segmentation problems, in addition to classifying pixels depending on whether they belong to a desired class of objects, individual instances within the class also had to be distinguished (the so-called instance segmentation problem, in contrast to the simpler semantic segmentation problem mentioned above).

#### 5.2 Parametric description of data

The next step after identifying active regions is usually to measure their characteristics. For example, for a group of sunspots, the area, the extent in latitude and longitude, and the angle of inclination to the equator are determined and a certain morphological class is assigned, following, for example, the Zurich classification system. More complex topological characteristics are sometimes considered. A similar analysis is carried out for other types of regions. Mathematically, this is about assigning certain quantitative characteristics to objects. The question of which characteristics can be useful, for example, when studying solar activity or predicting space weather factors, is open; a more general question regarding new methods for quantitatively describing objects can be considered.

A neural network model of a variational autoencoder type was proposed to describe the structure of sunspot groups. The general architecture of the model, shown in Fig. 8, consists of two parts: an encoder and a decoder. It is worth noting that this architecture has much in common with the U-Net model architecture in Fig. 6: both models use the concept of latent space and are based on the idea of compression and restoration.

![[Figure 8.png|Figure 8]]
**Figure 8:** Variational autoencoder architecture. Input image is passed through a sequence of convolutional layers (encoder), resulting in two arrays that are interpreted as parameters of a normal distribution. Random vector from resulting distribution is passed through a second sequence of convolutional layers (decoder), resulting in reconstructed image.

A set of sunspot group images from the catalogue of the Kislovodsk Mountain Astronomical Station was used for training. The task of the model at the training stage was to construct encoder and decoder functions such that their composition (i.e., compression followed by restoration) would reproduce the original image. In addition, the PCA method was used to rank the parameters obtained at the output of the encoder. As a result, a model was constructed that transforms sunspot group images 256 × 256 pixels in size into 283 numerical parameters. Next, the obtained parameters were analyzed and thereby given a physical interpretation. For this, one or more parameters were varied with the others fixed, and the restored images were studied. An example of the obtained images is shown in Fig. 9, whence we can conclude that one of the parameters characterizes the general bipolar or unipolar structure of the group and the other is responsible for the tilt angle of the group. As an application based on the obtained parameters, a model for assessing the complexity and for automatic classification of sunspot groups was proposed.

The idea of studying and using features extracted by machine learning methods also appears in a number of other studies. For example, an autoencoder was used to describe spectral line profiles; it was also shown that the obtained parameters allow a physical interpretation: in particular, they characterize the spectral line's asymmetry and width. An autoencoder model was used to identify anomalous images and localize active regions in a series of solar disk images taken by the SDO satellite. Also based on parameters extracted from solar disk images, a model was proposed for generating synthetic images and predicting the radio emission flux that the parameters of magnetic active regions identified by an autoencoder can serve as precursors for solar flares and can be used to improve the accuracy and lead time of forecasts.

The above examples from solar physics, as well as a number of similar studies in other areas, show that machine learning methods open up new possibilities for describing and analyzing data. It is noteworthy that some of the parameters identified by the models coincide with those on which experts working with data primarily focus. And while an expert comes to a conclusion about the importance of some parameters based on many years of experience and knowledge of the subject area, a machine learning model does this only by collecting statistics. We also note that elements of this approach can often be found in classification, regression, and forecasting models, for example, as additional parameters on which the model is based. In the next section, we consider these constructions in more detail.

### 5.3 Solar activity forecasts

The task of forecasting solar activity parameters arguably accumulates all the main theoretical models and observational data. At the same time, many mechanisms remain not entirely clear and are in fact replaced by empirically (statistically) established connections. The use of machine learning methods is of interest because they allow revealing connections in a much wider data space and, as a result, obtaining more accurate models on the one hand and new ideas for theoretical models on the other.

The forecast tasks fall into two large groups: the forecast of 'rare' events and the forecast of indices. We discuss their properties.

#### 5.3.1 Forecasting rare events

This problem includes forecasting events that can be called rare against the background of long-term solar activity: solar flares (the forecast of powerful flares of classes M and X being of practical interest, first and foremost), coronal mass ejections (CMEs), solar proton events (SPEs), etc. The standard technique is to discretize time (for example, by days) and reduce the problem to a binary classification one: whether an event occurs in a given time interval. Due to the rare nature of events, the number of examples when an event occurred is much less than when it did not, which makes the sample highly unbalanced (for example, for one region with an X-flare, there are 800 regions without flares). When attempting to construct standard forecast models based on machine learning (or another statistical method), it turns out that the models tend to ignore rare events and collapse to a trivial solution: predicting the predominant class. This requires the use of techniques that increase the significance of error for rare events, for example, by introducing weight coefficients into the loss functions, artificially balancing the sample by selecting an equal number of positive and negative examples, or simulating new positive examples.

Another feature resulting from the discretization of time in forecast models is that it is not entirely clear how the accuracy of the model can be assessed. For example, according to a forecast, an event should occur within 24 hours, but it actually occurs after 24 hours and 1 minute. During standard training and testing of the model, such a case is classified as erroneous, although from an intuitive standpoint, the model was close to the correct answer.

Similar problems arise with the very definition of what is considered an event and what is not. In fact, most events are by definition associated with some conventional threshold (for example, the particle energy being greater than 10 MeV for SPE-type events). In this case, a false positive forecast may also be associated not with the global inaccuracy of the model but with the proximity to the threshold value. It may seem that such nuances are somehow leveled out against the majority background, but it is worth remembering that the total number of rare events is small, and an error in a single event can actually affect the entire model. Therefore, the practice is to use not one metric (for example, the fraction of correct predictions) but a set of metrics that characterize the predictive power of the model from different sides. And it is by no means guaranteed that one model that is better than another in terms of one metric would be better in terms of all metrics simultaneously. As an example, in Table 3, we show a number of metrics often encountered in papers on the forecasting of solar activity events. Because most binary classification metrics are calculated from the error matrix, the following notation is used: TP (true positive) is the number of correctly predicted positive classes (an event did occur), FP (false positive) is the number of false positive responses, TN (true negative) is the number of correctly predicted negative classes (no event), FN (false negative) is the number of false negative responses, P = TP + FN is the number of positive classes in a sample, and N = TN + FP is the number of negative classes in a sample.

We note yet another aspect encountered in assessing the model accuracy. The concept of a positive or negative forecast for logistic regression models is of a threshold nature: the model forecast is a number in the range from 0 to 1 (interpreted as the probability of a given object belonging to the positive class), which is then compared with the standard decision threshold 0.5. When the threshold changes, the error matrix and hence the metrics derived from it change. For example, when the threshold increases, the type-I error decreases but the type-II error increases. For an integral assessment of the effect of the threshold, the sensitivity is plotted against 1 - specificity by varying the threshold from 0 to 1. The resulting graph is called the receiver operating characteristic (ROC) curve. By definition, the graph is nondecreasing and connects the origin with the point (1, 1). The area under the graph is called the ROC AUC (area under curve) metric; it is equal to 1 for an ideal classifier (Fig. 10). The ROC AUC metric has a simple probabilistic interpretation: it is the probability that the response of the model to a randomly selected example from the positive class will be higher than the response to a randomly selected example from the negative class.

Models for the prediction of active events are usually constructed for an active region that is a potential source of solar flares, CMEs, SPEs, and other events. Accordingly, the problem of identifying active regions arises, and the accuracy of the forecast depends on its solution. In machine learning, it is important that the models be based on the same sample, otherwise they are difficult to compare. A major contribution to solving this problem was to propose an identification algorithm and a dataset of SHARP (Space-weather HMI Active Region Patches) active regions and their parameters on vector magnetograms. In a subsequent paper, the first model for predicting solar flares was presented using the SHARP catalogue and a machine learning model.

The diversity of subsequent models and comparison of the results is a subject of a separate large review. In particular, a review of solar flare forecasting models is given in a series of studies and in a recent review of SPE forecasting models, more than 30 SPE forecasting models were presented. We discuss one of them based on a neural network model, following the original study.

The SPE forecasting model aggregates the parameters of magnetic active regions from the SHARP dataset observed during the day and forms a forecast for the SPE occurrence on the next day. The parameters of the active region pass through a fully connected neural network, which encodes them into a smaller number of variables. The parameters obtained from different active regions are combined into one vector, which is passed through a second fully connected neural network with a logistic activation function (sigmoid) on the output neuron. The model architecture is shown in Fig. 11. When training the network, the sample imbalance (1 to 34) was taken into account by duplicating positive examples many times. The proposed encoding-aggregation-forecast scheme is quite general and allows numerous variations. For example, it is possible to encode not the parameters of the active region but the image of the active region itself, use additional parameters from other telescopes during aggregation, and replace the neural network model with other machine learning models.

![[Figure 9.png|Figure 9]]
**Figure 9:** Reconstruction of images of sunspot groups by varying two parameters (denoted as Z1 and Z5) that encode the image. Interpretation follows from the figure: parameter Z1 encodes the unipolar or bipolar structure of the group, and Z5 defines its tilt.

![[Figure 10.png|Figure 10]]
**Figure 10:** Examples of ROC curves for different classifiers. Each point on a curve corresponds to model accuracy metrics (sensitivity and 1 - specificity) for a certain classification threshold; to plot the curve, threshold is varied from zero to one. Green dashed line is a purely random forecast model. Red curve is a model that works better than random. Orange curve is a model that is close to an error-free one. Blue curve is a model that gives inverse predictions (meaning that interpreting its predictions exactly to the contrary would give a good classifier). Area under ROC curve is called ROC AUC metric.

![[Figure 11.png|Figure 11]]
**Figure 11:** Neural network architecture for predicting SPEs. First neural network (AR block) encodes parameters of active region. Obtained parameters from different active regions are fed to input of second neural network, which outputs forecast.

| Name | Definition | Interpretation |
|---|---|---|
| Precision | TP / (TP + FP) | Fraction of correct answers in positive forecasts |
| Accuracy | (TP + TN) / N | Fraction of correct answers |
| False positive rate (FPR), type-I error | FP / N | False positive prediction probability |
| False negative rate (FNR), type-II error | FN / P | False negative prediction probability |
| Sensitivity, recall, true positive rate (TPR) | TP / P | True positive prediction probability, 1 - type-II error |
| Specificity, selectivity, true negative rate (TNR) | TN / N | True negative prediction probability, 1 - type-I error |
| F1 score | 2 * precision * recall / (precision + recall) | Harmonic mean of precision and recall |
| True Skill Statistic (TSS) | TP/P - FP/N | 1 - type-I error - type-II error |

**Table 3:** Binary classification metrics.

#### 5.3.2 Forecasting solar activity indices

Unlike the problem of forecasting rare events, which in the standard setting is a classification problem, the forecast of solar activity indices (for example, the solar cycle, X-ray fluxes, and the Kp index of geomagnetic activity) is based on time series. The difference between the two standpoints is that classification is an interpolation problem, while time series forecasting is an extrapolation problem. This difference also propagates into the structure of the models. As regards neural networks, fully connected and convolutional networks are more common in the first case, and recurrent neural networks (RNNs), in the second case. A feature of recurrent models is the memory mechanism, which allows accumulating information when passing through a sequence of unlimited length. Figure 12 shows the simplest example of an RNN, in which, for a current observed value $x_t$, the internal state $h_t$ is updated by the rule

$$h_t = f(U x_t + V h_{t-1}), \quad (12)$$

and the output (predicted) value $y_t$ is calculated as a function of the internal state,

$$y_t = g(W h_t), \quad (13)$$

where $f$ and $g$ are activation functions, and $U$, $V$, and $W$ are trainable parameters of the model.

In practice, more complex architectures are used that involve the concept of long short-term memory, analyze the sequence bidirectionally, and contain several levels of recurrent models.

In 2021 (the beginning of the growth of the 25th solar cycle), a comparative graph of 34 forecasts of the amplitude of the 25th solar cycle was presented, demonstrating a large uncertainty in the forecast for the future, even though all methods successfully predicted past cycles (Fig. 13).

In the machine learning framework, this situation can be explained as follows. A forecast of the next cycle maximum can be obtained either by directly predicting the maximum or by obtaining a forecast of the entire cycle, for example, using a recurrent model (Fig. 14), and then finding its maximum. In the first case, the size of the training sample is equal not to the number of years of observations of solar cycles but to the number of known maxima. Due to the large uncertainty of historical data, this number does not exceed several dozen. In the second case, a well-known problem arises: how to ensure the stability of a model that uses previous forecast values when constructing a forecast. For this, the model can be trained on a long forecast (of the order of the cycle length), but this again narrows the effective volume of the training sample to a size of the order of the number of solar cycles, as in the first case. Thus, a highly underdetermined problem arises (the volume of the training sample is several dozen, and a typical neural network model has orders of magnitude more free parameters), resulting in a large uncertainty in the forecast.

Nevertheless, studies of the possibilities of solar cycle forecasting using machine learning methods are ongoing, contributing to the study of fundamental problems in the nature of solar cyclicity.

The forecast of geomagnetic activity indices using machine learning methods is the subject of some studies; the forecast of solar wind is the subject of other studies. We note that these studies address the problem of short-term forecasts (several days in advance), which significantly increases the volume and variability of the training sample and allows expecting a more stable forecast.

![[Figure 12.png|Figure 12]]
**Figure 12:** Structure of simplest recurrent neural network. Left of gray arrow: compact view. New internal state $h_{t+1}$ is calculated based on current $h_t$ and observed $x_t$ values; output value $y_t$ is a function of internal state $h_t$; $U$, $V$, and $W$ are trainable parameters of the model. Right of gray arrow: expanded view of calculation of sequence of internal states and output values.

![[Figure 13.png|Figure 13]]
**Figure 13:** Forecasts of amplitude of 25th solar cycle.

![[Figure 14.png|Figure 14]]
**Figure 14:** Forecast based on recurrent model: $x_t$ is last observed value of time series. Forward forecast is built on the basis of preceding forecast values $\hat{x}_{t+k}$.

### 5.4 Reconstruction of observational data

A close relation exists between the different solar activity phenomena underlain by solar magnetic fields. The presence of this intrinsic relation allows posing the machine learning problem to transform observational data of one type into others. As a first example, we consider the problem of reconstructing polarity maps of a large-scale magnetic field based on observations of solar filaments.

It is assumed that solar filaments (prominences) form along the neutral lines of the solar magnetic field. Knowing the positions of the filaments, we can estimate the position of the neutral line and then arrange the polarity signs such that different signs are on different sides of the neutral line. The finer details of this process were described in literature, but even taking them into account leaves the result of polarity map reconstruction ambiguous, and the main catalogues of polarity maps were created manually over a long period of time. The idea of an automated approach, proposed recently, is based on posing an optimization problem and solving it by a function defined by a fully connected neural network. A polarity map filled with plus and minus signs is regarded as the surface of a smooth function depending on the heliographic coordinates, where the plus sign corresponds to the function value +1 and the minus sign, to -1, the function value along the neutral line being 0. The optimization problem is constructed based on the idea that the filament coordinates define the coordinates at which the function is equal to zero, and on a number of physical constraints that prevent the convergence to the trivial (zero) solution. It turns out that the solution to such an optimization problem is close to a polarity map constructed manually by an expert (Fig. 15).

A finer structure of the magnetic field (e.g., coronal magnetic fields) can be reconstructed from photospheric observations. Typically, this is done using a force-free field model (system of equations),

$$\nabla \cdot \mathbf{B} = 0, \quad (14)$$
$$(\nabla \times \mathbf{B}) \times \mathbf{B} = 0,$$

with photospheric magnetograms $\mathbf{B}_0(x, y)$ taken as the boundary conditions:

$$\mathbf{B}(x, y, 0) = \mathbf{B}_0(x, y). \quad (15)$$

The numerical solution to this problem is greatly complicated by the fact that the observational data do not fully agree with the force-free field model, and a possible solution must violate one condition or the other. The situation that has arisen contradicts standard numerical methods for solving differential equations, but can be implemented in the framework of the PINN model. Let us recall that the idea is that the neural network defines a function of spatial coordinates, which is then optimized to satisfy the differential equation and boundary conditions as accurately as possible. In the context of reconstructing the coronal magnetic field, such an idea was demonstrated in recent studies (Fig. 16). In another study, the PINN model was used to simulate the propagation of shock waves in the solar atmosphere.

Another example of a reconstruction problem is the translation of solar disk images obtained in one spectral region into images from another spectral region (image-to-image translation). In one study, a convolutional neural network model was proposed for reconstructing SDO/AIA images in the UV range using SDO/HMI magnetograms. A solution to the inverse problem was presented using a generative cGAN model (see the example in Fig. 17). As an application, the authors presented magnetograms of the far side of the Sun based on STEREO/EUVI observations. Further development of translation methods and new combinations of images were studied in other works.

The task of increasing the spatial resolution of images and reducing noise can also be assigned to this group of tasks. One of the first attempts to use a convolutional neural network in this context was made in early studies. The model was trained on pairs of images with different spatial resolutions obtained by magnetohydrodynamic (MHD) modeling, and testing was carried out on real SDO/HMI magnetograms (Fig. 18). For further studies of models based on machine learning, see recent reviews.

The problem of the reliability of the resulting reconstructions was discussed in specialized literature. The authors conclude that, although the results are statistically quite plausible, the reconstruction quality is significantly reduced for rare and powerful events.

The reconstruction problem is especially relevant when working with historical observational data stored in the form of handwritten notes and sketches. Using such data for systematic analysis requires digitization, and machine learning can help here when working with large catalogues. For example, in one study, the problem of digitizing more than 10,000 pages of handwritten tables specifying the positions of spots, faculae, and prominences from the Zurich Observatory archives was solved. The main difficulty was the large variability of handwriting, and therefore the model trained on a limited (albeit large) subsample of examples had difficulty recognizing text from a new handwriting. The solution was an adaptive approach: the model (a convolutional recurrent neural network, CRNN) was automatically retrained on each new page, using only the fact that the numbers in the tables with coordinates were not completely arbitrary but manifested some dependence. An example of the digitization result is shown in Fig. 19.

We note that the adaptive approach significantly extends the capabilities of machine learning models. In the classical approach, a model is trained on a sample consisting of pairs of independent and dependent variables $(x_i, y_i)$, and is then used unaltered on new data (supervised learning). In practice, however, significant statistical changes can occur in the data over time, which causes the quality of the model to decline. Again, the standard solution is to prepare a new training sample and retrain the model. However, this is not always possible: changes may occur frequently, and the process of preparing training data usually takes a long time. Moreover, the fact that there is a decrease in the model quality is usually apparent not immediately but only after some time. Therefore, approaches are being developed in which models are automatically and continuously adjusted using self-training.

![[Figure 15.png|Figure 15]]
**Figure 15:** Reconstruction examples of a magnetic field polarity map for three synoptic rotations (by rows). (a) Synoptic map specifying positions of solar filaments. (b) Polarity map constructed by an expert. (c) Averaging over 100 realizations of neural network reconstruction model. (d) Binarized version of (c).

![[Figure 16.png|Figure 16]]
**Figure 16:** Coronal magnetic field reconstruction above an active region using PINN model.

![[Figure 17.png|Figure 17]]
**Figure 17:** Example of operation of SDO/HMI magnetogram reconstruction model based on SDO/AIA image of solar disk in 304-Å line. (a) Original SDO/AIA image, (b) reconstruction result, and (c) original SDO/HMI image.

![[Figure 18.png|Figure 18]]
**Figure 18:** *(Place for Figure 18: Super-resolution of SDO/HMI magnetograms)*

![[Figure 19.png|Figure 19]]
**Figure 19:** *(Place for Figure 19: Digitization of historical Zurich Observatory tables)*