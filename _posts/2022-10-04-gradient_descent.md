---
title: Gradient descent
categories: [Lab3, teaching]
tags: [gradient descent, optimization, minimization]
math: true
---
## Optimization in Machine Learning
Training a Machine Learning model usually ends up to an _optimization problem_. As an example, we already saw that <a href="{{ site.baseurl }}{% link _posts/2022-09-27-LDA.md %}">LDA algorithm</a> requires to solve

$$
    \max_{q} q^T S_w q \qquad \text{s.t.} \> q^T S_b q = 1
$$

Moreover, it can be shown that <a href="{{ site.baseurl }}{% link _posts/2022-09-26-PCA.md %}">PCA</a> is equivalent to find the directions $q_1, \dots, q_K$ with $\|\|q_i\|\| = 1$ for $i = 1, \dots, K$, over which the spread of the data is maximized. Mathematically, this can be re-written as

$$
    \max_{q: ||q||_2=1} || q^T X ||_2
$$

but

$$
    \max_{q: ||q||_2=1} || q^T X ||_2 = \max_{q: ||q||_2=1} \frac{|| q^T X ||_2}{||q||_2}  = ||X||_2
$$

by the definition of matrix norm. The relationship between PCA and SVD comes from the fact that $\|\| X \|\|_2 = \sigma_1$, where $\sigma_1$ is the largest eigenvalue of $X$.

## Introduction to Optimization
From the examples above, it is clear that it is necessary to find a _general_ way of solving optimization algorithm. For the sake of generality, from now on we will consider a general _continuous_ function $f: \mathbb{R}^n \to \mathbb{R}$. Given a set of _admissible solutions_ $\Omega \subseteq \mathbb{R}^n$, an optimization problem is a problem of the form

$$
    \min_{x \in \Omega} f(x)
$$

If $\Omega = \mathbb{R}^n$, we say that the optimization is **unconstrained**. If $\Omega \subset \mathbb{R}^n$, the problem is **constrained**. In this first part, we will always assume $\Omega = \mathbb{R}^n$, i.e. the unconstrained setup.

### Convexity
A common assumption in optimization is the _convexity_. By definition, a function $f: \mathbb{R}^n \to \mathbb{R}$ is convex if 

$$
    f(tx_1 + (1-t)x_2) \leq tf(x_1) + (1-t)f(x_2) \qquad \forall x_1, x_2 \in \mathbb{R}^n, \quad \forall t \in [0, 1]
$$

![](https://upload.wikimedia.org/wikipedia/commons/thumb/c/c7/ConvexFunction.svg/1200px-ConvexFunction.svg.png)

The importance of convex functions in optimization is that if $f(x)$ is convex, than every minimum point $x^*$ of $f(x)$ is a _global minimum_ and the set of global minima is connected (the $n$-dimensional extension of the concept of _interval_). On the other side, if a function is non-convex (_NOTE:_ the opposite of convex is **not** concave), then there can be multiple distinct minimum points, some of which are _local minimum_ while others are _global minimum_. Since in ML applications we want to find global minima, and discriminating between local and global minima of a function is an NP-hard problem, having convexity is a good thing. 

![](https://miro.medium.com/max/1400/1*itDZ27aJkg3-eVFYIV13Ww.png)

### First-order sufficient condition
Most of the algorithms to find the minimum points of a given function $f(x)$ are based on the following property:

>**_First-order sufficient condition:_** If $f: \mathbb{R}^n \to \mathbb{R}$ is a continuously differentiable function and $x^* \in \mathbb{R}^n$ is a minimum point of $f(x)$, then $\nabla f(x^*) = 0$. 

Moreover, it holds

>**_First-order necessary condition:_** If $f: \mathbb{R}^n \to \mathbb{R}$ is a continuously differentiable function and $\nabla f( x^* ) = 0$ for $x^* \in \mathbb{R}^n$, then $x^*$ is either a (local) minimum, a (local) maximum or a saddle point of $f(x)$.

Consequently, we want to find a point $x^* \in \mathbb{R}^n$ such that $\nabla f(x^*) = 0$. Those points are called **stationary points** of $f(x)$.

![](https://www.offconvex.org/assets/saddle/minmaxsaddle.png)

## Gradient descent (GD)
The most common algorithm to solve optimization problems is the so-called Gradient Descent (GD). It is an iterative algorithm, i.e. an algorithm that iteratively updates the estimate of the solution, converging to the correct solution after infinite steps, such that, at each iteration, the successive estimate is computed by moving in the direction of maximum decreasing of $f(x)$: $- \nabla f(x)$. Specifically, 

$$
    x_{k+1} = x_k - \alpha_k \nabla f(x_k) \qquad k = 0, 1, \dots
$$

where the initial iterate, $x_0 \in \mathbb{R}^n$, is given as input and the **step-size** (equivalently, **learning rate**) $\alpha_k > 0$ controls the decay rapidity of $f(x)$ for any $k \in \mathbb{N}$. 