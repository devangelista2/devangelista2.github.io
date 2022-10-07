---
title: Classification with clustering algorithms
categories: [Lab2, teaching]
tags: [LDA, PCA, clustering, dimensionality reduction, SVD, unsupervised learning, classification]
math: true
---
## What is classification?
Consider, as an example, the MNIST dataset we used to test PCA and LDA clustering algorithm. Here, we have a dataset $X \in \mathbb{R}^{d \times N}$ representing handwritten digits, and a vector $Y \in \mathbb{R}^N$ such that the $i$-th element of $Y$ is the digit represented in the $i$-th column of $X$. When this is the case, we say that the numbers collected in $Y$ are the _classes_ associated with the elements in $X$. 

An important task in Machine Learning is to understand the semantic relationship between a datapoint $x$ and the corresponding class $y$. This task is called **classification**.

> **_Classification:_** A classification algorithm is a model mapping a datapoint $x \in \mathbb{R}^d$ to the corresponding class $y$, chosen among a **finite** set $\mathcal{C} = \\{ C_1, \dots, C_K \\}$.

In practice, when developing a ML classification algorithm, we are required to implement a rule $f_\theta(x)$, associating $x$ to a _potential_ class $\hat{y} = f_\theta(x)$. The classification guess is correct if the predicted class $\hat{y}$ equals the right class $y$. 

### Decision boundaries
Most of the classification algorithm is based on the concept of **decision boundaries**. The idea is to _learn_ a curve (either a straight line or a smooth, non-linear curve), called a _boundary_, from the training set and then, given a test datapoint, it will be classified to the first class if it is below the curve, to the second class if it is above the curve. This idea is summarized in the following Figure.

![](https://www.researchgate.net/publication/349186066/figure/fig1/AS:989978611953666@1613040702298/Example-of-overfitting-in-classification-a-Decision-boundary-that-best-fits-training.png)

### Measuring the accuracy
The accuracy of a classification algorithm is easy to compute. We can just consider the percentage of correct guesses of the model over the test set. Mathematically, this is implemented as

$$
    Acc(f_\theta) = \frac{1}{N_{test}} \sum_{i=1}^{N_{test}} \mathbb{1}(f_\theta(x_i) = y_i)
$$

where

$$
    \mathbb{1}(f_\theta(x_i) = y_i) = \begin{cases} 1 \qquad \text{if } f_\theta(x_i) = y_i \\ 0 \qquad \text{if } f_\theta(x_i) \neq y_i \end{cases}
$$

the higher the accuracy of a classification algorithm, the better it is in practice.

## Classification with clustering
Clustering can be used to define a classification algorithm. In particular, here, the idea is that since by definition the datapoints that are semantically similar are close together in the projected space of a clustering algorithm, we can hope that points living in the same class, will be mapped inside of the corresponding cluster. Consequently, we can define a classification algorithm in the following way:

- Compute the projection matrix $P \in \mathbb{R}^{k \times d}$ associated with a clustering algorithm;
- In the projected space, compute the _centroid_ of each cluster by taking the mean of the projected points living in the same class;
- Given a new datapoint $x \in \mathbb{R}^d$, project it into the cluster space by $z = Px$;
- Compute the distance between $z$ and each cluster centroid $c_i$, $i=1, \dots, K$ as $d_i = \|\| z - c_i \|\|_2^2$;
- Classify $x$ to be the class of the centroid such that $d_i$ is the smallest.

Note that the algorithm above naturally defines decision boundaries, as descripted in the following Figure.

![](https://images.deepai.org/glossary-terms/d78ba3ab5b644ea6baa74604e6559f76/cluster-analysis.jpg)