.. _Linear_regression:

=================
Linear Regression
=================

.. contents::
    :local:
    :depth: 2


Introduction
============

Assume we have some :math:`n` training data points :math:`{(X_i, Y_i)}_{i = 1}^{n}` that comes from an known/unknown distribution :math:`\mathcal{P}_{XY}`. Where, the :math:`X_i \in \mathcal{X} \in \mathbb{R}^d`
comes from the input distribution :math:`\mathcal{P}_{X}` and :math:`Y_i \in \mathcal{Y} \in \mathbb{R}` comes from a conditioned distribution :math:`\mathcal{P}_{Y/X}`. Now, we have a new data point :math:`X_{n+1}` and
we need to estimate the output for that point. We can think of an example that a hospital has some patients data. :math:`X_i` consists of some risk factors such as age, blood pressure, height, body weight, etc., for a patient and 
:math:`Y_i` is a measure of kidney function(e.g., eGFR). Using the previous :math:`n` data points, our goal is to estimate the output of the new data point :math:`X_{n+1}`.

We can design an algorithm :math:`A \in \mathcal{A}` that takes the :math:`n` data points as an input and return a learned function :math:`f \in \mathcal{F}` such that :math:`f: \mathcal{X} \rightarrow \mathcal{Y}`. Now, if we know the learned :math:`f` 
we can estimate the new output for the :math:`X_{n+1}`. There are different ways to learn the :math:`f` function. We will discuss some of them in this sections with python codes. We name linear regression because the predicted target value :math:`\hat{Y}` is 
expected to be a linear combination of the features of :math:`X`. In mathematical notation, we can write :math:`\hat{Y}` as follows:

.. math::

\hat{Y}(\theta, X) = \theta^{0} * X^0 + \theta^{1}*X^1 + \theta^{2} * X^2 +...+ \theta^{d} * X^d + b

Where, :math:`\theta = [\theta^0, \theta^1, \theta^2, ...., \theta^d] \in mathcal{\theta}` is the parameters  and :math:`b` is the bias term. 

Ordinary Least Squares
======================

