.. _Linear_regression:

=================
Linear Regression
=================

.. contents::
    :local:
    :depth: 2


Introduction
============

Assume we have some :math:`n` training data points :math:`{(X_i, Y_i)}_{i = 1}^{n}` that comes from an known/unknown distribution :math:`\mathcal{P}_{XY}`. Where, the :math:`X_i \in \mathbb{R}^d`
comes from the input distribution :math:`\mathcal{P}_{X}` and :math:`Y_i \in \mathbb{R}` comes from a conditioned distribution :math:`\mathcal{P}_{Y/X}`. Now, we have a new data point :math:`X_{n+1}` and
we need to estimate the output for that point. We can think of an example that a hospital has some patients data. :math:`X_i` consists of some risk factors such as age, blood pressure, height, body weight, etc., for a patient and 
:math:`Y_i` is a measure of kidney function(e.g., eGFR). Using the previous :math:`n` data points, our goal is to estimate the output of the new data point :math:`X_{n+1}`.

We can design an algorithm :math:`A \in \mathcal{A}` that takes the :math:`n` data points as an input and return a learned function :math:`f in \mathcal{F}` such that :math:`f: \mathcal{X} \rightarrow \mathcal{Y}`.

