.. _Linear_regression:

=================
Linear Regression
=================

.. contents::
    :local:
    :depth: 2


Introduction
============

Linear Regression is a supervised machine learning algorithm where the predicted output is continuous and has a constant slope. It's used to predict values within a continuous range, (e.g. sales, price) rather than trying to classify them into categories (e.g. cat, dog). There are two main types:

.. rubric:: Simple regression

Simple linear regression uses traditional slope-intercept form, where :math:`m` and :math:`b` are the variables our algorithm will try to "learn" to produce the most accurate predictions. :math:`x` represents our input data and :math:`y` represents our prediction.

.. math::

  y = mx + b