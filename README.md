# What Makes a Satisfying Life? Prediction and Interpretation with Machine-Learning Algorithms

In the repository, there are all the scripts that I have written to get the results in the paper "What Makes a Satisfying Life? Prediction and Interpretation with Machine Learning Algorithms", the first chapter of my doctoral thesis. The scripts are written in Python and R.

The paper was then written in collaboration with Andrew Clark, Conchita D'Ambrosio and Alexandre Tkatchenko.

Considering two specifications - called Original and Extended - we explored the potential of Machine Learning algorithms in predicting and explaining the determinants of life satisfaction. Starting from the specification used in Layard et al. (2014), here called the Original, we augmented the model adding more variables describing at a more granular level the same variables.

We considered multiple types of Penaliwed Linear Regression - Ridge, LASSO, and Elastic Nets with alpha at 0.25, 0.50, and 0.75 - as well as Random Forests. Experiments with other algorithms (including Neural Networks, Gradient Boosting, Kernel Ridge Regressions and Mixture Density Networks) have been done, but are
not presented here as they're not included in the paper.

Moreover, we investigate the potential of Model-Agnostic Interpretative algotrithms such as Shapley Values and their associaed mean absolute values, "Derived" Coefficients from linear specifications and Permutation Importances.

All the scripts are rich in comments, which you can gloss over if you already have expertise in programming with R and Python. The reason for this choice is 
that people working in wellbeing often come from disciplines in which the mainstreams are STATA, SPSS or SAS. Hence, I hope in this manner the results are easier 
to then reproduce. All the scripts have, in their name, the order in which they need to be run.

In order to obtain the results, it is necessary to have access to the British Cohort Study data. 

Since the paper is currently under review at academic journals, it is not published here.
