Introduction:

This package contains a Matlab implementation (tested on Matlab R2016a) of Multi-Label Mixtures-of-Experts (ML-ME) [Hong, Batal, and Hauskrecht, 2015] that builds ensemble mixtures of structured prediction models for multi-label classification.

To train a ML-ME model, use MLME/train_MLME.m. To use a trained model for prediction, use MLME/MAP_prediction_MCC.m or MLME/MAP_prediction_MCTBN.m depending on your modeling option ([Read et al., 2009; Batal, Hong, and Hauskrecht, 2013]).

demo.m contains a demonstration script that learns and uses the ML-ME models on the Scene dataset [Boutell et al., 2004].


----

Disclaimer:

This code package can be used for academic purposes only. We do not guarantee that the code is correct, current or complete, and do not provide any technical support. Accordingly, the users are advised to confirm the correctness of the package before making any decisions with it.


----

Reference:

[Hong, Batal, and Hauskrecht, 2015] C. Hong, I. Batal, and M. Hauskrecht. A generalized mixture framework for multi-label classification. SIAM International Conference on Data Mining, Vancouver, BC, Canada. April 2015.

[Batal, Hong, and Hauskrecht, 2013] I. Batal, C. Hong, and M. Hauskrecht. An efficient probabilistic framework for multi-dimensional classification. ACM International Conference on Information and Knowledge Management, Burlingame, CA, USA. October 2013.

[Read et al., 2009] J. Read, B. Pfahringer, G. Holmes, and E. Frank. Classifier chains for multi-label classification. Joint European Conference on Machine Learning and Knowledge Discovery in Databases. Springer Berlin Heidelberg, 2009.

[Fan et al., 2008] R. Fan, K. Chang, C. Hsieh, X. Wang, and C. Lin. LIBLINEAR: A library for large linear classification, Journal of Machine Learning Research 9(2008), 1871-1874. Software available at http://www.csie.ntu.edu.tw/~cjlin/liblinear

[Schmidt, 2005] M. Schmidt. minFunc: unconstrained differentiable multivariate optimization in Matlab. http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html, 2005.

[Boutell et al., 2004] M. Boutell, J. Luo, X. Shen, and C. Brown. Learning multi-label scene classification. Pattern Recognition, 37(9):1757-1771, 2004.
