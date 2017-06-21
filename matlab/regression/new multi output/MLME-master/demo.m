addpath(genpath('CTBN'))
addpath(genpath('CC'))
addpath(genpath('MLME'))
addpath(genpath('liblinear-weights-2.01'))
addpath(genpath('LBFGS'))
addpath('base')
addpath('utils')

load('data/scene.mat');

global LR_implementation;
LR_implementation = 'liblinear';
lambda = 1;

HO = cvpartition(Y(:,1), 'HoldOut', .7);
X_tr = X(HO.training,:);
Y_tr = Y(HO.training,:);
X_ts = X(HO.test,:);
Y_ts = Y(HO.test,:);

% CC = [];
% [CC.model, CC.permutation] = CCs2_train_weighted(X_tr, Y_tr, ones(length(Y_tr),1));
% % CCmodel =  CC.model;
% % CCW =[];
% % mind =size(CCmodel{1},2);
% % CCW=[];
% % for i=1:length(CCmodel)
% %     a=CCmodel{i};
% %     CCW=[CCW;a(1:mind)]
% % end
% [Y_pred, Y_log_prob] =  CC_predict(CC.model, X_ts, Y_ts, CC.permutation);
% RS_CC = getMeasuresMLC(Y_ts, Y_pred, Y_log_prob);
% 
% MLME_CC = train_MLME(X_tr, Y_tr, 3, lambda, 50, 'CC');
% [Y_pred, Y_log_prob] = MAP_prediction_MCC(MLME_CC.base, MLME_CC.gate, X_ts, Y_ts);
% RS_MCC = getMeasuresMLC(Y_ts, Y_pred, Y_log_prob);
% 
CTBN = learn_weighted_structure(X_tr, Y_tr, ones(length(Y_tr),1), []);
[Y_pred, Y_log_prob] = MAP_prediction(CTBN, X_ts, Y_ts);
RS_CTBN = getMeasuresMLC(Y_ts, Y_pred, Y_log_prob);

% MLME_CTBN = train_MLME(X_tr, Y_tr, 3, lambda, 50, 'CTBN');
% [Y_pred, Y_log_prob] = MAP_prediction_MCTBN(MLME_CTBN.base, MLME_CTBN.gate, X_ts, Y_ts);
% RS_MCTBN = getMeasuresMLC(Y_ts, Y_pred, Y_log_prob);

% RS_CC
% RS_MCC
% RS_CTBN
% RS_MCTBN