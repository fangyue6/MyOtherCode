
% load the data
load('example_data.mat');

% get the predictions for the test data
[test_outputs,test_labels]=MIMLfast(train_data,train_targets,test_data);