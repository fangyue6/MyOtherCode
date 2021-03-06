algs = {'AllFea', 'LapScore', 'SPFS', 'UDFS', 'LLCFS', 'MCFS',  'NDFS', 'RUFS', 'JELSR_lpp', 'GLSPFS', 'FSSL_11_11_5'};
lab_cluster = 'local'; % the matlab distributed computing server (MDCS) name, you may use 'local' as default 
lab_cluster_size = 11; % number of node
lab_email_username = '';% the email notification service provided by our lab, you can also use other public email configuration.
lab_email_password = '';
ds = {'USPS_9298n_256d_10c', 'wap_1560n_8460d_20c_tfidf',  ...
    'webbb_texas_814n_4029d_7c_binary', 'webkb_washington_1166n_4165d_7c_binary', ...
    'Carcinom_174n_9182d_11c', 'binaryalphadigs_1404n_320d_36c'};
ds = {'JAFFE_213n_676d_10c'}; % demo data
for i1 = 1:1%length(ds)
    dataset = ds{i1};
    job = batch(@run_exp1_func, 4, {dataset, algs, 'lab_email_username', 'lab_email_password'},...
        'Profile', lab_cluster, 'pool', lab_cluster_size, ...
        'AttachedFiles', {[dataset, '.mat'], 'eppMatrix.mexa64', 'eppMatrix.mexglx'},...
        'CaptureDiary',true, 'CurrentDirectory', '.');
end
