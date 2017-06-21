function [ Y_pred, Y_log_prob ] = MAP_prediction( T, X, Y, is_switching )

% param
is_profiling = false;

if ~exist('is_switching','var')
    is_switching = true;
end

% init
Y_pred = zeros( size(X,1), length(T) );
Y_log_prob = zeros( size(X,1), 1 );
if is_profiling, pf_comp = 0; pf_maxsum = 0; end;
is_multiclass = false;
for i = 1:length(T)
    if T{i}.card > 2
        is_multiclass = true;
    end
end

% proc
for i = 1:size(X,1)
    x = X(i,:);
    t1 = clock;
    [ T ] = compute_log_potentials(T, x, is_switching);
    %save( 'T', 'T' );
    t2 = clock;
    if ~is_multiclass  % binary instances
        Y_pred(i,:) = maxsum_forest(T);
    else  % multi-calss instances
        Y_pred(i,:) = naive_MAP_inference(T);
    end
    t3 = clock;
    
    Y_log_prob(i) = evaluate_probability(T,  Y(i,:));
    
    if is_profiling
        pf_comp = pf_comp + etime(t2,t1);
        pf_maxsum = pf_maxsum + etime(t3,t2);
    end
end




if is_profiling
    fprintf( '(pf)for-compute_log_potentials: %f s\n', pf_comp );
    fprintf( '(pf)for-maxsum_forest: %f s\n', pf_maxsum );
end