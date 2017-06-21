% maxsum_forest: returns the joint assignment with the highest probability
% (wrapper for maxsum_tree)
function [ assn logprob ] = maxsum_forest( T )

% options
is_verbose = false;

% init
n_nodes = size(T, 2);

% traverse + examine T
lookup = nan(n_nodes, 1);   % node# -> cell_index lookup table
max_node_i = -1;
for i = 1:n_nodes
    lookup(T{i}.node) = i;
    
    if( max_node_i < T{i}.node )
        max_node_i = T{i}.node;
    end
end

% init for returns
assn = nan(1, max_node_i);
logprob = 0;

% proc
for i = 1:n_nodes
    if isempty(T{i}.parent)
        % bfs a tree; build a new cell S with that single tree
        root = T{i}.node;
        
        S = cell(1,1);
        s_count = 0;
        
        %stack = java.util.Stack();
        %stack.push(root);
        st = zeros(n_nodes+1, 1); % empty: st(1)=1 / full: st(1)==n_nodes+1
        st(1)=1;    % empty
        st(1)=st(1)+1;
        st(st(1)) = root;
        
        while st(1) ~= 1 %~stack.empty
            %node = stack.pop();
            node = st(st(1)); st(1)=st(1)-1;
            l = lookup(node);
            
            s_count = s_count + 1;
            S{s_count} = T{l};
            
            for j = 1:length(T{l}.children)
                %stack.push(T{l}.children(j));
                st(1)=st(1)+1;
                st(st(1)) = T{l}.children(j);
            end
        end
        
        % invoke the inner function to find MAP assignmnet
        if is_verbose
            [ a_assn a_logprob ] = maxsum_tree( S )
        else
            [ a_assn a_logprob ] = maxsum_tree( S );
        end
        
        for j = 1:size(a_assn,2)
            if ~isnan(a_assn(j))
                assn(j) = a_assn(j);
            end
        end
        logprob = logprob + a_logprob;
    end
end

end % -- end of 'function [ assn logprob ] = maxsum_forest( T )'


% maxsum_tree: inner function that returns the joint assignment with the highest probability
function [ assn logprob ] = maxsum_tree( T )

% options
is_debug = false;

% type check (T:cell, T{i}:struct)


% init
n_nodes = size(T,2);
root = nan;

% color coding
%(WHITE=unvisited; GRAY=visited, but incomplete; BLACK=finished)
WHITE = 0; GRAY = 1; BLACK = 2;

T_aux = cell(size(T));
for i = 1:n_nodes
    T_aux{i}.color = WHITE; % bookkeeping for traverse/expand T
    T_aux{i}.n_msg_req = 0;
    T_aux{i}.msg = [];  % T{i}.node -> T{i}.parent
    T_aux{i}.assn = []; % T{i}.node
end

% traverse + examine T
max_node_i = -1;
for i = 1:n_nodes
    if( max_node_i < T{i}.node )
        max_node_i = T{i}.node;
    end
end

lookup = nan(max_node_i, 1);   % node# -> cell_index lookup table
for i = 1:n_nodes
    lookup(T{i}.node) = i;
    T_aux{i}.n_msg_req = length(T{i}.children);
    
    if isempty(T{i}.parent)
        if ~isnan(root)
            fprintf( 'err: multiple roots exist\n' );
            return;
        end
        root = T{i}.node;
    end
end

if isnan(root)
    fprintf( 'err: no root exists\n' );
    return;
end


% structure verification (optional)
if is_debug
    root
    lookup
    % check links
end


% proc
%stack = java.util.Stack();
%stack.push(root);
st = zeros(n_nodes+1, 1); % empty: st(1)=1 / full: st(1)==n_nodes
st(1)=1;    % empty
st(1)=st(1)+1;
st(st(1)) = root;
        

while st(1) ~= 1    %~stack.empty
    %node = stack.pop();
    node = st(st(1)); st(1)=st(1)-1;
    i = lookup(node);
    if(T_aux{i}.color == WHITE)
        if isempty(T{i}.children)   % leaf node
            % maximization
            [ T_aux{i}.msg T_aux{i}.assn ] = max(T{i}.log_potential, [], 2);
            T_aux{i}.assn = T_aux{i}.assn - 1;  % convert index[1,2] to label[0,1]
            
            if( T{i}.node ~= root )     % if ~isempty(T{i}.parent)
                r = lookup(T{i}.parent);    % receiver's index in T
                T_aux{r}.n_msg_req = T_aux{r}.n_msg_req - 1;
            end
            T_aux{i}.color = BLACK;
            
            if is_debug
                node
                [ a b ] = max(T{i}.log_potential, [], 2)
            end
        else                        % otherwise
            %stack.push(T{i}.node);
            st(1)=st(1)+1;
            st(st(1)) = T{i}.node;
            for j = 1:length(T{i}.children)
                % stack.push(T{i}.children(j));
                st(1)=st(1)+1;
                st(st(1)) = T{i}.children(j);
            end
            T_aux{i}.color = GRAY;
        end
    elseif(T_aux{i}.color == GRAY)
        if(T_aux{i}.n_msg_req == 0)
            % maximization
            sum_msg = [0;0];
            for j = 1:length(T{i}.children)
                child_node = T{i}.children(j);
                child_i = lookup(child_node);
                sum_msg = sum_msg + T_aux{child_i}.msg;
            end
            
            if size(T{i}.log_potential,2) == size(sum_msg,1)
                tmp = T{i}.log_potential;
                for j = 1:size(tmp,1)
                    tmp(j,:) = tmp(j,:) + sum_msg';
                end
                [ T_aux{i}.msg T_aux{i}.assn ] = max(tmp, [], 2);
                T_aux{i}.assn = T_aux{i}.assn - 1;
                
                if( T{i}.node ~= root )
                    r = lookup(T{i}.parent);    % receiver's index in T
                    T_aux{r}.n_msg_req = T_aux{r}.n_msg_req - 1;
                end
                
                T_aux{i}.color = BLACK;
                
                if is_debug
                    node
                    tmp
                    [ a b ] = max(tmp, [], 2)
                end
            else
                fprintf( 'err: while traverse (0x02)\n' );
            end
        else
            fprintf( 'err: while traverse (0x03)\n' );
        end
    end
end

% traverse verfication
if is_debug
    sum = 0;
    for i = 1:n_nodes
        sum = sum + T_aux{i}.color;
    end
    
    if sum == BLACK * n_nodes
        fprintf('msg: 1st pass - all went well\n');
    else
        fprintf('err: 1st pass - something is missing (%d != %d)\n', sum, BLACK*n_nodes);
        
    end
end


% return
assn = nan(1, n_nodes);
% root assignment
i = lookup(root);
assn(i) = T_aux{i}.assn;
logprob = T_aux{i}.msg;

for j = 1:length(T{i}.children)
    %stack.push(T{i}.children(j));
    st(1)=st(1)+1;
    st(st(1)) = T{i}.children(j);
end

while st(1) ~= 1    %~stack.empty
    %node = stack.pop();
    node = st(st(1)); st(1)=st(1)-1;
    i = lookup(node);
    parent_i = lookup(T{i}.parent);
    if( length(T_aux{parent_i}.assn) ~= 1 )
        fprintf( 'err: while traverse (0x05)\n' );
    end
    T_aux{i}.assn = T_aux{i}.assn(T_aux{parent_i}.assn+1);
    assn(i) = T_aux{i}.assn;
    
    for j = 1:length(T{i}.children)
        %stack.push(T{i}.children(j));
        st(1)=st(1)+1;
        st(st(1)) = T{i}.children(j);
    end
end

% re-ordering
tmp = nan(1, max_node_i);
for i = 1:size(lookup, 1)
    if(lookup(i)>0 && lookup(i)<=max_node_i)
        tmp(i) = assn(lookup(i));
    end
end
assn = tmp;

end % -- end of 'function [ assn logprob ] = maxsum_tree( T )'
