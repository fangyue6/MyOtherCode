function [ T T_weight] = directed_maximum_spanning_tree( S, all_nodes, root, node_weights )

%sort the edges by weight
for i=1:length(S)
    W(i)=S{i}.weight;
end

selected_nodes=[root];
T={};
T_weight=0;

if exist('W', 'var')
    [W edge_order]=sort(W,'descend');

    for i=1:length(edge_order)
        e=S{edge_order(i)};

        %Each node has at most one parent
        if(isempty(find(selected_nodes==e.to)))

           %node for the source
           idx_source = search_tree( e.from, T );
           idx_dest=search_tree( e.to, T );

           %both source and dist are not found
           if(idx_source == -1 && idx_dest == -1)
               T = add_tree_node( T, length(T)+1, e.from, [], e.to);
               T = add_tree_node( T, length(T)+1, e.to, e.from, []);
               selected_nodes=[selected_nodes e.to];
               T_weight=T_weight+e.weight;
           end

           %if source not found, but dist is found
           if(idx_source == -1 && idx_dest ~= -1)
               T = add_tree_node( T, length(T)+1, e.from, [], e.to);
               T{idx_dest}.parent=e.from;
               selected_nodes=[selected_nodes e.to];
               T_weight=T_weight+e.weight;

           end


           %if source is found, but dist is not found
           if(idx_source ~= -1 && idx_dest == -1)
               T = add_tree_node( T, length(T)+1, e.to, e.from, []);
               T{idx_source}.children=[T{idx_source}.children e.to];
               selected_nodes=[selected_nodes e.to];
               T_weight=T_weight+e.weight;

           end

           %both source and dist are found: check for cycles!!
           if(idx_source ~= -1 && idx_dest ~= -1)

               %check that no cycle will be created
               if(~check_cycle( T, e.to, e.from))
                   T{idx_source}.children=[T{idx_source}.children e.to];
                   T{idx_dest}.parent=e.from;
                   selected_nodes=[selected_nodes e.to];
                   T_weight=T_weight+e.weight;

               end
           end

        end
    end
else
    fprintf(2,'warn: all class variables are independent; equiv to BR.\n');
end


%all the rest of nodes as independent nodes
for i=1:length(all_nodes)
    if(search_tree(all_nodes(i),T)==-1)
         T = add_tree_node( T, length(T)+1, all_nodes(i), [], []);
    end
end

%add the weights for all nodes without parents
for i=1:length(T)
    if(isempty(T{i}.parent))
         T_weight=T_weight+node_weights(T{i}.node);
    end
end

