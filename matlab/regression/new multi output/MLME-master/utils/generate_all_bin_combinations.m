function [ Y ] = generate_all_bin_combinations( d )

for i = 1:power(2,d)
    bin = dec2bin(i-1);
    for j = 1:d
        if j <= d-length(bin);
            Y(i,j) = 0;
        else
            Y(i,j) = bin2dec(bin(j-(d-length(bin))));
        end
    end
end
end
