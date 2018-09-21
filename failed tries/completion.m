load NewsvendorData_Label_;

D = cell(1,d);
for i = 1:d
   D{i} = demand(find(label_tag == i));
end

mtx = zeros(d, n);
for i = 1:d
    mtx(i, :) = repmat(D{i},1,d);
end

demand = mtx;