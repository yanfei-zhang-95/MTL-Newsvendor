clc
clear
close
%%
a = 1:10;
%%
mtx = zeros(10,10);
for i=1:10
    b = 1;
    mtx(i,b) = a(i);
    for j=1:10
        if j ~= i
            b = b + 1;
            mtx(i, b) = a(j);
        end
    end
end

%%
load NewsvendorData_Label_
%%
[~,O] = size(unique(Ltrain));
[~, train_] = size(XtrainB);
D = cell(1,O);
for i = 1:O
   D{i} = Dtrain(find(Ltrain == i));
end
%%
mtx = zeros(O, train_);
for i = 1:O
    count_start = 1;
    label = length(D{i});
    count_end = count_start + label;
    mtx(i, count_start:count_end-1) = D{i};
    count_start = count_end;
    for j = 1:O
        if j ~= i
            label = length(D{j});
            count_end = count_start +label;
            mtx(i, count_start:count_end-1) = D{j};
            count_start = count_end;
        end
    end
end