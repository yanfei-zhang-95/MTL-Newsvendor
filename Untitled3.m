clc
clear
close
%%
load NewsvendorData_Label;
x = XtrainB;
y = Dtrain;
%%
beta = pinv(x*x')*(x*y');
%%
predicted_val = beta' * XtestB;
error = (predicted_val - Dtest).^2;
disp(mean(error))