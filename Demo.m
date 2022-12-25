% Demo Data regression of SCN
clear;
clc;
close all;
format long;

load('Function_Data.mat');

   
%% Parameter Setting
L_max = 50;                    % maximum hidden node number
tol = 0.00001;                    % training tolerance
T_max = 100;                    % maximun candidate nodes number
Lambdas = [200:1:250];% scope sequence
r =  [0.999999]; % 1-r contraction sequence
nB = 1;       % batch size
alpha  = 1e-6;
MaxIter = 20;


%% Model Initialization
gscn = GSCN(L_max, MaxIter, tol, Lambdas, r, 0, alpha, 1);
% gscn = GSCN(L_max, MaxIter, tol, Lambdas, r, 0, alpha, 2);

%% Model Training
% M is the trained model
% per contains the training error with respect to the increasing L

[gscn, gscn_per] = gscn.Regression(X, T);
disp(gscn);
%% Training error demo
figure;
plot(gscn_per.Error, 'r.-'); hold on;
xlabel('L');
ylabel('RMSE');
legend('Training RMSE');

[gscn_test_rmse, ~] = gscn.GetResult(X2, T2);
disp(['GSCN Test RMSE: ', num2str(gscn_test_rmse)]);



%% Model output vs target on training dataset
gscn_O1 = gscn.GetOutput(X);
figure;
plot(X,T, 'r.-'); hold on;
plot(X,gscn_O1, 'b.-');  
xlabel('X');
ylabel('Y');
legend('Training Target', 'GSCN');

%% Model output vs target on test dataset
gscn_O2 = gscn.GetOutput(X2);
figure;
plot(X2, T2, 'r.-'); hold on;
plot(X2, gscn_O2, 'b.-');
xlabel('X');
ylabel('Y');
legend('Test Target', 'GSCN');


% % The End 


