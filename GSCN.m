%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Greedy Stochastic Configuration Netsworks Class (Matlab)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Copyright (C) 2022
classdef GSCN
    properties
        Name = 'Greedy Stochastic Configuration Networks';
        version = '1.0 beta';
        % Basic parameters (networks structure)
        L       % hidden node number / start with 1
        W       % input weight matrix
        b       % hidden layer bias vector
        Beta    % output weight vector
        % Configurational parameters
        tol     % tolerance
        Lambdas % random weights range, linear grid search
        r
        L_max   % maximum number of hidden neurons
        T_max   % Maximum times of random configurations
        nPop = 30 % number population
        eta = 0.1
        % Else
        nB = 1 % how many node need to be added in the network in one loop
        verbose = 50 % display frequency
        COST = 0   % final error
        mode = 1   % 1: GSCN-I 2: GSCN-II
        alpha = 10^-6
    end
    %% Funcitons and algorithm
    methods
        %% Initialize one SCN model
        function obj = GSCN(L_max, T_max, tol, Lambdas, r, eta, alpha,mode)
            format long;  
            obj.L = 1;
            if ~exist('L_max', 'var') || isempty(L_max)
                obj.L_max = 100;
            else
                obj.L_max = L_max;
                if L_max > 5000
                    obj.verbose = 500; % does not need too many output
                end
            end
            if ~exist('T_max', 'var') || isempty(T_max)
                obj.T_max=  100;
            else
                obj.T_max = T_max;
            end
            if ~exist('tol', 'var') || isempty(tol)
                obj.tol=  1e-4;
            else
                obj.tol = tol;
            end
            if ~exist('Lambdas', 'var') || isempty(Lambdas)
                obj.Lambdas =  1:0.1:5;
            else
                obj.Lambdas = Lambdas;
            end
            
            if ~exist('r', 'var') || isempty(r)
                obj.r= 0.99999999;
            else
                obj.r = r;
            end
            
            if ~exist('eta', 'var') || isempty(eta)
                obj.eta=  0.1;
            else
                obj.eta = eta;
            end
            obj.mode = mode;
            obj.alpha = alpha;
        end
        
        
        
        function [obj, Ksi_t] = Fitness(obj, E0, hPpos, X, r_L)
            [n,d] = size(X); % Get Sample and feature number
            [~,m] = size(E0);
            % Calculate kesi_1 ... kesi_m
            w = hPpos(1,1:d);
            w = w';
            bb = hPpos(1,d+1:d+1);
            H_t = logsig(bsxfun(@plus, X*w, bb));
           
            ksi_m = zeros(1, m);
            for i_m = 1:m                            
                eq = E0(:,i_m);
                gk = H_t;
                tmp1 = ((eq'*gk)^2)/(gk'*gk);
                tmp2 = (eq'*eq);
                xi_L = tmp1 - (1 - r_L) * tmp2;

                if (gk'* gk == 0 || xi_L <= 0 )
                     ksi_m(i_m) = inf;
                else
                    ksi_m(i_m) = tmp2 - tmp1 + obj.eta /(obj.L) * (norm(w)^2 + norm(bb)^2);
                end
            end 
             Ksi_t = sum(ksi_m); 
        end
        
        %% Search for best {WB, bB, Lambda} with HPO
        function [WB, bB, Flag] = SC_Search_HPO(obj, X, E0)
            
            for i=1:length(obj.Lambdas)
                Lambda = obj.Lambdas(i);
                for j=1:length(obj.r)
                    r_L = obj.r(j);
                    [WB, bB, Flag] = SC_Search(obj,X, E0, Lambda, r_L);
                    if Flag == 0
                        break;
                    end
                end
                if Flag == 0
                    break;
                end
            end
        end
        
        function [WB, bB, Flag] = SC_Search(obj, X, E0, Lambda, r_L)
            Flag =  1;% 0: continue; 1: stop; return a good node /or stop training by set Flag = 1
            WB  = [];
            bB  = [];
            [~,d] = size(X); % Get Sample and feature number
            [~,m] = size(E0);
            % HPO Parameters
            MaxIt = obj.T_max;
            
            dim = d + 1;
            ub = Lambda;
            lb = -Lambda;
            
            % Constriction Coefeicient
            B = 0.1;
            % Initialization
%              global HPpos 
            HPpos = rand(obj.nPop, dim).*(ub - lb) + lb;
%              global HPposFitness 
            HPposFitness = zeros(obj.nPop, 1);
            for i=1:size(HPpos,1)
                [obj, HPposFitness(i)] = obj.Fitness(E0, HPpos(i,:), X, r_L);
            end
            
            [~,indx] = min(HPposFitness);
            Target = HPpos(indx,:);   % Target HPO
            TargetScore =HPposFitness(indx);
            for it = 2:MaxIt

                c = 1 - it*((0.98)/MaxIt);   % Update C Parameter
                kbest=round(obj.nPop*c);        % Update kbest
                for i = 1:obj.nPop
                    r1=rand(1,dim)<c;
                    r2=rand;
                    r3=rand(1,dim);
                    idx=(r1==0);
                    z=r2.*idx+r3.*~idx;
                    if rand<B
                        xi=mean(HPpos);
                        dist = pdist2(xi,HPpos);
                        [~,idxsortdist]=sort(dist);
                        % disp(['--',num2str(kbest), ' --- ', num2str(size(idxsortdist))])
                        SI=HPpos(idxsortdist(kbest),:);
                        HPpos(i,:) =HPpos(i,:)+0.5*((2*(c)*z.*SI-HPpos(i,:))+(2*(1-c)*z.*xi-HPpos(i,:)));
                    else
                        for j=1:dim
                            rr=-1+2*z(j);
                            
                            HPpos(i,j)= 2*z(j)*cos(2*pi*rr)*(Target(j)-HPpos(i,j))+Target(j);

                        end
%                         rr = -1 + 2 * z;
%                         HPpos(i,:) = (2 * z) .* cos(2*pi*rr).*(Target - HPpos(i,:)) + Target; 
                    end
                    HPpos(i,:) = min(max(HPpos(i,:),lb),ub);
                    % Evaluation
                    [obj, HPposFitness(i)] = obj.Fitness(E0, HPpos(i,:), X, r_L);
                    % Update Target
                    if HPposFitness(i)<TargetScore
                        Target = HPpos(i,:);
                        TargetScore = HPposFitness(i);  
                    end
                end
                %disp(['Iteration: ',num2str(it),' Best Cost = ',num2str(TargetScore)]);
            end
            WB = Target(1:d)';
            bB = Target(d+1:d+1);
            % check whether satisfy the inequality equation or not
            gk = logsig(bsxfun(@plus, X*WB, bB));
            ksi_m = zeros(1, m);
            for i_m = 1:m
                eq = E0(:,i_m);
                [obj, ksi_m(i_m)] = obj.InequalityEq(eq, gk, r_L);
            end
            if min(ksi_m) > 0
                Flag = 0;
            end
        end
        
         %% inequality equation return the ksi
        function  [obj, ksi] = InequalityEq(obj, eq, gk, r_L)
            ksi = ((eq'*gk)^2)/(gk'*gk) - (1 - r_L)*(eq'*eq);
        end
        
        %% Add nodes to the model
        function obj = AddNodes(obj, w_L, b_L)
            obj.W = cat(2,obj.W, w_L);
            obj.b = cat(2,obj.b, b_L);
            obj.L = length(obj.b);
        end
        
        %% Compute the Beta, Output, ErrorVector and Cost
        function [obj, O, E, Error] = UpgradeSCN(obj, X, T)
            H = obj.GetH(X);

            obj = obj.ComputeBeta(H,T);
            
            O = H*obj.Beta;
            E = T - O;
            Error =  Tools.RMSE(E);

            obj.COST = Error;
        end      
        
        %% ComputeBeta  
        function [obj, Beta] = ComputeBeta(obj, H, T)
            if obj.mode == 2
                A = H;
                [m,n] = size(A);
                [U,S,V] = svd(A,0);
                r = sum(diag(S)>obj.alpha*S(1,1));
                [Q,R,P] = qr(V(:,1:r)');
                Atilde = A*P;
                [Qtilde,Rtilde] = qr(Atilde(:,1:r),0);
                z = Rtilde\(Qtilde'*T);
                [~, d] = size(T);
                Beta = P*[z;zeros(n-r,d)];
                
                obj.Beta = Beta;
            else
                Beta = pinv(H)*T;
                obj.Beta = Beta;

            end
        end  
        
        %% Regression
        function [obj, per] = Regression(obj, X, T)             
            per.Error = [];
            E = T;
            Error =  Tools.RMSE(E);
            disp([obj.Name, " mode:", obj.mode]);
            per.Error = cat(2, per.Error, repmat(Error, 1, obj.nB));
            while (obj.L < obj.L_max) && (Error > obj.tol)            
                if mod(obj.L, obj.verbose) == 0
                    fprintf('L:%d\t\tRMSE:%.6f \r', obj.L, Error );
                end
                [w_L, b_L, Flag] = SC_Search_HPO(obj, X, E);% Search for candidate node / Hidden Parameters
                if Flag == 1
                    break;% could not find enough node
                end
                obj = AddNodes(obj, w_L, b_L);                 
                [obj, ~ , E, Error ] = obj.UpgradeSCN(X, T); % Calculate Beta/ Update all                
                %log
                per.Error = cat(2, per.Error, repmat(Error, 1, obj.nB));
            end% while
            fprintf('#L:%d\t\tRMSE:%.6f \r', obj.L, Error );
            disp(repmat('*', 1,30));
        end
        
        %% Classification
        function [obj, per] = Classification(obj, X, T)            
            per.Error = []; % Cost function error
            per.Rate = [];  % Accuracy Rate
            E = T;
            Error =  Tools.RMSE(E);            
            Rate = 0;
            disp([obj.Name, " mode:", obj.mode]);
            while (obj.L < obj.L_max) && (Error > obj.tol)
                if mod(obj.L, obj.verbose) == 0
                    fprintf('L:%d\t\t RMSE:%.6f; \t\tRate:%.2f\r', obj.L, Error, Rate);
                end
                [w_L, b_L, Flag] = SC_Search_HPO(obj, X, E);
                if Flag == 1
                    break;% could not find enough node
                end
                obj = AddNodes(obj, w_L, b_L);                
                [obj, ~, E, Error ] = obj.UpgradeSCN(X, T); % Calculate Beta/ Update all
                O = obj.GetLabel(X);
                Rate = 1- confusion(T',O');
                % Training LOG
                per.Error = cat(2, per.Error, repmat(Error, 1, obj.nB));
                per.Rate = cat(2, per.Rate,  repmat(Rate, 1, obj.nB));
            end% while
            fprintf('#L:%d\t\t RMSE:%.6f; \t\tRate:%.2f\r', obj.L, Error, Rate);
            disp(repmat('*', 1,30));
        end    
        
        
        %% Output Matrix of hidden layer
        function H = GetH(obj, X)
            H =  obj.ActivationFun(X);
        end
        % Sigmoid function
        function H = ActivationFun(obj,  X)
            H = logsig(bsxfun(@plus, X*[obj.W],[obj.b]));              
        end
        %% Get Output
        function O = GetOutput(obj, X)
            H = obj.GetH(X);
            O = H*[obj.Beta];
        end
        %% Get Label
        function O = GetLabel(obj, X)
            O = GetOutput(obj, X);
            O = Tools.OneHotMatrix(O);
        end
        %% Get Accuracy
        function [Rate, O] = GetAccuracy(obj, X, T)
            O = obj.GetLabel(X);
            Rate = 1- confusion(T',O');
        end
        %% Get Error, Output and Hidden Matrix
        function [Error, O, H, E] = GetResult(obj, X, T)
            % X, T are test data or validation data
            H = obj.GetH(X);
            O = H*(obj.Beta);
            E = T - O;
            Error =  Tools.RMSE(E);
        end
 
    end % methods
end % class
