addpath(genpath(pwd))

%% Problem
func=@mybergomi_function;   % function
q=1;                % number of outputs
d_ind=0;            % index dimension
N=16;              % probability space dimension

%% Settings
sg_struct.func=func;
sg_struct.N=N;
sg_struct.d_ind=d_ind;
sg_struct.scheme=['H','H','H','H','H','H','H','H','H','H','H','H','H','H','H','H']; % sparse grid scheme
sg_struct.scheme_detail={[0.0,1],[0.0,1],[0.0,1],[0.0,1],[0.0,1],[0.0,1],[0.0,1],[0.0,1],[0.0,1],[0.0,1],[0.0,1],[0.0,1],[0.0,1],[0.0,1],[0.0,1],[0.0,1]};

%sg_struct.scheme=['C','C','C','C','C']; % sparse grid scheme
%sg_struct.scheme_detail={[0.2,0.5],[0.1,0.3],[0.05,0.15],[0.01,0.05],[0.01,0.05]};
sg_struct.I=[];

%% Validation set
val_size=10; % does not impact the errors (quad and interpolation)
samples_norm=randn(val_size,N); % generates val_size x N normal random vector: samples_norm
samples=zeros(size(samples_norm));

for i=1:1:size(samples,1),  % size(samples,1)=val_size
for n=1:1:sg_struct.N
	if sg_struct.scheme(n)=='C'  
    	samples(i,n)=sg_struct.scheme_detail{n}(1)+(sg_struct.scheme_detail{n}(2)-sg_struct.scheme_detail{n}(1))*samples_norm(i,n);
    end
    if sg_struct.scheme(n)=='H'   % generates samples with a given distribution dicated by sg_struct.scheme_detail
    	samples(i,n)=sg_struct.scheme_detail{n}(1)+sg_struct.scheme_detail{n}(2)*samples_norm(i,n);
    end
end
end

%Y_true computes val_size realization of the function evaluated at created
%samples
Y_true=zeros(size(samples,1),q);
for j=1:1:size(samples,1),
     Y_true(j,:)=func(samples(j,:));
end

%% Initial Sparse grid - Smolyak set
w=2; %level w of Smolyak  sparse grid
% the previous param is controlling to which level to reach

adm=@(i) sum(i-1)<=w;   % Smolyak set at level w
profit=@(i) 1;          % equal profit function
num_add=-1;             % -1 = add all, 1 = add the highest profit, 2=add the two of highest profit
% the previous param is controlling which indices to include
[sg_struct.I,I_added]=get_indexset(sg_struct,adm,profit,num_add); % get indexset

[evalObj,knots_lvl,weights_lvl]=sgbuild(sg_struct); % build the sparse grid
sg_struct.evalObj=evalObj; 
sg_struct.knots_lvl=knots_lvl;
sg_struct.weights_lvl=weights_lvl;
C_comb=get_combination(sg_struct.I);
sg_struct.C=C_comb; %this is not clear to me what is it

%% SG quadrature evaluation
[Q_out,Zquad]=sgquad(sg_struct); %Zquad not clear to me what is it


%% Error model
qoi_choice=1; % error estimate wrt first quantity of interest

logerr=log(abs(Zquad(2:end,qoi_choice))); % start at 2 as 1st not a difference
%logerr  it is not clear to me what it means exactly
M=i2m(sg_struct.I(2:end,:),sg_struct.scheme); % this gives the number of points
                                             %for each index vector (keep
                                             %in mind that we may chnage
                                             %i2m for H because it is
                                             %different to what we are
                                             %using in MISC code
predictors=[ones(size(logerr))];
for n=1:1:N
    predictors=[predictors,M(:,n)];
end
[b,bint,r,rint,stats]=regress(logerr,predictors);
C=exp(b(1));
g=[];
for n=1:1:N
    g=[g,-b(1+n)];
end
err_fun=@(i) C*exp(-sum(i2m(i,sg_struct.scheme).*g)); %C not needed in front
% err_fun is not well understood to me and where have been used, well it
% seems giving the rate of the decay of quad error but not sure hwere to
% use it
%% Create index set

Error_I=[];Q_vec=[];Count=[];
K=30; % number of points to compute the error (and it increases as we increase the number 
      %of indices by selecting the most profitable
for k=1:1:K,
    %% Update index set
    w_max=20; adm=@(i) max(i-1)<=w_max;  % Restrict allowed indices to full tensor-product at level w
    num_add=1;%-1; % -1 = add all
    profit=@(i) err_fun(i)./prod(i); %Profit estimator (e.g., for Gauss-Hermite)
    %profit=@(i) err_fun(i)./(prod(2.^(i)-1)-prod(2.^(i-1)-1)); %(e.g., Clenshaw-Curtis)
    
    [sg_struct.I,I_added]=get_indexset(sg_struct,adm,profit,num_add); % add indices

    %% Build sparse grid
    [evalObj,knots_lvl,weights_lvl]=sgbuild(sg_struct);
    sg_struct.evalObj=evalObj;
    sg_struct.knots_lvl=knots_lvl;
    sg_struct.weights_lvl=weights_lvl;
    C_comb=get_combination(sg_struct.I);
    sg_struct.C=C_comb;
    
    %% Sparse grid interpolation on samples
    [Y_out,Z]=sginterp(sg_struct,samples_norm);
    
    %% Sparse grid quadrature
    [Q_out,Zquad]=sgquad(sg_struct);

    %% Error model
    
    % Fit error rates and the associated constant
    logerr=log(abs(Zquad(2:end,qoi_choice))); % start at 2 as 1st not a difference
    M=i2m(sg_struct.I(2:end,:),sg_struct.scheme); 
    predictors=[ones(size(logerr))];
    for n=1:1:N
        predictors=[predictors,M(:,n)];
    end
    [b,bint,r,rint,stats]=regress(logerr,predictors);
    C=exp(b(1));
    g=[];
    for n=1:1:N
        g=[g,-b(1+n)];
    end
    % Error estimator
    err_fun=@(i) C*exp(-sum(i2m(i,sg_struct.scheme).*g)); %C not needed in front
    I_margin=get_margin(sg_struct.I);

    %% Validation 
    Count=[Count,length(sg_struct.evalObj)];
    
    % collect interpolation error vector
   %rel_interp_error_1=mean(abs(Y_true(:,qoi_choice)-Y_out(:,qoi_choice)))/mean(abs(Y_true(:,qoi_choice)));
   %as was done by Joakim
    %rel_interp_error_1=mean(abs(Y_true(:,qoi_choice)-Y_out(:,qoi_choice)));
     %without normalization 
    rel_interp_error_1=mean(abs(Y_true(:,qoi_choice)-Q_out(:,qoi_choice)));    %we just use the mean for
    % the difference between the the function evaluation and the SG
    % solution 
    Error_I=[Error_I,rel_interp_error_1];
    
    % collect quadrature solutions
    Q_vec=[Q_vec,Q_out(qoi_choice)]; %Error of first QoI
    
end

%% Error plot
figure(1);
loglog(Count(1:(end-1)),abs(Q_vec(1:(end-1))-Q_vec(end)),'-bo',Count,Error_I,'-kx');
hold on;
legend('Quadrature','Interpolation');
xlabel('Number of collocation points')
ylabel('Error')

%% Overkill Monte Carlo solution
ok_bool=0;
if ok_bool
    ok_size=1000000;
    samples_norm=randn(ok_size,N);
    samples=zeros(size(samples_norm));

    for i=1:1:size(samples,1),
    for n=1:1:sg_struct.N
        if sg_struct.scheme(n)=='C'  
            samples(i,n)=sg_struct.scheme_detail{n}(1)+(sg_struct.scheme_detail{n}(2)-sg_struct.scheme_detail{n}(1))*samples_norm(i,n);
        end
        if sg_struct.scheme(n)=='H'  
            samples(i,n)=sg_struct.scheme_detail{n}(1)+sg_struct.scheme_detail{n}(2)*samples_norm(i,n);
        end
    end
    end
    Y_ok=zeros(size(samples,1),q);
    for j=1:1:size(samples,1),
         Y_ok(j,:)=func(samples(j,:));
    end
    mean(Y_ok)
    disp(['SG-Quadrature versus Monte Carlo: ',num2str(abs(Q_vec(end)-mean(Y_ok(:,qoi_choice))))])
end