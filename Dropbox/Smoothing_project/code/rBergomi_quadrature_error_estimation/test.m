%addpath(genpath(pwd))

%% Problem
func=@myfunction;   % function
q=2;                % number of outputs
d_ind=0;            % index dimension
N_init=2;   % initial probability space dimension
N=10;        % probability space dimension

%% Settings
sg_struct.func=func;
sg_struct.N=N;
sg_struct.d_ind=d_ind;
sg_struct.scheme=['CC','CC']; % sparse grid scheme
sg_struct.scheme_detail={[0.5,0.2],[0.01,0.02]};
sg_struct.I=[];

w=5;
adm=@(i) sum(i-1)<=w; % Smolyak set at level w=4
profit=@(i) 1; % testing

%% Create index set
num_add=-1; % -1 = all.
[sg_struct.I,I_added]=get_indexset(sg_struct,adm,profit,num_add);


%% Build sparse grid
[evalObj,knots_lvl,weights_lvl]=sgbuild(sg_struct);
sg_struct.evalObj=evalObj;
sg_struct.knots_lvl=knots_lvl;
sg_struct.weights_lvl=weights_lvl;
C_comb=get_combination(sg_struct.I);
sg_struct.C=C_comb;

%input=keys(evalObj);
%for i=1:1:length(input)
%    evalObj(input{i})
%end

%% Evaluate sparse grid interpolation
samples=rand(1000,N);
[Y_out,Z,Y]=sginterp(sg_struct,samples);
%M=delta_to_comb(sg_struct.I);

X=[sg_struct.scheme_detail{1};sg_struct.scheme_detail{2}];	% input domain
for i=1:1:size(samples,1),
for n=1:1:sg_struct.N
	if sg_struct.scheme(n)=='CC'  
    	samples(i,n)=sg_struct.scheme_detail{n}(1)+(sg_struct.scheme_detail{n}(2)-sg_struct.scheme_detail{n}(1))*samples(i,n);
    end
end
end
y_true=zeros(size(samples,1),q);
for j=1:1:size(samples,1),
     y_true(j,:)=func(samples(j,:));
end
mean(y_true)

%% Evaluate sparse grid quadrature
Yquad=sgquad(sg_struct);



%% Validation 
%samples=repmat(X(1,:),size(samples,1),1)+samples.*(repmat(X(2,:),size(samples,1),1)-repmat(X(1,:),size(samples,1),1));

% quadrature comparison
rel_quad_error=abs(mean(y_true)-Yquad)/mean(abs(y_true))

% interpolation comparison
rel_interp_error_1=mean(abs(y_true(:,1)-Y_out(:,1)))/mean(abs(y_true(:,1)))
rel_interp_error_2=mean(abs(y_true(:,2)-Y_out(:,2)))/mean(abs(y_true(:,2)))


% 
% q_choice=1;
% size(Z)
% Z_delta_mean=mean(Z,2);
% Z_delta_mean=squeeze(Z_delta_mean);
% Z_delta_mean=Z_delta_mean(:,q_choice);
% 
% I=sg_struct.I;
% pre=log(abs(Z_delta_mean(2:end,:)));
% size(pre)
% m_1=i2m(I(2:end,1),'CC');
% m_2=i2m(I(2:end,2),'CC');
% predictors=[ones(size(pre)),m_1,m_2];
% [b,bint,r,rint,stats]=regress(pre,predictors);
% C=exp(b(1));
% g=[-b(2)*log(exp(1)),-b(3)*log(exp(1))];
% g
% err_fun=@(i) C*exp(-sum(i2m(i,'CC').*g));
% 
% i_1=1:1:(max(I(:,1))+1);
% i_2=1:1:(max(I(:,2))+1);
% [I_1,I_2]=meshgrid(i_1,i_2);
% 
% Z_plot=zeros(size(I_1));
% Z_est_plot=zeros(size(I_1));
% for i=1:1:max(i_1)
%     for j=1:1:max(i_2)
%         idx=[i,j];
%         [tf,index]=ismember(idx,I,'rows');
%         if i==1 && j==1,
%             Z_plot(i,j)=NaN;
%             Z_est_plot(i,j)=NaN;
%         else
%         if tf==1
%             Z_plot(i,j)=Z_delta_mean(index);
%             Z_est_plot(i,j)=err_fun(idx);
%         else
%             Z_plot(i,j)=NaN;
%             Z_est_plot(i,j)=NaN;
%         end
%         end
%     end
% end
% figure(1);
% surf(I_1,I_2,log10(abs(Z_plot)));
% colorbar;
% lim=caxis;
% caxis([-30,-3]);
% view(2);
% figure(2);
% surf(I_1,I_2,log10(abs(Z_est_plot)));
% colorbar;
% lim=caxis;
% caxis([-30,-3]);
% view(2);