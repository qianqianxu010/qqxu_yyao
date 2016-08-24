% Assume the first column is user_id,
% the second and the third columns are about candidates_id,
% the last column is outcome y
clear;
clc;

load bt_simu.mat;
data =  Pair_Compar;
kappa =20;
nt = 50;
trate = 50;

m = size(data,1); %% number of comparison
n = max(max(data(:,2:3)));  %% number of item
p = max(data(:,1)); %% number of user

u = data(:,1);
i = data(:,2);
j = data(:,3);
y = data(:,4);

%%%%   y = d * s + x1 * delta + x2 * gamma
d = sparse([1:m,1:m],[i;j],[ones(1,m),-ones(1,m)],m,n);
x1 = sparse([1:m,1:m],[u+(i-1)*p,u+(j-1)*p],[ones(1,m),-ones(1,m)],m,p*n);
x2 = sparse(1:m,u,ones(1,m),m,p);
x = [x1,x2];

%%%% split the training and test
p_train = 1;
train = (rand(1,m)<p_train);
test = ~train;
train = find(train);

%%%% training models
% LB
group = (1:p)'*ones(1,n);
group = [group(:)',(p+1):(2*p)];
tic()
result = lb_xqq(d(train,:),x(train,:),y(train),kappa,[],[],nt,trate,group,1);
toc()
t = result.tlist;
alpha = result.alpha;


%%% Cross-validation
K =5;
folds = mod(randperm(length(train)),K)+1;
% LB
residmat = zeros(K,nt);
for i=1:K
    tic()
    fit = lb_xqq(d(train(folds~=i),:),x(train(folds~=i),:),y(train(folds~=i)),kappa,alpha,t,[],[],group);
    i
    toc()
    res = y(train(folds==i))*ones(1,nt).*( d(train(folds==i),:)*fit.s_path +x(train(folds==i),:)*fit.path);
    residmat(i,:) = (1-mean(sign(res)))/2;
end
cv_error = mean(residmat);
cv_sd = sqrt(var(residmat)/K);

errorbar(1:nt,cv_error,cv_sd,cv_sd);
%%% Determine the optimal choice
k = find(cv_error==min(cv_error));
delta_1=result.path(1:n*p,k);

temp=reshape(delta_1,p,n);
t_bias=temp;
preference_bias_user=[];
for i=1:p
    deltas_sum=sum(temp(i,:));
    if deltas_sum~=0
        preference_bias_user=[preference_bias_user
            i];
    end
end
gamma=result.path(n*p+1:end,k);
temp2=find(gamma~=0);
position_bias_user=temp2;
%%%%%Jumping time
delta_sum=zeros(p,50);
for i=1:50
    delta_1=result.path(1:n*p,i);
    temp=reshape(delta_1,p,n);
    for j=1:p
        delta_sum(j,i)=sum(abs(temp(j,:)));
    end
end
gamma2=result.path(n*p+1:end,:);

t=result.tlist;
a =zeros(p,2)+Inf;
for i = length(t):-1:1
    a(gamma2(:,i)~=0,1) = t(i)
    a(delta_sum(:,i)~=0,2)= t(i)
end

[position_id position_index]=sort(a(:,1));
detected1=[position_index position_id];
[preference_id preference_index]=sort(a(:,2));
detected2=[preference_index preference_id];

position_detected=detected1(1:length(position_bias_user),:);
preference_detected=detected2(1:length(preference_bias_user),:);
L2_distance_preference=zeros(length(preference_bias_user),0);
data=Pair_Compar;

detected=preference_detected;
[compare,~]=size(data);
user = data(:,1);
y = data(:,4);
data = data(:,2:4);

score_GT=result.s_path(:,k);
[a_GT b_GT]=sort(score_GT,'descend');
list_GT=[a_GT b_GT];
sample_num=[];

for ID=1:length(preference_bias_user)
    t=find(Pair_Compar(:,1)==detected(ID,1));
    temp=Pair_Compar(t(1):t(end),1:4);
    y=temp(:,4);
    data3 = temp(:,2:4);
    
    score=score_GT+t_bias(detected(ID,1),:)';
    sample_num=[sample_num
        length(t)];
    if length(score)==n
        L2_distance_preference(ID,1)=sqrt(sum((score_GT-score).*(score_GT-score)));
        score_user(ID,:)= score';
        
    end
end

Z=detected(:,2);
colormap(flipud(colormap));
figure(1);
scatter(L2_distance_preference,sample_num,50,Z,'filled');hold on;
colormap(flipud(colormap));
%%%%%position bias
L2_distance_preference=zeros(length(position_bias_user),0);
data=Pair_Compar;
detected=position_detected;
[compare,~]=size(data);
user = data(:,1);
y = data(:,4);
data = data(:,2:4);

score_GT=result.s_path(:,k);
[a_GT b_GT]=sort(score_GT,'descend');

list_GT=[a_GT b_GT];
sample_num=[];
for ID=1:length(position_bias_user)
    t=find(Pair_Compar(:,1)==detected(ID,1));
    temp=Pair_Compar(t(1):t(end),1:4);
    y=temp(:,4);
    data3 = temp(:,2:4);
    
    gamma2=gamma(detected(ID,1));
    gamma1=0;
    sample_num=[sample_num
        length(t)];
    L2_distance_preference(ID,1)=abs(gamma1-gamma2);
    
end
Z=detected(:,2);
colormap(flipud(colormap));

figure(2);
scatter(L2_distance_preference,sample_num,50,Z,'filled');hold on;
colormap(flipud(colormap));

