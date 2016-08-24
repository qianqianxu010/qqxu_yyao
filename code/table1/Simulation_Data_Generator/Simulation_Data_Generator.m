n_item = 30;
n_user = 500;
p1 = 0.4; %percentage of user with nonzero gamma
p2 = 0.4; %percentage of user with nonzero delta
s1 = 2; %standard deviation of nonzero gamma
s2 = 3; %maximal standard deviation of nonzero delta
%sigma = 0.3; %standard deviation of gaussian noise
model = 2;

theta = randn(n_item,1);
theta = theta - mean(theta);
sample_per_user = randi([100,500],n_user,1);
gamma = zeros(n_user,1);
delta = zeros(n_item,n_user);

ind1 = (rand(n_user,1)<p1);
gamma(ind1) = randn(sum(ind1),1)*s1;
ind2 = (rand(n_user,1)<p2);
ss2 = rand(1,sum(ind2))*s2;
delta(:,ind2) = randn(n_item,sum(ind2)).*(ones(n_item,1)*ss2);
delta = delta-ones(n_item,1)*mean(delta);

m = sum(sample_per_user);
u = [];
for i=1:n_user
    u = [u;ones(sample_per_user(i),1)*i];
end
i = randi(n_item,m,1);
j = randi(n_item-1,m,1);
j = j + (j>i);
%epsilon = randn(m,1)*sigma;
res = gamma(u) + theta(i) - theta(j) + delta(i+(u-1)*n_item) - delta(j+(u-1)*n_item);% + epsilon;

switch model
    case 1 %% uniform model
        pij = (res/max(abs(res))+1)/2;
    case 2 %% BT
        pij = 1./(1+exp(-res));
    case 3 %% Thurstone
        pij = normcdf(res);
    case 4 %% Angular
        pij = (sin(pi/2*res/max(abs(res)))+1)/2;
end
y = (rand(m,1)<pij)*2-1;
data = [u,i,j,y];
Pair_Compar=data;

  
           