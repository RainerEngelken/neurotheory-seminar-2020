% Using Bayesian Linear Regression to model motorcycle accident data.
function bayesian_linear_regression()
% --- Set up some parameters -----%
basis = 2;      % Which basis to use (1:polynomial, 2:RBF)
m     = 100;    % number of basis functions to use (except for the constant function)
a     = 0.0025;% precision (1/variance) of Y_i given w
b     = 10^-2; % prior precision (1/variance)

% Load the dataset into Matlab
all = load('motor.mat');
x = all.Xtrain;
y = all.Ytrain;
x_test = all.Xtest;
y_test = all.Ytest;
x_excluded = 57.6;
y_excluded = 10.7;

% number of data points
n = size(x,1);
n_test = size(x_test,1);

% Rescale x-values to the interval [-1,1]
rescale = @(values) (values-30)/30;
x = rescale(x);
x_test = rescale(x_test);
x_excluded = rescale(x_excluded);

% plot data
figure(1); clf; hold on
plot(x,y,'bo')
plot(x_test,y_test,'ro')
title(gca,['RBF: b (precision) = ' num2str(b),' a (precision) = ' num2str(a)])
set(gca,'FontSize',16);


% polynomial basis functions
polynomials = @(x_i,powers) x_i.^powers;

% form the design matrix using the polynomial basis functions
function A = design_matrix_polynomial(x,powers)
    m_0 = length(powers);
    n_0 = length(x);
    A = zeros(n_0,m_0);
    for i = 1:n_0
        A(i,:) = polynomials(x(i),powers);
    end
end


% radial basis functions
RBF = @(x_i,means,sigma) exp(-(x_i-means).^2/(2*sigma^2));

% form the design matrix using the radial basis functions
function A = design_matrix_radial(x,means,sigma)
    m_0 = length(means);
    n_0 = length(x);
    A = zeros(n_0,m_0);
    for i = 1:n_0
        A(i,:) = RBF(x(i),means,sigma);
    end
    A = [ones(n_0,1) A];  % insert a constant column
end

% x points for plotting
x_plot = linspace(-1,1,200)';

if basis==1
    % polynomial basis
    powers = 0:m;
    % form the design matrix A
    A = design_matrix_polynomial(x,powers);
    A_test = design_matrix_polynomial(x_test,powers);
    A_plot = design_matrix_polynomial(x_plot,powers);
else
    % RBF basis
    means = linspace(-1,1,m);
    sigma = 3*(means(2)-means(1));
    % form the design matrix A
    A = design_matrix_radial(x,means,sigma);
    A_test = design_matrix_radial(x_test,means,sigma);
    A_plot = design_matrix_radial(x_plot,means,sigma);
end


% compute the posterior distribution of w (given the data)

% posterior covariance
I = eye(size(A,2));
lamb = a*(A'*A) + b*I;
C = pinv(lamb); 

% posterior mean
mu = a*C*A'*y; 

% mean of Y (for each point in x_plot) given the data 
u = A_plot*mu; 

% the variance of Y (for each point in x_plot) given the data
variance = (1/a) + A_plot*C*A_plot' ; 

% plot the mean prediction function f(x) given the data
plot(x_plot',u,'k')
% Plot 2x error bars (+/- 2*standard deviation) for Y given the data
standard_deviation = sqrt(variance);  
plot(x_plot',u+2*standard_deviation,'g','LineWidth',1) 
plot(x_plot',u-2*standard_deviation,'g','LineWidth',1) 
axis([-1,1,-200,200])


% Compute model evidence
enabled = false;
if enabled
    b_values = logspace(-14,2,200);
    n_values = length(b_values);
    L = zeros(n_values,1);
    test_error = zeros(n_values,1);
    for i = 1:n_values
        b = b_values(i);
        
        % posterior covariance
      
        I = eye(size(A,2));
        lamb = a*(A'*A) + b*I;
        C = pinv(lamb);

        % posterior mean
        mu = a*C*A'*y; 
        
        % compute the marginal log-likelihood
        L(i) = (1/2)*((m+1)*log(b) + n*log(a/(2*pi)) + sum(log(eig(C)))...
              - b*(mu'*mu) - a*sum((y-A*mu).^2));
        
        % compute the test error
        test_error(i) = sqrt(mean((y_test-A_test*mu).^2));
    end

    figure(2); clf; hold on
    plot(log10(b_values),L,'b.')

    figure(3); clf; hold on
    plot(log10(b_values),log10(test_error),'r.')
end
    

end










