
V2 = xlsread('dados atualizados.xlsx',4,'C2:C163');
V3 = xlsread('dados atualizados.xlsx',4,'D2:D163');
V4 = xlsread('dados atualizados.xlsx',4,'E2:E163');
V6 = xlsread('dados atualizados.xlsx',4,'G2:G163');
V15 = xlsread('dados atualizados.xlsx',1,'P2:P163');

entrada = V4;
saida = V15;
n=1;


x = entrada';
t = saida';

% Choose a Training Function
trainFcn = 'trainscg';  % Scaled conjugate gradient backpropagation.

% Set the number of hidden layers
num_hidden_layers = 5; % Modify this value to change the number of hidden layers

% Set the learning rate (adjust the value as needed)
learnRate = 0.05;  % Learning rate (e.g., 0.01)

% Set the momentum (adjust the value as needed)
momentum = 0.5;   % Momentum coefficient (e.g., 0.9)

num_neuronios = 5;


% Set the number of neurons in each hidden layer
hiddenLayerSize = 10 * ones(1, num_hidden_layers);

% Setup Division of Data for Training, Validation, Testing
trainRatio = 70/100;
valRatio = 15/100;
testRatio = 15/100;

% Set showWindow to false to suppress the training window
net.trainParam.showWindow = false;

% Number of repetitions for training
num_repetitions = 1;

% Initialize arrays to store results
accuracy_results = zeros(num_repetitions, 1);
percentErrors_results = zeros(num_repetitions, 1);
performance_results = zeros(num_repetitions, 1);

    fileID = fopen('resultadosfase3C.txt', 'a');
    fprintf(fileID, '\n ----------------------------------------------------------------\nModelo 3\n');
    fprintf(fileID, '\nNúmero de camadas: %d\n', num_hidden_layers);
    fprintf(fileID, '\nNúmero de neurônios por camada: %d\n', num_neuronios);
    fprintf(fileID, 'Taxa de aprendizado: %.2f\n', learnRate);
    fprintf(fileID, 'Momentum: %.2f\n', momentum);
    fclose(fileID);

% Training loop
%for n = 1:num_repetitions
    % Create a Pattern Recognition Network
    net = patternnet([hiddenLayerSize num_neuronios], trainFcn);
    
    % Set learning rate and momentum
    net.trainParam.lr = learnRate;
    net.trainParam.mc = momentum;
    
    % Setup Division of Data for Training, Validation, Testing
    net.divideParam.trainRatio = trainRatio;
    net.divideParam.valRatio = valRatio;
    net.divideParam.testRatio = testRatio;

    % Train the Network
    [net, tr] = train(net, x, t);

    % Test the Network
    y = net(x);
    tind = vec2ind(t);
    yind = vec2ind(y);
    accuracy = sum(tind == yind) / numel(tind) * 100;
    percentErrors = sum(tind ~= yind) / numel(tind) * 100;
    performance = perform(net, t, y);

    figure;
plotconfusion(t, y);

    % Store the results in arrays
    accuracy_results(n) = accuracy;
    percentErrors_results(n) = percentErrors;
    performance_results(n) = performance;
    
    % Save performance to the text file
    fileID = fopen('resultadosfase3C.txt', 'a');
    fprintf(fileID, 'Repetição %d: Performance: %.4f\n', n, performance);
    fclose(fileID);
%end

% Calculate mean accuracy, mean percentErrors, and mean performance
mean_accuracy = mean(accuracy_results);
mean_percentErrors = mean(percentErrors_results);
mean_performance = mean(performance_results);

% Save overall results to the text file
fileID = fopen('resultadosfase3C.txt', 'a');
fprintf(fileID, '\nMédia do percentual de acertos: %.2f%%\n', mean_accuracy);
fprintf(fileID, 'Média do percentual de erros: %.2f%%\n', mean_percentErrors);
fprintf(fileID, 'Média da performance: %.4f\n', mean_performance);
fclose(fileID);

disp('Resultados salvos em "resultadosfase3C.txt".');