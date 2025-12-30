clear
close all
clc

net = alexnet;
analyzeNetwork(net)
inputSize = net.Layers(1).InputSize;
labelsMatrix = readmatrix("sports_labels.csv", OutputType="string");

%% Loading the training set
display("Loading the training set")
trainingSet = labelsMatrix(labelsMatrix(:, 4) == "train", 1:2);
trainingLabels = categorical(trainingSet(:, 1));
trainingImages = replace(trainingSet(:, 2), "/", filesep);
trainingImages = string(pwd()) + filesep + trainingImages;

path_db = "./train/";
trainingDatastore = imageDatastore(path_db, IncludeSubfolders=true);

% Removing unlabled/invalid images
display("Removing unlabled/invalid images")

[~, trainingIndices] = ismember(trainingImages, trainingDatastore.Files);
trainingIndices = trainingIndices(trainingIndices > 0);
trainingLabels = trainingLabels(trainingIndices);

% Creating the new datastore
display("Creating the new datastore")
subsetDatastore = subset(trainingDatastore, trainingIndices);
subsetDatastore.Labels = trainingLabels;

% Creating the augmentedImageDatastore
display("Creating the augmentedImageDatastore")

imageAugmenter = imageDataAugmenter( ...
    'RandRotation', [-20, 20], ...
    'RandXReflection', true, ...
    'RandXTranslation', [-10, 10], ...
    'RandYTranslation', [-10, 10], ...
    'RandScale', [0.8, 1.2]);

resizedTrainingDatastore = augmentedImageDatastore(inputSize, subsetDatastore, ColorPreprocessing="gray2rgb");

%% Loading the validation set
display("Loading the validation set")
validationSet = labelsMatrix(labelsMatrix(:, 4) == "valid", 1:2);
validationLabels = categorical(validationSet(:, 1));
validationDatastore = imageDatastore("valid", IncludeSubfolders=true, Labels=validationLabels);
resizedValidationDatastore = augmentedImageDatastore(inputSize, validationDatastore, ColorPreprocessing="gray2rgb");

%% Loading the test set
display("Loading the test set")
testSet = labelsMatrix(labelsMatrix(:, 4) == "test", 1:2);
testLabels = categorical(testSet(:, 1));
testDatastore = imageDatastore("test", IncludeSubfolders=true, Labels=testLabels);
resizedTestDatastore = augmentedImageDatastore(inputSize, testDatastore, ColorPreprocessing="gray2rgb");

%% Training the network
display("Training the network")
layersTransfer = net.Layers(1:end-3);
layersTransfer = freezeWeights(layersTransfer);

numClasses = 100;
layers = [layersTransfer
          fullyConnectedLayer(numClasses, WeightLearnRateFactor=20, BiasLearnRateFactor=20);
          softmaxLayer
          classificationLayer];

options = trainingOptions("sgdm", ...
                          "MiniBatchSize", 64, ...
                          "MaxEpochs", 15, ...
                          "InitialLearnRate", 1e-4, ...
                          "LearnRateSchedule", 'piecewise', ...
                          "LearnRateDropPeriod", 5, ...
                          "LearnRateDropFactor", 0.1, ...
                          "Shuffle", "every-epoch", ...
                          "ValidationData", resizedValidationDatastore, ...
                          "ValidationFrequency", 200, ...    
                          "Verbose", false, ...
                          "Plots", "training-progress");
                         %"ExecutionEnvironment", "gpu", ...
% MiniBatchSize: possibile alzare 64 o 128 in base all'HW. più alto valore più dati alla scheda video
% ValidationFrequency: ogni X (3 prima) mini-batch, MATLAB interrompe il training per testare l'intera rete sul validation set. 
% Se hai migliaia di immagini, con un batch size di 10, fai centinaia di validazioni per epoca. 
% Consiglio: Imposta la frequenza in modo che avvenga una volta sola (o al massimo due) per epoca.
% Calcolo: Se hai ad esempio 2000 immagini e un batch size di 64, un'epoca dura circa 2000 / 64 circa iterazioni. Imposta ValidationFrequency a 30.
                        

model = trainNetwork(resizedTrainingDatastore, layers, options);

%% Evaluation
display("Evaluation")
[predictedLabels, scores] = classify(model, resizedTestDatastore);
CM = confusionmat(testLabels, predictedLabels);
CM = CM ./ repmat(sum(CM, 2), 1, size(CM, 2));
accuracy = mean(diag(CM));
figure(1), clf
imagesc(CM), colorbar
title(["Accuracy: " num2str(accuracy)])

% Computing accuracy
numel(find(predictedLabels == testLabels)) / numel(predictedLabels)

%% Degraded test set classification
path_degraded = "./test_degradato/";
degradedDatastore = imageDatastore(path_degraded, IncludeSubfolders=true, Labels=testLabels);
resizedDegradedDatastore = augmentedImageDatastore(inputSize, degradedDatastore, ColorPreprocessing="gray2rgb");

[predictedLabels, scores] = classify(model, resizedDegradedDatastore);
CM = confusionmat(testLabels, predictedLabels);
CM = CM ./ repmat(sum(CM, 2), 1, size(CM, 2));
accuracy = mean(diag(CM));
figure(1), clf
imagesc(CM), colorbar
title(["Accuracy: " num2str(accuracy)])

numel(find(predictedLabels == testLabels)) / numel(predictedLabels)