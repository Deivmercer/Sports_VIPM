net = alexnet;
analyzeNetwork(net)
inputSize = net.Layers(1).InputSize;
labelsMatrix = readmatrix("sports_labels.csv", OutputType="string");

%% Loading the training set
trainingSet = labelsMatrix(labelsMatrix(:, 4) == "valid", 1:2);
trainingLabels = categorical(trainingSet(:, 1));
trainingImages = replace(trainingSet(:, 2), "/", filesep);
trainingImages = string(pwd()) + filesep + trainingImages;

path_db = "./valid/";
trainingDatastore = imageDatastore(path_db, IncludeSubfolders=true);

% Removing unlabled/invalid images
[~, trainingIndices] = ismember(trainingImages, trainingDatastore.Files);
trainingIndices = trainingIndices(trainingIndices > 0);
trainingLabels = trainingLabels(trainingIndices);

% Creating the new datastore
subsetDatastore = subset(trainingDatastore, trainingIndices);
subsetDatastore.Labels = trainingLabels;

% Splitting into training and validation datasets
[trainingSplitDatastore, validationDatastore] = splitEachLabel(subsetDatastore, 0.8,'randomized');

% Creating the augmentedImageDatastore
imageAugmenter = imageDataAugmenter(...
    'RandXReflection', true, ...
    'RandXTranslation', [-20, 20], ...
    'RandYTranslation', [-20, 20], ...
    'RandRotation', [-5, 5]);

resizedTrainingDatastore = augmentedImageDatastore(inputSize, trainingSplitDatastore, ColorPreprocessing="gray2rgb", ...
    DataAugmentation=imageAugmenter);
resizedValidationDatastore = augmentedImageDatastore(inputSize, trainingSplitDatastore, ColorPreprocessing="gray2rgb");

%% Loading the test set
testSet = labelsMatrix(labelsMatrix(:, 4) == "test", 1:2);
testLabels = categorical(testSet(:, 1));
testDatastore = imageDatastore("test", IncludeSubfolders=true, Labels=testLabels);
resizedTestDatastore = augmentedImageDatastore(inputSize, testDatastore, ColorPreprocessing="gray2rgb");

%% Training the network
layersTransfer = net.Layers(1:end - 3);
layersTransfer = freezeWeights(layersTransfer);

numClasses = 100;
layers = [layersTransfer
          fullyConnectedLayer(numClasses, WeightLearnRateFactor=20, BiasLearnRateFactor=20);
          softmaxLayer
          classificationLayer];

options = trainingOptions("rmsprop", ...
                          "MiniBatchSize", 32, ...
                          "MaxEpochs", 10, ...
                          "InitialLearnRate", 1e-4, ...
                          "LearnRateSchedule", 'piecewise', ...
                          "LearnRateDropPeriod", 5, ...
                          "LearnRateDropFactor", 0.5, ...
                          "Shuffle", "every-epoch", ...
                          "ValidationData", resizedValidationDatastore, ...
                          "ValidationFrequency", 5, ...
                          "ValidationPatience", 5, ...
                          "Verbose", false, ...
                          "Plots", "training-progress", ...
                          'L2Regularization', 0.01);

model = trainNetwork(resizedTrainingDatastore, layers, options);

%% Evaluation
[predictedLabels, scores] = classify(model, resizedTestDatastore);
CM = confusionmat(testLabels, predictedLabels);
CM = CM ./ repmat(sum(CM, 2), 1, size(CM, 2));
accuracy = mean(diag(CM));
figure(1), clf
imagesc(CM), colorbar
title(["Accuracy: " num2str(accuracy)])

% Computing accuracy
numel(find(predictedLabels == testLabels)) / numel(predictedLabels)
