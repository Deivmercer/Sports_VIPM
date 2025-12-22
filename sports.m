net = alexnet;
analyzeNetwork(net)
inputSize = net.Layers(1).InputSize;
labelsMatrix = readmatrix("sports_labels.csv", OutputType="string");

%% Loading the training set
trainingSet = labelsMatrix(labelsMatrix(:, 4) == "train", 1:2);
trainingLabels = categorical(trainingSet(:, 1));
trainingImages = replace(trainingSet(:, 2), "/", filesep);
trainingImages = string(pwd()) + filesep + trainingImages;

path_db = "./train/";
trainingDatastore = imageDatastore(path_db, IncludeSubfolders=true);

% Removing unlabled/invalid images
[~, trainingIndices] = ismember(trainingImages, trainingDatastore.Files);
trainingIndices = trainingIndices(trainingIndices > 0);
trainingLabels = trainingLabels(trainingIndices);

% Creating the new datastore
subsetDatastore = subset(trainingDatastore, trainingIndices);
subsetDatastore.Labels = trainingLabels;

% Creating the augmentedImageDatastore
imageAugmenter= imageDataAugmenter();
resizedTrainingDatastore = augmentedImageDatastore(inputSize, subsetDatastore, ColorPreprocessing="gray2rgb");

%% Loading the validation set
validationSet = labelsMatrix(labelsMatrix(:, 4) == "valid", 1:2);
validationLabels = categorical(validationSet(:, 1));
validationDatastore = imageDatastore("valid", IncludeSubfolders=true, Labels=validationLabels);
resizedValidationDatastore = augmentedImageDatastore(inputSize, validationDatastore, ColorPreprocessing="gray2rgb");

%% Loading the test set
testSet = labelsMatrix(labelsMatrix(:, 4) == "test", 1:2);
testLabels = categorical(testSet(:, 1));
testDatastore = imageDatastore("test", IncludeSubfolders=true, Labels=testLabels);
resizedTestDatastore = augmentedImageDatastore(inputSize, testDatastore, ColorPreprocessing="gray2rgb");

%% Training the network
layersTransfer = net.Layers(1:end-3);
layersTransfer = freezeWeights(layersTransfer);

numClasses = 100;
layers = [layersTransfer
          fullyConnectedLayer(numClasses, WeightLearnRateFactor=20, BiasLearnRateFactor=20);
          softmaxLayer
          classificationLayer];

options = trainingOptions("sgdm", ...
                          "MiniBatchSize", 10, ...
                          "MaxEpochs", 6, ...
                          "InitialLearnRate", 1e-4, ...
                          "Shuffle", "every-epoch", ...
                          "ValidationData", resizedValidationDatastore, ...
                          "ValidationFrequency", 3, ...
                          "Verbose", false, ...
                          "Plots", "training-progress");

model = trainNetwork(resizedTrainingDatastore, layers, options);

%% Evaluation
[predictedLabels, scores] = classify(model, resizedTestDatastore);

% Computing accuracy
numel(find(predictedLabels == testLabels)) / numel(predictedLabels)