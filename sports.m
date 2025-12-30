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
                          "MiniBatchSize", 32, ... 
                          "MaxEpochs", 10, ...
                          "InitialLearnRate", 1e-4, ...
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

% Computing accuracy
numel(find(predictedLabels == testLabels)) / numel(predictedLabels)


%%==============================================================
%% TASK 2 OTTIMIZZATO - ENHANCEMENT SOFT + FINETUNING FORTE
%%==============================================================
display("=== TASK 2 OTTIMIZZATO: Test set degradato ===")

%% 2.1 Caricamento test set degradato
testDegradedSet = labelsMatrix(labelsMatrix(:, 4) == "test", 1:2);
testDegradedLabels = categorical(testDegradedSet(:, 1));
testDegradedDatastore = imageDatastore("test_degradato", IncludeSubfolders=true, Labels=testDegradedLabels);
resizedTestDegraded = augmentedImageDatastore(inputSize, testDegradedDatastore, ColorPreprocessing="gray2rgb");

%% Test RAW PRIMA (baseline)
display("Test degraded RAW (modello originale)");
[predRaw, ~] = classify(model, resizedTestDegraded);
accRaw = mean(predRaw == testDegradedLabels);
fprintf('Accuracy test degraded RAW: %.3f\n', accRaw);

%% APPROCCIO 1: ENHANCEMENT SOFT (non distruttivo)
display("Approccio 1: Enhancement SOFT unsupervised")

function enhancedImg = enhanceSingleImageSoft(imgPath, targetSize)
    img = imread(imgPath);
    if size(img, 3) == 1
        img = cat(3, img, img, img);
    end
    img = imresize(img, targetSize(1:2));
    img = im2double(img);
    
    % ENHANCEMENT MOLTO LEGGERO (non distruttivo)
    % 1. Contraste enhancement (CLAHE)
    lab = rgb2lab(img);
    lab(:,:,1) = adapthisteq(lab(:,:,1), 'ClipLimit', 0.03, 'Distribution', 'rayleigh');
    img_enhanced = lab2rgb(lab);
    
    % 2. Denoising MOLTO leggero (Gaussian 3x3)
    h = fspecial('gaussian', [3 3], 0.8);
    denoised = imfilter(img_enhanced, h, 'replicate');
    
    % 3. Wiener con PSF MOLTO piccola
    psf = fspecial('gaussian', [5 5], 0.8);
    noiseVar = 0.005;  % MOLTO basso
    enhancedImg = deconvwnr(denoised, psf, noiseVar);
    enhancedImg = mat2gray(enhancedImg);
end

% Processa con enhancement soft
display("Applicazione enhancement SOFT...");
numImages = numel(testDegradedDatastore.Files);
enhancedImages = cell(numImages, 1);
for i = 1:numImages
    if mod(i, 100) == 0
        fprintf('Processate %d/500 immagini...\n', i);
    end
    imgPath = testDegradedDatastore.Files{i};
    enhancedImages{i} = enhanceSingleImageSoft(imgPath, inputSize);
end

enhancedArray4D = cat(4, enhancedImages{:});
[predEnhanced, ~] = classify(model, enhancedArray4D);
accEnhanced = mean(predEnhanced == testDegradedLabels);
fprintf('Accuracy test degraded ENHANCED SOFT: %.3f\n', accEnhanced);

%% APPROCCIO 2: FINETUNING FORTE (migliore)
display("Approccio 2: Finetuning FORTE per degradazioni");

% Data augmentation PIÙ aggressiva per degradazioni
degradationAugmenter = imageDataAugmenter( ...
    'RandRotation', [-15, 15], ...
    'RandXReflection', true, ...
    'RandXTranslation', [-8, 8], ...
    'RandYTranslation', [-8, 8], ...
    'RandScale', [0.85, 1.15], ...
    'RandXShear', [-0.1, 0.1], ...
    'RandYShear', [-0.1, 0.1]);

degradedTrainingDatastore = augmentedImageDatastore(inputSize, subsetDatastore, ...
    ColorPreprocessing="gray2rgb", DataAugmentation=degradationAugmenter);

% FINETUNING PIÙ PROFONDO: sblocca ANCHE conv layers finali
layersFinetune = net.Layers(1:end-8);  % Più layer congelati
layersFinetune = freezeWeights(layersFinetune);  % Solo ULTIMI trainable

finetuneLayers = [layersFinetune
                  net.Layers(end-7:end-4)  % fc6 + fc7 trainable
                  fullyConnectedLayer(numClasses, WeightLearnRateFactor=5, BiasLearnRateFactor=5)
                  softmaxLayer
                  classificationLayer];

optionsFinetune = trainingOptions("sgdm", ...
    "MiniBatchSize", 32, ...      % Batch originale
    "MaxEpochs", 8, ...           % Più epoche
    "InitialLearnRate", 1e-4, ... % LR più alto
    "LearnRateSchedule", "piecewise", ...
    "LearnRateDropFactor", 0.5, ...
    "LearnRateDropPeriod", 4, ...
    "Shuffle", "every-epoch", ...
    "ValidationData", resizedValidationDatastore, ...
    "ValidationFrequency", 150, ...
    "Verbose", false, ...
    "Plots", "training-progress");

modelFinetuned = trainNetwork(degradedTrainingDatastore, finetuneLayers, optionsFinetune);

[predFinetuned, ~] = classify(modelFinetuned, resizedTestDegraded);
accFinetuned = mean(predFinetuned == testDegradedLabels);
fprintf('Accuracy test degraded FINETUNED FORTE: %.3f\n', accFinetuned);

%% CONFRONTO OTTIMIZZATO
fprintf('\n=== CONFRONTO TASK 2 OTTIMIZZATO ===\n');
fprintf('Task 1 (test normale):           %.3f\n', mean(predictedLabels == testLabels));
fprintf('Test degraded RAW:               %.3f\n', accRaw);
fprintf('Test degraded ENHANCED SOFT:     %.3f\n', accEnhanced);
fprintf('Test degraded FINETUNED FORTE:   %.3f\n', accFinetuned);
