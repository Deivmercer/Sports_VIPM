close all
clear
clc

net = alexnet;
sz = net.Layers(1).InputSize;
% analyzeNetwork(net)

%% ============================================================================
%% BLOCCO 1: PREPARAZIONE RETE
%% ============================================================================


layersTransfer_M1 = net.Layers(1:end - 8);
layersTransfer_M1 = freezeWeights(layersTransfer_M1);

layersTransfer_M2 = net.Layers(1:end - 8);
layersTransfer_M2 = freezeWeights(layersTransfer_M2);

numClasses = 100;

%% ============================================================================
%% BLOCCO 2: LOADING DATASET
%% ============================================================================
labelsMatrix = readmatrix("sports_labels.csv", OutputType="string");

trainingSet = labelsMatrix(labelsMatrix(:, 4) == "train", 1:2);
trainingLabels = categorical(trainingSet(:, 1));
trainingImages = replace(trainingSet(:, 2), "/", filesep);
trainingImages = string(pwd()) + filesep + trainingImages;
path_db = "./train/";
trainingDatastore = imageDatastore(path_db, IncludeSubfolders=true);

[~, trainingIndices] = ismember(trainingImages, trainingDatastore.Files);
trainingIndices = trainingIndices(trainingIndices > 0);
trainingLabels = trainingLabels(trainingIndices);

subsetDatastore = subset(trainingDatastore, trainingIndices);
subsetDatastore.Labels = trainingLabels;

validationSet = labelsMatrix(labelsMatrix(:, 4) == "valid", 1:2);
validationLabels = categorical(validationSet(:, 1));
validationDatastore = imageDatastore("valid", IncludeSubfolders=true, Labels=validationLabels);

testSet = labelsMatrix(labelsMatrix(:, 4) == "test", 1:2);
testLabels = categorical(testSet(:, 1));
testDatastore = imageDatastore("test", IncludeSubfolders=true, Labels=testLabels);

testDegradedSet = labelsMatrix(labelsMatrix(:, 4) == "test", 1:2);
testDegradedLabels = categorical(testDegradedSet(:, 1));
testDegradedDatastore = imageDatastore("test_degradato", IncludeSubfolders=true, Labels=testDegradedLabels);
testDegradedFiles = testDegradedDatastore.Files;
n_test_degraded = length(testDegradedFiles);

%% MODELLO 1

layers1 = [layersTransfer_M1
           net.Layers(end - 7:end - 4)
           fullyConnectedLayer(numClasses, "WeightLearnRateFactor", 20, "BiasLearnRateFactor", 20)
           softmaxLayer
           classificationLayer];

pixelRange = [-10 10];
imageAugmenterStandard = imageDataAugmenter( ...
    "RandXReflection", true, ...
    "RandXTranslation", pixelRange, ...
    "RandYTranslation", pixelRange, ...
    "RandRotation", [-20 20], ...
    "RandScale", [0.8 1.2]);

augimdsTrain1 = augmentedImageDatastore(sz(1:2), subsetDatastore, ...
    ColorPreprocessing="gray2rgb", DataAugmentation=imageAugmenterStandard);

augimdsVal = augmentedImageDatastore(sz(1:2), validationDatastore, ColorPreprocessing="gray2rgb");

options1 = trainingOptions("adam", ...
    "MiniBatchSize", 64, ...
    "MaxEpochs", 15, ...
    "InitialLearnRate", 1e-4, ...
    "LearnRateSchedule", 'piecewise', ...
    "LearnRateDropPeriod", 5, ...
    "LearnRateDropFactor", 0.1, ...
    "Shuffle", "every-epoch", ...
    "ValidationData", augimdsVal, ...
    "ValidationFrequency", 200, ...
    "Verbose", false, ...
    "Plots", "training-progress", ...
    "L2Regularization", 0.01);

model1 = trainNetwork(augimdsTrain1, layers1, options1);

augimdsTest = augmentedImageDatastore(sz(1:2), testDatastore, ColorPreprocessing="gray2rgb");
tic
[lab_pred1_clean, ~] = classify(model1, augimdsTest);
toc
acc1_clean = mean(lab_pred1_clean == testDatastore.Labels);
fprintf('Modello1 Test Clean: %.4f\n', acc1_clean);

%% MODELLO 2
display("Creating degraded training datastore...")

imageAugmenterAgg = imageDataAugmenter( ...
    "RandXReflection", true, ...
    "RandXTranslation", [-8 8], ...
    "RandYTranslation", [-8 8], ...
    "RandRotation", [-15 15], ...
    "RandScale", [0.85 1.15], ...
    "RandXShear", [-0.1 0.1]);

augimdsTrain2 = augmentedImageDatastore(sz(1:2), subsetDatastore, ...
    ColorPreprocessing="gray2rgb", DataAugmentation=imageAugmenterAgg);
augimdsTrain2.MiniBatchSize = 64;

% Appling the custom transformation to each image in the augmentedImageDatastore
customAugmentedDatastore = transform(augimdsTrain2, @customAugmenter);

layers2 = [layersTransfer_M2
           net.Layers(end - 7:end - 4)
           fullyConnectedLayer(numClasses, "WeightLearnRateFactor", 20, "BiasLearnRateFactor", 20)
           softmaxLayer
           classificationLayer];

options2 = trainingOptions("adam", ...
    "MiniBatchSize", 64, ...
    "MaxEpochs", 15, ...
    "InitialLearnRate", 1e-4, ...
    "LearnRateSchedule", 'piecewise', ...
    "LearnRateDropPeriod", 5, ...
    "LearnRateDropFactor", 0.1, ...
    "Shuffle", "every-epoch", ...
    "ValidationData", augimdsVal, ...
    "ValidationFrequency", 200, ...
    "Verbose", false, ...
    "Plots", "training-progress", ...
    "L2Regularization", 0.01);


model2 = trainNetwork(customAugmentedDatastore, layers2, options2);

tic
[lab_pred2_clean, ~] = classify(model2, augimdsTest);
toc
acc2_clean = mean(lab_pred2_clean == testDatastore.Labels);

%% TEST DEGRADATO + ENHANCEMENT

function enhanced = unsupervisedEnhance(img)
    if ismatrix(img)
        img = cat(3, img, img, img);
    elseif size(img, 3) == 1
        img = repmat(img, [1, 1, 3]);
    end
    
    img_double = im2double(img);
    enhanced = imbilatfilt(img_double, 0.1, 0.1);
    gamma = 1.2;
    enhanced = enhanced .^ (1/gamma);
    enhanced = uint8(enhanced * 255);
end

augimdsDegraded = augmentedImageDatastore(sz(1:2), testDegradedDatastore, ColorPreprocessing="gray2rgb");

[predRaw1, ~] = classify(model1, augimdsDegraded);
[predRaw2, ~] = classify(model2, augimdsDegraded);
accRaw1 = mean(predRaw1 == testDegradedDatastore.Labels);
accRaw2 = mean(predRaw2 == testDegradedDatastore.Labels);

pred1_enh = categorical(repmat("", n_test_degraded, 1));
pred2_enh = categorical(repmat("", n_test_degraded, 1));
for i = 1:n_test_degraded
    img = imread(testDegradedFiles{i});
    enhanced = unsupervisedEnhance(img);
    enhanced_resized = imresize(enhanced, sz(1:2));
    
    if ismatrix(enhanced_resized)
        enhanced_resized = cat(3, enhanced_resized, enhanced_resized, enhanced_resized);
    end
    
    [pred1_enh(i), ~] = classify(model1, enhanced_resized);
    [pred2_enh(i), ~] = classify(model2, enhanced_resized);
    if mod(i, 100) == 0, fprintf('%d/%d ', i, n_test_degraded); end
end
acc1_enh = mean(pred1_enh == testDegradedDatastore.Labels);
acc2_enh = mean(pred2_enh == testDegradedDatastore.Labels);


%% TABELLA FINALE
fprintf('\n╔══════════════════════════════════════════════════════════════════════════════╗\n');
fprintf('║                    CONFRONTO TASK 1 - TASK 2 COMPLETO                        ║\n');
fprintf('╚══════════════════════════════════════════════════════════════════════════════╝\n');
fprintf('┌────────────────────┬────────────┬──────────────┬──────────────────┐\n');
fprintf('│ Modello            │ Test Clean │ Degr RAW    │ Degr+Enhanced     │\n');
fprintf('├────────────────────┼────────────┼──────────────┼──────────────────┤\n');
fprintf('│ Mod1 (Standard)    │   %.4f    │   %.4f     │     %.4f       │\n', acc1_clean, accRaw1, acc1_enh);
fprintf('│ Mod2 (Degr+Agg)    │   %.4f    │   %.4f     │     %.4f       │\n', acc2_clean, accRaw2, acc2_enh);
fprintf('└────────────────────┴────────────┴──────────────┴──────────────────┘\n');

%% Freeze weights function
function layers = freezeWeights(layers)
    for ii = 1:size(layers,1)
        props = properties(layers(ii));
        for p = 1:numel(props)
            propName = props{p};
            if ~isempty(regexp(propName, 'LearnRateFactor$', 'once'))
                layers(ii).(propName) = 0;
            end
        end
    end
end
