close all
clear
clc

%% Carica AlexNet
net = alexnet;
sz = net.Layers(1).InputSize;
layers = {'pool1', 'conv4', 'relu7'};

%% Carica dataset Sports
labelsMatrix = readmatrix("sports_labels.csv", OutputType="string");

% VALIDATION -> TRAINING
val_set    = labelsMatrix(labelsMatrix(:,4)=="valid", 1:2);
val_labels = categorical(val_set(:,1));
val_files  = string(pwd()) + filesep + replace(val_set(:,2), "/", filesep);
val_ds     = imageDatastore("./valid/", 'IncludeSubfolders', true);
val_idx    = ismember(val_ds.Files, val_files);
val_ds     = subset(val_ds, val_idx);
val_ds.Labels = val_labels;
Ntr = numel(val_ds.Files);

% Test
te_set    = labelsMatrix(labelsMatrix(:,4)=="test", 1:2);
te_labels = categorical(te_set(:,1));
te_files  = string(pwd()) + filesep + replace(te_set(:,2), "/", filesep);
te_ds     = imageDatastore("./test/", 'IncludeSubfolders', true);
te_idx    = ismember(te_ds.Files, te_files);
te_ds     = subset(te_ds, te_idx);
te_ds.Labels = te_labels;
Nte = numel(te_ds.Files);

%% Estrazione features per ogni layer
tic
for ii = 1:length(layers)
    feat_tr = [];
    labels_tr = [];

    disp("estrazione features layer: " + layers{ii} + " (TRAINING)")
    tic
    for i = 1:Ntr
        im = double(readimage(val_ds, i));
        % AlexNet richiede RGB (3 canali)
        if size(im,3)==1
            im = repmat(im, 1, 1, 3);  % Converti grayscale -> RGB
        end
        im = imresize(im, sz(1:2));
        feat_temp = activations(net, im, layers{ii}, 'OutputAs', 'row');
        feat_tr = [feat_tr; feat_temp];
        labels_tr = [labels_tr; double(val_ds.Labels(i))];
    end
    toc

    %% Estrazione features TEST
    feat_te = [];
    labels_te = [];

    disp("estrazione features layer: " + layers{ii} + " (TEST)")
    tic
    for i = 1:Nte
        im = double(readimage(te_ds, i));
        % AlexNet richiede RGB (3 canali)
        if size(im,3)==1
            im = repmat(im, 1, 1, 3);  % Converti grayscale -> RGB
        end
        im = imresize(im, sz(1:2));
        feat_temp = activations(net, im, layers{ii}, 'OutputAs', 'row');
        feat_te = [feat_te; feat_temp];
        labels_te = [labels_te; double(te_ds.Labels(i))];
    end
    toc

    %% Normalizzazione (doppio test: senza e con)
    normalize = false;

    for jj = 1:2
        if (normalize)
            feat_tr_norm = feat_tr ./ sqrt(sum(feat_tr.^2, 2));
            feat_te_norm = feat_te ./ sqrt(sum(feat_te.^2, 2));
        else
            feat_tr_norm = feat_tr;
            feat_te_norm = feat_te;
        end

        %% 1-NN
        disp("classificatore 1-NN")
        D = pdist2(feat_te_norm, feat_tr_norm);
        [~, idx_pred_te] = min(D, [], 2);
        lab_pred_te = labels_tr(idx_pred_te);
        acc_1nn = numel(find(lab_pred_te == labels_te)) / numel(labels_te);

        %% K-NN
        disp("classificatore K-NN")
        Mdl_knn = fitcknn(feat_tr_norm, labels_tr, 'NumNeighbors', 5);
        pred_knn = predict(Mdl_knn, feat_te_norm);
        acc_knn = mean(pred_knn == labels_te);

        %% SVM
        disp("classificatore SVM")
        Mdl_svm = fitcecoc(feat_tr_norm, labels_tr);
        pred_svm = predict(Mdl_svm, feat_te_norm);
        acc_svm = mean(pred_svm == labels_te);

        %% Accumula risultati
        results = [{layers{ii}, normalize, '1-NN', acc_1nn};
            {layers{ii}, normalize, 'K-NN', acc_knn};
            {layers{ii}, normalize, 'SVM',  acc_svm}]

        normalize = true;
    end
    normalize = false;
end
toc