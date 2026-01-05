function outputData = customAugmenter(inputData)

    imageBatch = inputData{:, 1};
    labelBatch = inputData{:, 2};

    display(size(imageBatch))
    display(size(imageBatch{1}))
    display(size(labelBatch))

    % figure; montage(imageBatch);

    augmentedImageCells = cell(numel(imageBatch), 1);

    for i = 1:length(imageBatch)
        image = imageBatch{i};

        augmentedImage = addGaussianNoise(image);  
        augmentedImage = addSaltAndPepperNoise(augmentedImage);
        augmentedImage = addJPEGCompression(augmentedImage);

        augmentedImageCells{i} = augmentedImage;
    end

    display(size(augmentedImageCells))

    % figure; montage(augmentedImageBatch);

    outputData = table(augmentedImageCells, labelBatch);

    display(size(outputData))
end

function augmentedImage = addGaussianNoise(image)
    % Add Gaussian noise to the image
    augmentedImage = imnoise(image, 'gaussian', 0, 0.02);  % Mean = 0, Variance = 0.02
end

function augmentedImage = addSaltAndPepperNoise(image)
    % Add salt and pepper noise to the image
    augmentedImage = imnoise(image, 'salt & pepper', 0.02);  % 2% noise
end

function augmentedImage = addJPEGCompression(image)
    % Save and reload the image with JPEG compression to simulate artifacts
    tempFile = "temp_compressed_" + (posixtime(datetime("now")) * 1000) + ".jpg";
    imwrite(image, tempFile, 'jpg', 'Quality', 30);  % Set compression quality (lower is more compressed)
    augmentedImage = imread(tempFile);
    delete(tempFile);  % Clean up the temporary file
end