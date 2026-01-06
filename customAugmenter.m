function outputData = customAugmenter(inputData)

    imageBatch = inputData{:, 1};
    labelBatch = inputData{:, 2};

    augmentedImageCells = cell(numel(imageBatch), 1);

    for i = 1:length(imageBatch)
        image = imageBatch{i};

        augmentedImage = randomizeColors(image, [0.4, 1]);
        augmentedImage = randomizeBrightness(augmentedImage, [0.5, 1.5]);
        augmentedImage = addGaussianBlur(augmentedImage, [0.1, 1]);
        % augmentedImage = addGaussianNoise(augmentedImage, [0.01, 0.2], [0.001, 0.01]);  
        % augmentedImage = addSaltAndPepperNoise(augmentedImage, [0.0001, 0.01]);
        % augmentedImage = addJPEGCompression(augmentedImage, [70, 100]);

        augmentedImageCells{i} = augmentedImage;
    end

    outputData = table(augmentedImageCells, labelBatch);
end

function augmentedImage = addGaussianNoise(image, meanRange, varianceRange)
    % Add Gaussian noise to the image
    mean = randf(meanRange);
    variance = randf(varianceRange);
    augmentedImage = imnoise(image, 'gaussian', mean, variance);
end

function augmentedImage = addSaltAndPepperNoise(image, noiseRange)
    % Add salt and pepper noise to the image
    noise = randf(noiseRange);
    augmentedImage = imnoise(image, 'salt & pepper', noise);
end

function augmentedImage = addJPEGCompression(image, qualityRange)
    % Save and reload the image with JPEG compression to simulate artifacts
    tempFile = "temp_compressed_" + (posixtime(datetime("now")) * 1000) + ".jpg";
    quality = randi(qualityRange);
    imwrite(image, tempFile, 'jpg', Quality=quality);
    augmentedImage = imread(tempFile);
    delete(tempFile);
end

function augmentedImage = addGaussianBlur(image, sigmaRange)
    % Apply Gaussian blur to the image
    sigma = randf(sigmaRange);
    augmentedImage = imgaussfilt(image, sigma);
end

function augmentedImage = randomizeBrightness(image, range)
    brightnessFactor = randf(range);
    augmentedImage = im2double(image) * brightnessFactor;  
    augmentedImage = min(max(augmentedImage, 0), 1);
    augmentedImage = im2uint8(augmentedImage);
end

function augmentedImage = randomizeColors(image, range)
    rShift = randf(range);
    gShift = randf(range);
    bShift = randf(range);
    augmentedImage = im2double(image);
    augmentedImage(:,:,1) = augmentedImage(:,:,1) * rShift;
    augmentedImage(:,:,2) = augmentedImage(:,:,2) * gShift;
    augmentedImage(:,:,3) = augmentedImage(:,:,3) * bShift;
    augmentedImage = min(max(augmentedImage, 0), 1);
    augmentedImage = im2uint8(augmentedImage);
end

function value = randf(range)
    value = range(1) + (range(2) - range(1)) * rand;
end