clear all

SAMPLE_PERCENTAGE = 10;

%% Load video
video = VideoReader("test_videos\7_Xu5eMPIQCCY.webm");
if video.Duration == 0 || hasFrame(video) == false
    error("Invalid video.")
end

video.CurrentTime = 0;

label = split(video.Name, '_');
label = label(1);
label = str2num(label{1});

%% Sampling frames
totalFrames = floor(video.Duration * video.FrameRate);
framesToSample = floor(totalFrames * SAMPLE_PERCENTAGE / 100);
stepSize = floor(totalFrames / framesToSample);

[~, tmpFolderName, ~] = fileparts(video.Name);
tmpFolderName = "test_videos\" + tmpFolderName;
mkdir(tmpFolderName);
tmpFileNames = cell(1, numel(framesToSample));

for i = 1:framesToSample
    video.CurrentTime = (i - 1) * stepSize / video.FrameRate;
    frame = readFrame(video);

    tmpFileName = tmpFolderName + filesep + "frame_" + i + ".png";
        
    imwrite(frame, tmpFileName);
    tmpFileNames{i} = tmpFileName;
end

%% Creating augmented image datastore
labels = label * ones(framesToSample, 1);
labels = categorical(labels);
datastore = imageDatastore(tmpFolderName, Labels=labels);
resizedDatastore = augmentedImageDatastore([227, 227, 3], datastore, ColorPreprocessing="gray2rgb");

%% Loading model
load 'modello task1 91.mat'

%% Classify frames
[predictedLabels, scores] = classify(model, resizedDatastore);

% Computing accuracy
numel(find(predictedLabels == labels)) / numel(predictedLabels)

%% Cleaning up temporary folder
for i = 1:numel(tmpFileNames)
    delete(tmpFileNames{i});
end
rmdir(tmpFolderName);
