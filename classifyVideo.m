clear all

%% Sampling videos
SAMPLE_PERCENTAGE = 10;
files = dir("test_videos");
frames = [];
labels = [];
tmpFolders = [];

for i = 1:numel(files)

    % Skipping '.' and '..'
    if files(i).isdir
        continue
    end

    display("Sampling " + files(i).name)

    % Loading video
    video = VideoReader("test_videos\" + files(i).name);
    if video.Duration == 0 || hasFrame(video) == false
        error("Invalid video.")
    end
    video.CurrentTime = 0;

    % Creating 
    [~, tmpFolderName, ~] = fileparts(video.Name);
    tmpFolderName = "test_videos\" + tmpFolderName;
    if ~exist(tmpFolderName, 'dir')
        mkdir(tmpFolderName);
        tmpFolders = [tmpFolders tmpFolderName];
    end

    % Sampling frames
    totalFrames = floor(video.Duration * video.FrameRate);
    framesToSample = floor(totalFrames * SAMPLE_PERCENTAGE / 100);
    stepSize = floor(totalFrames / framesToSample);
    
    for j = 1:framesToSample
        tmpFileName = tmpFolderName + filesep + "frame_" + j + ".png";

        if ~exist(tmpFileName, 'file')
            video.CurrentTime = (j - 1) * stepSize / video.FrameRate;
            frame = readFrame(video);
            imwrite(frame, tmpFileName);
        end

        frames = [frames tmpFileName];
    end

    % Adding frame labels
    label = split(video.Name, '_');
    label = label(1);
    label = str2num(label{1});
    tmp_labels = label * ones(framesToSample, 1);
    tmp_labels = categorical(tmp_labels);
    labels = [labels; tmp_labels];
end

%% Creating augmented image datastore
datastore = imageDatastore(frames, Labels=labels);
resizedDatastore = augmentedImageDatastore([227, 227, 3], datastore, ColorPreprocessing="gray2rgb");

%% Loading model
load 'model.mat'

%% Classify frames
[predictedLabels, scores] = classify(model, resizedDatastore);

% Computing accuracy
numel(find(predictedLabels == labels)) / numel(predictedLabels)

%% Cleaning up temporary folders
if size(tmpFolders, 2) > 0
    for i = 1:numel(frames)
        delete(frames(i));
    end
    for i = 1:numel(tmpFolders)
        rmdir(tmpFolders(i));
    end
end
