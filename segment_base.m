%% segment_base.m
% Baseline U-Net for crop/weed semantic segmentation

clear; clc; close all;


% config

runBaseline      = true;
evaluateBaseline = true;

showDebugFigures = false;
showTrainingPlots = true;
quickTestMode    = false;
executionEnv     = 'auto';
inputSize        = [360 480 3];


% Training settings

if quickTestMode
    maxEpochsBase = 3;
    valFreqBase   = 40;
else
    maxEpochsBase = 15;
    valFreqBase   = 20;
end

miniBatchBase = 2;

if showTrainingPlots
    plotMode = 'training-progress';
else
    plotMode = 'none';
end


% Paths and data 

scriptDir = fileparts(mfilename('fullpath'));
imgDir = fullfile(scriptDir, 'cw_data', 'images');
segDir = fullfile(scriptDir, 'cw_data', 'segmentation');

assert(isfolder(imgDir), "Image folder not found: " + imgDir);
assert(isfolder(segDir), "Segmentation folder not found: " + segDir);

imds = imageDatastore(imgDir, ...
    'IncludeSubfolders', false, ...
    'FileExtensions', {'.png','.jpg','.jpeg','.bmp','.tif','.tiff'});
assert(~isempty(imds.Files), 'No raw images found.');

alignedMaskFiles = alignMaskFilesToImages(imds.Files, segDir);

classNames = ["background","crop","weed"];
labelIDs   = uint8([
      0   0   0;
      0 255   0;
    255   0   0]);

pxds = pixelLabelDatastore(alignedMaskFiles, classNames, labelIDs);

assert(numel(imds.Files) == numel(alignedMaskFiles), ...
    'Number of raw images and segmentation masks must match.');

for k = 1:numel(imds.Files)
    I = imread(imds.Files{k});
    M = imread(alignedMaskFiles{k});
    assert(size(I,1) == size(M,1) && size(I,2) == size(M,2), ...
        'Size mismatch at pair %d:\nImage: %s\nMask: %s', ...
        k, imds.Files{k}, alignedMaskFiles{k});
end


% Train / validation split

rng(0);
numImages = numel(imds.Files);
allIdx = randperm(numImages);
numTrain = round(0.8 * numImages);
trainIdx = allIdx(1:numTrain);
valIdx   = allIdx(numTrain+1:end);

imdsTrain = subset(imds, trainIdx);
pxdsTrain = subset(pxds, trainIdx);
imdsVal = subset(imds, valIdx);
pxdsVal = subset(pxds, valIdx);
valMaskFiles = alignedMaskFiles(valIdx);


% Resized datastores

pximdsTrain = pixelLabelImageDatastore(imdsTrain, pxdsTrain, ...
    'OutputSize', inputSize(1:2));
pximdsVal = pixelLabelImageDatastore(imdsVal, pxdsVal, ...
    'OutputSize', inputSize(1:2));


% Baseline model

lgraphBase = unetLayers(inputSize, numel(classNames), 'EncoderDepth', 2);

baselineModelFile   = fullfile(scriptDir, 'segmentnet_base.mat');
baselineMetricsFile = fullfile(scriptDir, 'baseline_metrics.mat');

optionsBase = trainingOptions('sgdm', ...
    'InitialLearnRate', 1e-3, ...
    'MaxEpochs', maxEpochsBase, ...
    'MiniBatchSize', miniBatchBase, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', pximdsVal, ...
    'ValidationFrequency', valFreqBase, ...
    'VerboseFrequency', 20, ...
    'Plots', plotMode, ...
    'Verbose', true, ...
    'ExecutionEnvironment', executionEnv);

if runBaseline
    netBase = trainNetwork(pximdsTrain, lgraphBase, optionsBase);
    save(baselineModelFile, ...
        'netBase', 'inputSize', 'classNames', 'labelIDs', ...
        'trainIdx', 'valIdx');
    disp(['Saved baseline model to: ' baselineModelFile]);
else
    assert(isfile(baselineModelFile), ...
        'No saved baseline model found. Set runBaseline = true once first.');
    S = load(baselineModelFile, 'netBase');
    netBase = S.netBase;
end


% Baseline evaluation

if evaluateBaseline
    [metricsBase, ~, ~] = runSegmentationEvaluation( ...
        imdsVal, valMaskFiles, netBase, inputSize, classNames, labelIDs, ...
        fullfile(scriptDir, 'eval_base'));

    disp('Baseline dataset metrics:');
    disp(metricsBase.DataSetMetrics);
    disp('Baseline per-class metrics:');
    disp(metricsBase.ClassMetrics);

    if ismember('MeanIoU', metricsBase.DataSetMetrics.Properties.VariableNames)
        meanIoUBase = metricsBase.DataSetMetrics.MeanIoU;
    else
        meanIoUBase = mean(metricsBase.ClassMetrics.IoU, 'omitnan');
    end
    fprintf('Baseline Mean IoU = %.5f\n', meanIoUBase);
    save(baselineMetricsFile, 'metricsBase', 'meanIoUBase');
    disp(['Saved baseline metrics to: ' baselineMetricsFile]);
end



%% helper functions
function alignedMaskFiles = alignMaskFilesToImages(imageFiles, segDir)
    segStruct = dir(fullfile(segDir, '*.png'));
    assert(~isempty(segStruct), 'No segmentation PNG files found in %s', segDir);
    maskMap = containers.Map('KeyType', 'char', 'ValueType', 'char');
    for i = 1:numel(segStruct)
        maskPath = fullfile(segStruct(i).folder, segStruct(i).name);
        [~, base, ~] = fileparts(segStruct(i).name);
        key = lower(base);
        if isKey(maskMap, key)
            error('Duplicate segmentation basename found: %s', base);
        end
        maskMap(key) = maskPath;
    end
    alignedMaskFiles = cell(numel(imageFiles), 1);
    for i = 1:numel(imageFiles)
        [~, imgBase, ~] = fileparts(imageFiles{i});
        key = lower(imgBase);
        if ~isKey(maskMap, key)
            error('No matching mask found for image: %s', imageFiles{i});
        end
        alignedMaskFiles{i} = maskMap(key);
    end
end

function [metrics, pxdsPred, imdsEval] = runSegmentationEvaluation( ...
    imdsVal, valMaskFiles, net, inputSize, classNames, labelIDs, evalRoot)

    evalImgDir   = fullfile(evalRoot, 'images_resized');
    evalTruthDir = fullfile(evalRoot, 'truth_resized');
    predRoot     = fullfile(evalRoot, 'predictions');

    if ~isfolder(evalRoot), mkdir(evalRoot); end
    if ~isfolder(evalImgDir), mkdir(evalImgDir); end
    if ~isfolder(evalTruthDir), mkdir(evalTruthDir); end
    if isfolder(predRoot), rmdir(predRoot, 's'); end
    mkdir(predRoot);

    for n = 1:numel(imdsVal.Files)
        I = readimage(imdsVal, n);
        Ismall = imresize(I, inputSize(1:2));
        [~, imgName, ~] = fileparts(imdsVal.Files{n});
        outImgPath   = fullfile(evalImgDir,   [imgName '.png']);
        outTruthPath = fullfile(evalTruthDir, [imgName '.png']);
        imwrite(Ismall, outImgPath);
        M = imread(valMaskFiles{n});
        Msmall = imresize(M, inputSize(1:2), 'nearest');
        imwrite(Msmall, outTruthPath);
    end

    imdsEval = imageDatastore(evalImgDir);
    pxdsTruthEval = pixelLabelDatastore(evalTruthDir, classNames, labelIDs);
    pxdsPred = semanticseg(imdsEval, net, ...
        'WriteLocation', predRoot, ...
        'MiniBatchSize', 1, ...
        'Verbose', true);
    metrics = evaluateSemanticSegmentation(pxdsPred, pxdsTruthEval, ...
        'Verbose', true);
end
