%% segment_imp.m
% Improved U-Net training setup for crop/weed semantic segmentation


clear; clc; close all;


% config

runImproved      = true;
evaluateImproved = true;

showDebugFigures = false;
showTrainingPlots = true;
quickTestMode    = false;
executionEnv     = 'auto';
inputSize        = [360 480 3];


% Improved experiment config

experimentName   = 'final_F_depth2_adam_noweights_reflectonly_lr5e4';
augmentationMode = 'reflectonly';
encoderDepthImp  = 2;
optimizerImp     = 'adam';
useClassWeights  = false;
learnRateImp     = 5e-4;


% Training settings

if quickTestMode
    maxEpochsImp = 5;
    valFreqImp   = 40;
else
    maxEpochsImp = 15;
    valFreqImp   = 20;
end

miniBatchImp = 1;

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


% Improved datastores

tbl = countEachLabel(pxdsTrain);

switch lower(augmentationMode)
    case 'current'
        augmenter = imageDataAugmenter( ...
            'RandXReflection', true, ...
            'RandRotation', [-10 10], ...
            'RandXTranslation', [-10 10], ...
            'RandYTranslation', [-10 10]);
    case 'mild'
        augmenter = imageDataAugmenter( ...
            'RandXReflection', true, ...
            'RandRotation', [-5 5], ...
            'RandXTranslation', [-5 5], ...
            'RandYTranslation', [-5 5]);
    case 'reflectonly'
        augmenter = imageDataAugmenter('RandXReflection', true);
    otherwise
        error('Unknown augmentationMode: %s', augmentationMode);
end

pximdsTrainImp = pixelLabelImageDatastore(imdsTrain, pxdsTrain, ...
    'OutputSize', inputSize(1:2), ...
    'DataAugmentation', augmenter);
pximdsValImp = pixelLabelImageDatastore(imdsVal, pxdsVal, ...
    'OutputSize', inputSize(1:2));


% Improved model

totalPixels = sum(tbl.PixelCount);
frequency = tbl.PixelCount / totalPixels;
classWeights = 1 ./ sqrt(frequency);
classWeights = classWeights / mean(classWeights);
classWeights = double(classWeights(:));

lgraphImp = unetLayers(inputSize, numel(classNames), 'EncoderDepth', encoderDepthImp);
lastLayerName = lgraphImp.Layers(end).Name;

if useClassWeights
    pxLayerImp = pixelClassificationLayer( ...
        'Name', lastLayerName, ...
        'Classes', classNames, ...
        'ClassWeights', classWeights);
else
    pxLayerImp = pixelClassificationLayer( ...
        'Name', lastLayerName, ...
        'Classes', classNames);
end

lgraphImp = replaceLayer(lgraphImp, lastLayerName, pxLayerImp);

improvedModelFile   = fullfile(scriptDir, 'segmentnet_imp.mat');
improvedMetricsFile = fullfile(scriptDir, 'improved_metrics.mat');

optionsImp = trainingOptions(optimizerImp, ...
    'InitialLearnRate', learnRateImp, ...
    'MaxEpochs', maxEpochsImp, ...
    'MiniBatchSize', miniBatchImp, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', pximdsValImp, ...
    'ValidationFrequency', valFreqImp, ...
    'VerboseFrequency', 20, ...
    'Plots', plotMode, ...
    'Verbose', true, ...
    'ExecutionEnvironment', executionEnv);

if runImproved
    netImp = trainNetwork(pximdsTrainImp, lgraphImp, optionsImp);
    save(improvedModelFile, ...
        'netImp', 'inputSize', 'classNames', 'labelIDs', ...
        'trainIdx', 'valIdx', 'classWeights', ...
        'augmentationMode', 'encoderDepthImp', ...
        'experimentName', 'optimizerImp', ...
        'useClassWeights', 'learnRateImp');
    disp(['Saved improved model to: ' improvedModelFile]);
else
    assert(isfile(improvedModelFile), ...
        'No saved improved model found. Set runImproved = true once first.');
    S = load(improvedModelFile, 'netImp');
    netImp = S.netImp;
end


% Improved evaluation

if evaluateImproved
    [metricsImp, ~, ~] = runSegmentationEvaluation( ...
        imdsVal, valMaskFiles, netImp, inputSize, classNames, labelIDs, ...
        fullfile(scriptDir, ['eval_imp_' experimentName]));

    disp('Improved dataset metrics:');
    disp(metricsImp.DataSetMetrics);
    disp('Improved per-class metrics:');
    disp(metricsImp.ClassMetrics);

    if ismember('MeanIoU', metricsImp.DataSetMetrics.Properties.VariableNames)
        meanIoUImp = metricsImp.DataSetMetrics.MeanIoU;
    else
        meanIoUImp = mean(metricsImp.ClassMetrics.IoU, 'omitnan');
    end
    fprintf('Improved Mean IoU = %.5f\n', meanIoUImp);
    save(improvedMetricsFile, 'metricsImp', 'meanIoUImp');
    disp(['Saved improved metrics to: ' improvedMetricsFile]);
end





%% helper function
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
