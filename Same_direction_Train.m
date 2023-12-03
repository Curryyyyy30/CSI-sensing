clc;
clear;
close all;

%% 得到第一种类别的数据文件索引
CurrentPath1     = '.\Same_direction\two_persons\';
FilesStruct1     = dir(fullfile(CurrentPath1));
lengthf1         = length(FilesStruct1) - 2;

for i = 1 : lengthf1
    FileNamePath        = strcat(CurrentPath1,FilesStruct1(i+2,1).name);
    if i < lengthf1 * 0.25 + 1
        LOS1(:,:,i)                 = FileNamePath;
    elseif i < lengthf1 * 0.5 + 1
        NLOS1(:,:,i-lengthf1*0.25)  = FileNamePath;
    elseif i < lengthf1 * 0.75 + 1
        offset1(:,:,i-lengthf1*0.5) = FileNamePath;
    else
        delay1(:,:,i-lengthf1*0.75) = FileNamePath;
    end
end

%% 得到第二种类别的数据文件索引
CurrentPath2     = '.\Same_direction\one_person\';
FilesStruct2     = dir(fullfile(CurrentPath2));
lengthf2         = length(FilesStruct2) - 2;

for i = 1 : lengthf2
    FileNamePath        = strcat(CurrentPath2,FilesStruct2(i+2,1).name);
    if i < lengthf2 / 3 + 1
        LOS2(:,:,i)                 = FileNamePath;
    elseif i < lengthf2 * 2 / 3 + 1
        NLOS2(:,:,i-lengthf2 / 3)  = FileNamePath;
    else
        delay2(:,:,i-lengthf2*2/3) = FileNamePath;
    end
end

%% 得到第三种类别的数据文件索引
CurrentPath3     = '.\Same_direction\three_persons\';
FilesStruct3     = dir(fullfile(CurrentPath3));
lengthf3         = length(FilesStruct3) - 2;

for i = 1 : lengthf3
    FileNamePath        = strcat(CurrentPath3,FilesStruct3(i+2,1).name);
    if i < lengthf3 *0.25 + 1
        LOS3(:,:,i)                 = FileNamePath;
    elseif i < lengthf3 * 0.5 + 1
        NLOS3(:,:,i-lengthf3 *0.25)  = FileNamePath;
    elseif i < lengthf3 * 0.75 + 1
        offset3(:,:,i-lengthf3 *0.5)  = FileNamePath;
    else
        delay3(:,:,i-lengthf3*0.75) = FileNamePath;
    end
end


%% 得到第四种类别的数据文件索引
CurrentPath4     = '.\Same_direction\four_persons\';
FilesStruct4     = dir(fullfile(CurrentPath4));
lengthf4         = length(FilesStruct4) - 2;

for i = 1 : lengthf4
    FileNamePath        = strcat(CurrentPath4,FilesStruct4(i+2,1).name);
    if i < lengthf4 * 0.25 + 1
        LOS4(:,:,i)                 = FileNamePath;
    elseif i < lengthf4 * 0.5 + 1
        NLOS4(:,:,i-lengthf4*0.25)  = FileNamePath;
    elseif i < lengthf4 * 0.75 + 1
        offset4(:,:,i-lengthf4*0.5) = FileNamePath;
    else
        delay4(:,:,i-lengthf4*0.75) = FileNamePath;
    end
end

%% 配置
lenT        = ceil(lengthf1/4 * 0.75) + ceil(lengthf2/3 * 0.75) + ceil(lengthf3/4 * 0.75)+ ceil(lengthf4/4 * 0.75);        % 3/4用于训练
lenV        = floor(lengthf1/4 * 0.25) + floor(lengthf2/3 * 0.25) + floor(lengthf3/4 * 0.25) + floor(lengthf4/4 * 0.25);     % 1/4用于验证
compression = 121;                          % CNN网络中CSI频域压缩
STFTlen     = 284;                          % CNN网络中STFT变换前数据长度
winlen      = 28;                           % CNN网络中STFT变换窗长
R           = 24;                           % CNN网络中STFT变换窗移动间隔
NFFT        = 256;                          % CNN网络中STFT变换窗内FFT的点数
fs          = 100;                          % CNN网络中STFT变换采样率
win         = blackman(winlen, 'periodic'); % CNN网络中STFT变换窗函数
S           = zeros(NFFT/2+1,(STFTlen-R)/(winlen-R));
DTrain      = cell(lenT*8,1);                                   % LSTM-FCN/LSTM网络训练数据
DALLTrain   = cell(lenT*8,1);
DSTFTTrain  = zeros(size(S,1),size(S,2),1,lenT*8);              % CNN网络训练数据
DValidation = cell(lenV*8,1);                                   % LSTM-FCN/LSTM网络验证数据
DALLValidation      = cell(lenV*8,1);
DSTFTValidation     = zeros(size(S,1),size(S,2),1,lenV*8);      % CNN网络验证数据

%% 加载第一种类别的训练数据
for i = 1 : ceil(lengthf1/4 * 0.75)
    getPath         = LOS1(:,:,i);
    data            = importdata(getPath);
    getPath         = offset1(:,:,i);
    offset          = importdata(getPath);
    getPath         = delay1(:,:,i);
    delay           = importdata(getPath);
    t               = find(offset > offset(1)+2);
    data(:,:,t)     = [];
    offset(:,t)     = [];
    delay(:,t)      = [];
    t               = find(offset < offset(1)-2);
    data(:,:,t)     = [];
    offset(:,t)     = [];
    delay(:,t)      = [];
    [~,~,peakind]   = crossDet(delay);
    traindata       = zeros(size(data,3),size(data,1));
%     PCA
    for j = 1 : size(data,1)
        C       = cov(squeeze(data(j,:,:)).');
        [E,R0]  = eig(C);
        R0      = ones(1,242)*R0;
        TZ_ER   = [R0;E]';
        TZ_ER   = sortrows(TZ_ER,1,'descend');
        R0      = TZ_ER(:,1);
        E       = TZ_ER(:,2:end)';
        PCA     = E(:,1);
        traindata(:,j)= squeeze(data(j,:,:)).'*PCA;
    end
    stfttraindata   = zeros(STFTlen,size(data,1));
    if peakind <= size(delay,2)
        stfttraindata       = traindata(1:STFTlen,:);
    else
        stfttraindata       = traindata(size(data,3)-STFTlen+1:end,:);
    end
    for j = 1 : size(data,1)
        DTrain{(i-1)*8+j,1}     = traindata(:,j).';                           % 加载LSTM-FCN/LSTM网络训练数据
        DALLTrain{(i-1)*8+j,1}  = squeeze(data(j,:,:));
        [S(:,:), ~, ~]          = spectrogram(squeeze(abs(stfttraindata(:,j))), win, R, NFFT, fs);
        DSTFTTrain(:,:,1,(i-1)*8+j) = S;                                      % 加载CNN网络训练数据
    end
end

%% 加载第一种类别的验证数据
for i = ceil(lengthf1/4 * 0.75) + 1 : lengthf1/4
    getPath         = LOS1(:,:,i);
    data            = importdata(getPath);
    getPath         = offset1(:,:,i);
    offset          = importdata(getPath);
    getPath         = delay1(:,:,i);
    delay           = importdata(getPath);
    t               = find(offset > offset(1)+2);
    data(:,:,t)     = [];
    offset(:,t)     = [];
    delay(:,t)      = [];
    t               = find(offset < offset(1)-2);
    data(:,:,t)     = [];
    offset(:,t)     = [];
    delay(:,t)      = [];
    [~,~,peakind]   = crossDet(delay);
    traindata       = zeros(size(data,3),size(data,1));
    for j = 1 : size(data,1)
        C       = cov(squeeze(data(j,:,:)).');
        [E,R0]  = eig(C);
        R0      = ones(1,242)*R0;
        TZ_ER   = [R0;E]';
        TZ_ER   = sortrows(TZ_ER,1,'descend');
        R0      = TZ_ER(:,1);
        E       = TZ_ER(:,2:end)';
        PCA     = E(:,1);
        traindata(:,j)= squeeze(data(j,:,:)).'*PCA;
    end
    stfttraindata   = zeros(STFTlen,size(data,1));
    if peakind <= size(delay,2)
        stfttraindata       = traindata(1:STFTlen,:);
    else
        stfttraindata       = traindata(size(data,3)-STFTlen+1:end,:);
    end
    for j = 1 : 8
        DValidation{(i-ceil(lengthf1/4*0.75)-1)*8+j,1}     = squeeze(traindata(:,j)).';
        DALLValidation{(i-ceil(lengthf1/4*0.75)-1)*8+j,1}  = squeeze(data(j,:,:));
        [S(:,:), ~, ~]    = spectrogram(squeeze(abs(stfttraindata(:,j))), win, R, NFFT, fs);
        DSTFTValidation(:,:,1,(i-ceil(lengthf1/4*0.75)-1)*8+j) = S;
    end
end

%% 加载第二种类别的训练数据
for i = 1 : ceil(lengthf2/3 * 0.75)
    getPath         = LOS2(:,:,i);
    data            = importdata(getPath);
    getPath         = delay2(:,:,i);
    delay           = importdata(getPath);
    [~,~,peakind]   = crossDet(delay);
    traindata       = zeros(size(data,3),size(data,1));
    for j = 1 : size(data,1)
        C       = cov(squeeze(data(j,:,:)).');
        [E,R0]  = eig(C);
        R0      = ones(1,242)*R0;
        TZ_ER   = [R0;E]';
        TZ_ER   = sortrows(TZ_ER,1,'descend');
        R0      = TZ_ER(:,1);
        E       = TZ_ER(:,2:end)';
        PCA     = E(:,1);
        traindata(:,j)= squeeze(data(j,:,:)).'*PCA;
    end
    stfttraindata   = zeros(STFTlen,size(data,1));
    if peakind <= size(delay,2)
        stfttraindata       = traindata(1:STFTlen,:);
    else
        stfttraindata       = traindata(size(data,3)-STFTlen+1:end,:);
    end
    for j = 1 : 8
        DTrain{ceil(lengthf1/4*0.75)*8+(i-1)*8+j,1}     = squeeze(traindata(:,j)).';
        DALLTrain{ceil(lengthf1/4*0.75)*8+(i-1)*8+j,1}  = squeeze(data(j,:,:));
        [S(:,:), ~, ~]    = spectrogram(squeeze(abs(stfttraindata(:,j))), win, R, NFFT, fs);
        DSTFTTrain(:,:,1,ceil(lengthf1/4*0.75)*8+(i-1)*8+j) = S;
    end
end

%% 加载第二种类别的验证数据
for i = ceil(lengthf2/3 * 0.75) + 1 : lengthf2/3
    getPath         = LOS2(:,:,i);
    data            = importdata(getPath);
    getPath         = delay2(:,:,i);
    delay           = importdata(getPath);
    [~,~,peakind]   = crossDet(delay);
    traindata       = zeros(size(data,3),size(data,1));
    for j = 1 : size(data,1)
        C       = cov(squeeze(data(j,:,:)).');
        [E,R0]  = eig(C);
        R0      = ones(1,242)*R0;
        TZ_ER   = [R0;E]';
        TZ_ER   = sortrows(TZ_ER,1,'descend');
        R0      = TZ_ER(:,1);
        E       = TZ_ER(:,2:end)';
        PCA     = E(:,1);
        traindata(:,j)= squeeze(data(j,:,:)).'*PCA;
    end
    stfttraindata   = zeros(STFTlen,size(data,1));
    if peakind <= size(delay,2)
        stfttraindata       = traindata(1:STFTlen,:);
    else
        stfttraindata       = traindata(size(data,3)-STFTlen+1:end,:);
    end
    for j = 1 : 8
        DValidation{floor(lengthf1/4*0.25)*8+(i-ceil(lengthf2/3*0.75)-1)*8+j,1}     = squeeze(traindata(:,j)).';
        DALLValidation{floor(lengthf1/4*0.25)*8+(i-ceil(lengthf2/3*0.75)-1)*8+j,1} = squeeze(data(j,:,:));
        [S(:,:), ~, ~]    = spectrogram(squeeze(abs(stfttraindata(:,j))), win, R, NFFT, fs);
        DSTFTValidation(:,:,1,floor(lengthf1/4*0.25)*8+(i-ceil(lengthf2/3*0.75)-1)*8+j) = S;
    end
end

%% 加载第三种类别的训练数据
for i = 1 : ceil(lengthf3/4 * 0.75)
    getPath         = LOS3(:,:,i);
    data            = importdata(getPath);
    getPath         = offset3(:,:,i);
    offset          = importdata(getPath);
    getPath         = delay3(:,:,i);
    delay           = importdata(getPath);
    t               = find(offset > offset(1)+2);
    data(:,:,t)     = [];
    offset(:,t)     = [];
    delay(:,t)      = [];
    t               = find(offset < offset(1)-2);
    data(:,:,t)     = [];
    offset(:,t)     = [];
    delay(:,t)      = [];
    [~,~,peakind]   = crossDet(delay);
    traindata       = zeros(size(data,3),size(data,1));
    for j = 1 : size(data,1)
        C       = cov(squeeze(data(j,:,:)).');
        [E,R0]  = eig(C);
        R0      = ones(1,242)*R0;
        TZ_ER   = [R0;E]';
        TZ_ER   = sortrows(TZ_ER,1,'descend');
        R0      = TZ_ER(:,1);
        E       = TZ_ER(:,2:end)';
        PCA     = E(:,1);
        traindata(:,j)= squeeze(data(j,:,:)).'*PCA;
    end
    stfttraindata   = zeros(STFTlen,size(data,1));
    if peakind <= size(delay,2)
        stfttraindata       = traindata(1:STFTlen,:);
    else
        stfttraindata       = traindata(size(data,3)-STFTlen+1:end,:);
    end
    for j = 1 : 8
        DTrain{ceil(lengthf1/4*0.75)*8+ceil(lengthf2/3*0.75)*8+(i-1)*8+j,1}     = squeeze(traindata(:,j)).';
        DALLTrain{ceil(lengthf1/4*0.75)*8+ceil(lengthf2/3*0.75)*8+(i-1)*8+j,1}  = squeeze(data(j,:,:));
        [S(:,:), ~, ~]    = spectrogram(squeeze(abs(stfttraindata(:,j))), win, R, NFFT, fs);
        DSTFTTrain(:,:,1,ceil(lengthf1/4*0.75)*8+ceil(lengthf2/3*0.75)*8+(i-1)*8+j) = S;
    end
end

%% 加载第三种类别的验证数据
for i = ceil(lengthf3/4 * 0.75) + 1 : lengthf3/4
    getPath         = LOS3(:,:,i);
    data            = importdata(getPath);
    getPath         = offset3(:,:,i);
    offset          = importdata(getPath);
    getPath         = delay3(:,:,i);
    delay           = importdata(getPath);
    t               = find(offset > offset(1)+2);
    data(:,:,t)     = [];
    offset(:,t)     = [];
    delay(:,t)      = [];
    t               = find(offset < offset(1)-2);
    data(:,:,t)     = [];
    offset(:,t)     = [];
    delay(:,t)      = [];
    [~,~,peakind]   = crossDet(delay);
    traindata       = zeros(size(data,3),size(data,1));
    for j = 1 : size(data,1)
        C       = cov(squeeze(data(j,:,:)).');
        [E,R0]  = eig(C);
        R0      = ones(1,242)*R0;
        TZ_ER   = [R0;E]';
        TZ_ER   = sortrows(TZ_ER,1,'descend');
        R0      = TZ_ER(:,1);
        E       = TZ_ER(:,2:end)';
        PCA     = E(:,1);
        traindata(:,j)= squeeze(data(j,:,:)).'*PCA;
    end
    stfttraindata   = zeros(STFTlen,size(data,1));
    if peakind <= size(delay,2)
        stfttraindata       = traindata(1:STFTlen,:);
    else
        stfttraindata       = traindata(size(data,3)-STFTlen+1:end,:);
    end
    for j = 1 : 8
        DValidation{floor(lengthf1/4*0.25)*8+floor(lengthf2/3*0.25)*8+(i-ceil(lengthf3/4*0.75)-1)*8+j,1}        = squeeze(traindata(:,j)).';
        DALLValidation{floor(lengthf1/4*0.25)*8+floor(lengthf2/3*0.25)*8+(i-ceil(lengthf3/4*0.75)-1)*8+j,1}     = squeeze(data(j,:,:));
        [S(:,:), ~, ~]    = spectrogram(squeeze(abs(stfttraindata(:,j))), win, R, NFFT, fs);
        DSTFTValidation(:,:,1,floor(lengthf1/4*0.25)*8+floor(lengthf2/3*0.25)*8+(i-ceil(lengthf3/4*0.75)-1)*8+j) = S;
    end
end
%% 加载第四种类别的训练数据
for i = 1 : ceil(lengthf4/4 * 0.75)
    getPath         = LOS4(:,:,i);
    data            = importdata(getPath);
    getPath         = offset4(:,:,i);
    offset          = importdata(getPath);
    getPath         = delay4(:,:,i);
    delay           = importdata(getPath);
    t               = find(offset > offset(1)+1);
    data(:,:,t)     = [];
    offset(:,t)     = [];
    delay(:,t)      = [];
    t               = find(offset < offset(1)-1);
    data(:,:,t)     = [];
    offset(:,t)     = [];
    delay(:,t)      = [];
    [~,~,peakind]   = crossDet(delay);
    traindata       = zeros(size(data,3),size(data,1));
    for j = 1 : size(data,1)
        C       = cov(squeeze(data(j,:,:)).');
        [E,R0]  = eig(C);
        R0      = ones(1,242)*R0;
        TZ_ER   = [R0;E]';
        TZ_ER   = sortrows(TZ_ER,1,'descend');
        R0      = TZ_ER(:,1);
        E       = TZ_ER(:,2:end)';
        PCA     = E(:,1);
        traindata(:,j)= squeeze(data(j,:,:)).'*PCA;
    end
    stfttraindata   = zeros(STFTlen,size(data,1));
    if peakind <= size(delay,2)
        stfttraindata       = traindata(1:STFTlen,:);
    else
        stfttraindata       = traindata(size(data,3)-STFTlen+1:end,:);
    end
    for j = 1 : 8
        DTrain{ceil(lengthf1/4*0.75)*8+ceil(lengthf2/3*0.75)*8+ceil(lengthf3/4*0.75)*8+(i-1)*8+j,1}     = squeeze(traindata(:,j)).';
        DALLTrain{ceil(lengthf1/4*0.75)*8+ceil(lengthf2/3*0.75)*8+ceil(lengthf3/4*0.75)*8+(i-1)*8+j,1}  = squeeze(data(j,:,:));
        [S(:,:), ~, ~]    = spectrogram(squeeze(abs(stfttraindata(:,j))), win, R, NFFT, fs);
        DSTFTTrain(:,:,1,ceil(lengthf1/4*0.75)*8+ceil(lengthf2/3*0.75)*8+ceil(lengthf3/4*0.75)*8+(i-1)*8+j) = S;
    end
end

%% 加载第四种类别的验证数据
for i = ceil(lengthf4/4 * 0.75) + 1 : lengthf4/4
    getPath         = LOS4(:,:,i);
    data            = importdata(getPath);
    getPath         = offset4(:,:,i);
    offset          = importdata(getPath);
    getPath         = delay4(:,:,i);
    delay           = importdata(getPath);
    t               = find(offset > offset(1)+1);
    data(:,:,t)     = [];
    offset(:,t)     = [];
    delay(:,t)      = [];
    t               = find(offset < offset(1)-1);
    data(:,:,t)     = [];
    offset(:,t)     = [];
    delay(:,t)      = [];
    [~,~,peakind]   = crossDet(delay);
    traindata       = zeros(size(data,3),size(data,1));
    for j = 1 : size(data,1)
        C       = cov(squeeze(data(j,:,:)).');
        [E,R0]  = eig(C);
        R0      = ones(1,242)*R0;
        TZ_ER   = [R0;E]';
        TZ_ER   = sortrows(TZ_ER,1,'descend');
        R0      = TZ_ER(:,1);
        E       = TZ_ER(:,2:end)';
        PCA     = E(:,1);
        traindata(:,j)= squeeze(data(j,:,:)).'*PCA;
    end
    stfttraindata   = zeros(STFTlen,size(data,1));
    if peakind <= size(delay,2)
        stfttraindata       = traindata(1:STFTlen,:);
    else
        stfttraindata       = traindata(size(data,3)-STFTlen+1:end,:);
    end
    for j = 1 : 8
        DValidation{floor(lengthf1/4*0.25)*8+floor(lengthf2/3*0.25)*8+floor(lengthf3/4*0.25)*8+(i-ceil(lengthf4/4*0.75)-1)*8+j,1}        = squeeze(traindata(:,j)).';
        DALLValidation{floor(lengthf1/4*0.25)*8+floor(lengthf2/3*0.25)*8+floor(lengthf3/4*0.25)*8+(i-ceil(lengthf4/4*0.75)-1)*8+j,1}     = squeeze(data(j,:,:));
        [S(:,:), ~, ~]    = spectrogram(squeeze(abs(stfttraindata(:,j))), win, R, NFFT, fs);
        DSTFTValidation(:,:,1,floor(lengthf1/4*0.25)*8+floor(lengthf2/3*0.25)*8+floor(lengthf3/4*0.25)*8+(i-ceil(lengthf4/4*0.75)-1)*8+j) = S;
    end
end

%% 生成各类别标签
TTrain          = categorical([2*ones(ceil(lengthf1/4*0.75)*8,1);ones(ceil(lengthf2/3*0.75)*8,1);3*ones(ceil(lengthf3/4*0.75)*8,1);4*ones(ceil(lengthf4/4*0.75)*8,1)]);
TValidation     = categorical([2*ones(floor(lengthf1/4*0.25)*8,1);ones(floor(lengthf2/3*0.25)*8,1);3*ones(floor(lengthf3/4*0.25)*8,1);4*ones(floor(lengthf4/4*0.25)*8,1)]);

% LSTM
layers1 = [
    sequenceInputLayer(1,"Name","sequence","MinLength",200,"SplitComplexInputs",true)
    lstmLayer(128,"Name","bilstm","OutputMode","last")
    dropoutLayer(0.5,"Name","dropout")
    fullyConnectedLayer(4,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

% CNN
layers2 = [
    imageInputLayer([129 65 1],"Name","imageinput","SplitComplexInputs",true)
    convolution2dLayer([3 3],6,"Name","conv1")
    reluLayer("Name","relu1")
    maxPooling2dLayer([3 3],"Name","maxpool1")
    convolution2dLayer([3 3],12,"Name","conv2")
    reluLayer("Name","relu2")
    maxPooling2dLayer([3 3],"Name","maxpool2")
    flattenLayer("Name","flatten")
    fullyConnectedLayer(500,"Name","fc1")
    fullyConnectedLayer(3,"Name","fc2")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];

% LSTM-FCN
lgraph = layerGraph();
tempLayers = sequenceInputLayer(1,"Name","sequence","MinLength",200,"SplitComplexInputs",true);
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    lstmLayer(128,"Name","lstm","OutputMode","last")
    dropoutLayer(0.5,"Name","dropout")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution1dLayer(3,32,"Name","conv1d","Padding","same")
    batchNormalizationLayer("Name","batchnorm")
    reluLayer("Name","relu")
    convolution1dLayer(3,32,"Name","conv1d_1","Padding","same")
    batchNormalizationLayer("Name","batchnorm_1")
    reluLayer("Name","relu_1")
    convolution1dLayer(3,32,"Name","conv1d_2","Padding","same")
    batchNormalizationLayer("Name","batchnorm_2")
    reluLayer("Name","relu_2")
    globalAveragePooling1dLayer("Name","gapool1d")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    concatenationLayer(1,2,"Name","concat")
    fullyConnectedLayer(4,"Name","fc")
    softmaxLayer("Name","softmax")
    classificationLayer("Name","classoutput")];
lgraph = addLayers(lgraph,tempLayers);

% 清理辅助变量
clear tempLayers;
lgraph = connectLayers(lgraph,"sequence","lstm");
lgraph = connectLayers(lgraph,"sequence","conv1d");
lgraph = connectLayers(lgraph,"dropout","concat/in2");
lgraph = connectLayers(lgraph,"gapool1d","concat/in1");
plot(lgraph);
miniBatchSize = 64;

options = trainingOptions('sgdm', ...
    'ExecutionEnvironment','auto', ...
    'ValidationFrequency',48,...
    'MaxEpochs',50, ...
    'MiniBatchSize',miniBatchSize, ...
    'ValidationData',{DValidation,TValidation}, ...
    'GradientThreshold',2, ...
    'Shuffle','every-epoch', ...
    'Verbose',false, ...
    'Plots','training-progress');

[net1,info1] = trainNetwork(DTrain,TTrain,lgraph,options);
close