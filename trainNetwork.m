clear;
readMNISTData;
testnet=myNet;
testnet=addLayer(testnet,myConvLayer([32 32 1],[5 5 6]));
testnet=addLayer(testnet,myReLuLayer);
testnet=addLayer(testnet,myPoolingLayer([28 28 6]));
testnet=addLayer(testnet,myConvLayer([14 14 6],[5 5 16]));
testnet=addLayer(testnet,myReLuLayer);
testnet=addLayer(testnet,myPoolingLayer([10 10 16]));
testnet=addLayer(testnet,myFullConnLayer([120 400]));
testnet=addLayer(testnet,myReLuLayer);
testnet=addLayer(testnet,myFullConnLayer([84 120]));
testnet=addLayer(testnet,myReLuLayer);
testnet=addLayer(testnet,myFullConnLayer([10 84]));
testnet=addLayer(testnet,mySoftmaxLayer);
Iteration=100;
BatchNum=100;
BatchSize=600;
M=length(yt);
%alphaFactor=2;
maxTry=1000;
XBatch=zeros(32,32,BatchSize,BatchNum);
YBatch=zeros(10,BatchSize,BatchNum);

for i=1:BatchNum
    randIdx=randperm(M,BatchSize);
    XBatch(:,:,:,i)=reshape(Xt(:,:,randIdx),32,32,BatchSize);
    YBatch(:,:,i)=((1:10)==yt(randIdx))';
    Xt(:,:,randIdx)=[];
    yt(randIdx)=[];
    M=M-BatchSize;
end
meanJ=0;
for i=1:Iteration
    for b=1:BatchNum
        [testnet,J,h]=forwardPropogation(testnet,reshape(XBatch(:,:,:,b),32,32,1,BatchSize),YBatch(:,:,b),BatchSize);
        testnet=backwardPorpogation(testnet,h,YBatch(:,:,b),i);
        meanJ=meanJ+J;
    end
    meanJ=meanJ/BatchNum;
    fprintf("Ieration%d end，mean J= %f\n",i,meanJ);
end
rightans=0;
for ts=1:size(Xts,3)
    ytsM=(1:10)==yts;
    [maxh,i]=predict(testnet,Xts(:,:,ts));
    
    if ytsM(ts,i)==1
        rightans=rightans+1;
    end
end
fprintf("全部训练完成，正确率为%f",rightans/size(Xts,3)*100);


