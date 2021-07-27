clear;
readMNISTData;
initAlpha=0.01;
testnet=myNet;
testnet=addLayer(testnet,myConvLayer([32 32 1],[5 5 6],initAlpha));
testnet=addLayer(testnet,myReLuLayer);
testnet=addLayer(testnet,myPoolingLayer([28 28 6]));
testnet=addLayer(testnet,myConvLayer([14 14 6],[5 5 16],initAlpha));
testnet=addLayer(testnet,myReLuLayer);
testnet=addLayer(testnet,myPoolingLayer([10 10 16]));
testnet=addLayer(testnet,myFullConnLayer([400 120],initAlpha));
testnet=addLayer(testnet,myReLuLayer);
testnet=addLayer(testnet,myFullConnLayer([120 84],initAlpha));
testnet=addLayer(testnet,myReLuLayer);
testnet=addLayer(testnet,myFullConnLayer([84 10],initAlpha));
testnet=addLayer(testnet,mySoftmaxLayer);
Iteration=10;
BatchSize=100;
M=length(yt);
randIdx=randperm(M,BatchSize);
%alphaFactor=2;
for i=1:Iteration
    XBatch=Xt(:,:,randIdx);
    YBatch=(1:10)==yt(randIdx);
    meanJ=0;
    for b=1:BatchSize
        %fprintf("Iteration:%d,Batch:%d\n",i,b);
        J=inf;
        while 1
            [testnet,J,h]=forwardPropogation(testnet,XBatch(:,:,b),YBatch(b,:));
            if J<1e-3
                break
            end
            testnet=backwardPorpogation(testnet,h,YBatch(b,:));
        end
        meanJ=meanJ+J;
    end
    meanJ=meanJ/BatchSize;
    fprintf("Iteration %d end,mean cost=%f\n",i,meanJ);
    if meanJ<1e-9
        break
    end
end
rightans=0;
for ts=1:size(Xts,3)
    ytsM=(1:10)==yts;
    [maxh,i]=predict(testnet,Xts(:,:,ts));
    
    if ytsM(ts,i)==1
        rightans=rightans+1;
    end
end
fprintf("正确率为%f",rightans/size(Xts,3)*100);


