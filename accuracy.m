readMNISTData;
numSet=length(yts);
rightans=0;
ytsM=(1:10)==yts;
[maxh,h]=predict(testnet,reshape(Xts,32,32,1,numSet));

fprintf("正确率为%f\n",sum(h==yts')/numSet);
