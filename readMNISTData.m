f=fopen(fullfile(pwd, 'train-images-idx3-ubyte'),'r') ;
x1=fread(f,inf,'uint8');
fclose(f) ;
Xt=permute(reshape(x1(17:end),28,28,60e3),[2 1 3]) ;
Xt=rescale(padarray(Xt,[2 2],0,'both'));

f=fopen(fullfile(pwd, 't10k-images-idx3-ubyte'),'r') ;
x2=fread(f,inf,'uint8');
fclose(f) ;
Xts=permute(reshape(x2(17:end),28,28,10e3),[2 1 3]) ;
Xts=rescale(padarray(Xts,[2 2],0,'both'));

f=fopen(fullfile(pwd, 'train-labels-idx1-ubyte'),'r') ;
y1=fread(f,inf,'uint8');
fclose(f) ;
yt=(double(y1(9:end)'))' ;
yt(yt==0)=10;

f=fopen(fullfile(pwd, 't10k-labels-idx1-ubyte'),'r') ;
y2=fread(f,inf,'uint8');
fclose(f) ;
yts=(double(y2(9:end)'))';
yts(yts==0)=10;
clear f x1 x2 y1 y2;