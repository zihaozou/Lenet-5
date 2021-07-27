function g = reLuGradient(z)
%RELUGRADIENT 此处显示有关此函数的摘要
%   此处显示详细说明
g=zeros(size(z));
g(z<0)=0;
g(z>=0)=1;
end

