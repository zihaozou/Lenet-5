function sm = softmax(z)
%SOFTMAX 此处显示有关此函数的摘要
%   此处显示详细说明
sm=exp(z)./(sum(exp(z),'all'));
end

