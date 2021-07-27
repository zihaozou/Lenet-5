function avg = pooling(input)
%POOLING 此处显示有关此函数的摘要
%   此处显示详细说明
avg=zeros(size(input,1)/2,size(input,2)/2,size(input,3));
for k=1:size(input,3)
    for i=1:2:size(input,1)-1
        for j=1:2:size(input,2)-1
            avg(floor(i/2)+1,floor(j/2)+1,k)=((input(i,j,k)+input(i+1,j,k)...
                +input(i,j+1,k)+input(i+1,j+1,k))/4);
        end
    end
end
end

