classdef mySoftmaxLayer
    methods
        function [obj,output] = forward(obj,input)
            input=input-max(input);
            temp=sum(exp(input),'all');
            output=exp(input)./temp;
        end
        function error=backward(obj,h,y)
            error=h-y;
        end
    end
end

