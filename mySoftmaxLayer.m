classdef mySoftmaxLayer
    properties
        inputSize;
        outputSize;
    end
    methods
        function [obj,output] = forward(obj,input)
            input=input-max(input);
            temp=sum(exp(input));
            output=exp(input)./temp;
        end
        function error=backward(obj,h,y,epoch)
            error=h-y;
        end
    end
end

