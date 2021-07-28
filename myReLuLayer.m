classdef myReLuLayer
    properties
        inputSize;
        outputSize;
        x;
    end
    methods
        function [obj,output] = forward(obj,input)
            obj.x=input;
            output=max(0,input);
        end
        function [obj,error]=backward(obj,preError,epoch)
            error=obj.x>0;
            error=error.*preError;
        end
    end
end

