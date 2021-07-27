classdef myReLuLayer
    properties
        x
    end
    methods        
        function [obj,output] = forward(obj,input)
            obj.x=input;
            output=max(0,input);
        end
        function [obj,error]=backward(obj,preError)
            error=zeros(size(obj.x));
            error(obj.x>0)=1;
            error=error.*preError;
        end
    end
end

