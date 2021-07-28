classdef myShapeFormatterLayer
    %MYSHAPEFORMATTERLAYER 此处显示有关此类的摘要
    %   此处显示详细说明
    
    properties
        inputSize;
        outputSize;
        M;
    end
    
    methods
        function obj = myShapeFormatterLayer(inputSize,outputSize)
            obj.inputSize=inputSize;
            obj.outputSize=outputSize;
        end
        
        function [obj,output] = forward(obj,input)
            obj.M=size(input, 4);
            output=reshape(input,[obj.outputSize obj.M]);
        end
        
        function [obj,error] = backward(obj,preError,epoch)
            error=reshape(preError,[obj.inputSize obj.M]);
        end
    end
end

