classdef myFullConnLayer

    
    properties
        alpha;
        weight;
        bias;
        x;
    end
    
    methods
        function obj = myFullConnLayer(weightSize,alpha)
            obj.weight=randn(weightSize)*sqrt(2/weightSize(1));
            obj.bias=ones(1,weightSize(2));
            obj.alpha=alpha;
        end
        
        function [obj,output] = forward(obj,input)
            assert(numel(input)==size(obj.weight,1),"输入元素个数不符合要求");
            if ~isequal(size(input),[1 size(obj.weight,1)])
                input=reshape(input,[1 size(obj.weight,1)]);
            end
            obj.x=input;
            output=input*obj.weight+obj.bias;
            
        end
        function [obj,error]=backward(obj,preError)
            assert(numel(preError)==size(obj.weight,2),"输入元素个数不符合要求");
            if ~isequal(size(preError),[1 size(obj.weight,2)])
                preError=reshape(preError,[1 size(obj.weight,2)]);
            end
            dw=obj.x'*preError;
            db=preError;
            error=preError*obj.weight';
            obj.weight=obj.weight-obj.alpha*dw;
            obj.bias=obj.bias-obj.alpha*db;
        end
        function obj=changeAlpha(obj,newAlpha)
            obj.alpha=newAlpha;
        end
    end
end

