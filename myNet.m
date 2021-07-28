classdef myNet
    %MYNET 此处显示有关此类的摘要
    %   此处显示详细说明
    
    properties
        layers={};
    end
    
    methods
        function obj = addLayer(obj,layer)
            if isa(layer,'myReLuLayer') || isa(layer,'mySoftmaxLayer')
                assert(~isempty(obj.layers),"不能将myReLuLayer或mySoftmaxLayer置于网络第一层\n");
                layer.inputSize=obj.layers{end}.outputSize;
                layer.outputSize=obj.layers{end}.outputSize;
            end
            if ~isempty(obj.layers)
                assert(prod(obj.layers{end}.outputSize)==prod(layer.inputSize),"与上一层元素个数不一致\n");
                if ~isequal(obj.layers{end}.outputSize,layer.inputSize)
                    obj.layers{end+1}=myShapeFormatterLayer(obj.layers{end}.outputSize,layer.inputSize);
                end
            end
            
            obj.layers{end+1}=layer;
        end
        function [obj,J,h]=forwardPropogation(obj,X,y,M)
            assert(isa(obj.layers{end},'mySoftmaxLayer'),"最后一层必须为softmax层");
            h=X;
            for l=1:length(obj.layers)
                [obj.layers{l},h]=forward(obj.layers{l},h);
            end
            h=h+1e-10;
            J=sum(-(y.*log(h)+(1-y).*log(1-h)),'all')/M;
        end
        function obj=backwardPorpogation(obj,h,y,epoch)
            assert(isa(obj.layers{end},'mySoftmaxLayer'),"最后一层必须为softmax层");
            error=backward(obj.layers{end},h,y,epoch);
            for l=length(obj.layers)-1:-1:1
                [obj.layers{l},error]=backward(obj.layers{l},error,epoch);
            end
        end
        
        function [maxh,i]=predict(obj,X)
            h=X;
            for l=1:length(obj.layers)
                [~,h]=forward(obj.layers{l},h);
            end
            [maxh,i]=max(h,[],1);
        end
        
        function obj=changeAlpha(obj,newAlpha)
            for l=1:length(obj.layers)
                if ismethod(obj.layers{l},'changeAlpha')
                    obj.layers{l}=changeAlpha(obj.layers{l},newAlpha);
                end
            end
        end
    end
end

