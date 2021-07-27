classdef myConvLayer
    %MYCONVLAYER 此处显示有关此类的摘要
    %   此处显示详细说明
    
    properties
        alpha;
        kernel;
        bias;
        x;
        inputSize;
        outputSize;
    end
    
    methods
        function obj = myConvLayer(inputSize,kernelSize,alpha)
            obj.inputSize=inputSize;
            obj.kernel = randn(kernelSize(1),kernelSize(2),inputSize(3),kernelSize(3))*sqrt(2/prod(inputSize));
            obj.bias=ones(1,kernelSize(3));
            obj.alpha=alpha;
            obj.outputSize=[inputSize(1)-kernelSize(1)+1 inputSize(2)-kernelSize(2)+1 kernelSize(3)];
        end
        
        function [obj,output] = forward(obj,input)
            assert(numel(input)==prod(obj.inputSize,'all'),"输入尺寸不符合要求");
            if ~isequal(size(input),obj.inputSize)
                input=reshape(input,obj.inputSize);
            end
            obj.x=input;
            [~,~,numIn,numOut]=size(obj.kernel);
            output=zeros(obj.outputSize);
            for c=1:numOut
                for i=1:numIn
                    output(:,:,c)=output(:,:,c)+conv2(input(:,:,i),obj.kernel(:,:,i,c),'valid');
                end
                output(:,:,c)=output(:,:,c)+obj.bias(c);
            end
        end
        function [obj,error] = backward(obj,preError)
            assert(numel(preError)==prod(obj.outputSize,'all'),"输入尺寸不符合要求");
            if ~isequal(size(preError),obj.outputSize)
                preError=reshape(preError,obj.outputSize);
            end
            [~,~,numIN,numOut]=size(obj.kernel);
            dw=zeros(size(obj.kernel));
            error=zeros(obj.inputSize);
            for c=1:numOut
                for i=1:numIN
                    dw(:,:,i,c)=conv2(obj.x(:,:,i),preError(:,:,c),'valid');
                    error(:,:,i)=error(:,:,i)+conv2(rot90(obj.kernel(:,:,i,c),2),preError(:,:,c),'full');
                end
            end
            db=reshape(sum(preError,[1 2]),1,numOut);
            obj.kernel=obj.kernel-obj.alpha*dw;
            obj.bias=obj.bias-obj.alpha*db;
        end
        
        function obj=changeAlpha(obj,newAlpha)
            obj.alpha=newAlpha;
        end
    end
end

