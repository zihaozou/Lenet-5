classdef myConvLayer
    %MYCONVLAYER 此处显示有关此类的摘要
    %   此处显示详细说明
    
    properties
        alpha=0.001;
        beta1=0.9;
        beta2=0.999;
        epsilon=1e-8;
        kernel;
        bias;
        x;
        inputSize;
        outputSize;
        M;
        momentumW=0;
        momentumB=0;
        volocityW=0;
        volocityB=0;
    end
    
    methods
        function obj = myConvLayer(inputSize,kernelSize)
            obj.inputSize=inputSize;
            obj.kernel = randn(kernelSize(1),kernelSize(2),inputSize(3)...
                ,kernelSize(3))*sqrt(2/prod(inputSize));
            obj.bias=ones(1,kernelSize(3));
            obj.outputSize=[inputSize(1)-kernelSize(1)+1 inputSize(2)-kernelSize(2)+1 kernelSize(3)];
        end
        
        function [obj,output] = forward(obj,input)
            obj.x=input;
            obj.M=size(input,4);
            [~,~,numIn,numOut]=size(obj.kernel);
            output=zeros([obj.outputSize obj.M]);
            for m=1:obj.M
                for c=1:numOut
                    for i=1:numIn
                        output(:,:,c,m)=output(:,:,c,m)+conv2(input(:,:,i,m),...
                            obj.kernel(:,:,i,c),'valid');
                    end
                    output(:,:,c,m)=output(:,:,c,m)+obj.bias(c);
                end
            end
        end
        function [obj,error] = backward(obj,preError,epoch)
            [~,~,numIN,numOut]=size(obj.kernel);
            dw=zeros(size(obj.kernel));
            error=zeros([obj.inputSize obj.M]);
            for m=1:obj.M
                for c=1:numOut
                    for i=1:numIN
                        dw(:,:,i,c)=dw(:,:,i,c)+conv2(obj.x(:,:,i,m),preError(:,:,c,m),'valid');
                        error(:,:,i,m)=error(:,:,i,m)+conv2(rot90(...
                            obj.kernel(:,:,i,c),2),preError(:,:,c,m),'full');
                    end
                end
            end
            dw=dw./obj.M;
            db=reshape(mean(sum(preError,[1 2]),4),1,numOut);
            obj.momentumW=obj.beta1*obj.momentumW+(1-obj.beta1)*dw;
            obj.momentumB=obj.beta1*obj.momentumB+(1-obj.beta1)*db;
            obj.volocityW=obj.beta2*obj.volocityW+(1-obj.beta2)*(dw.^2);
            obj.volocityB=obj.beta2*obj.volocityB+(1-obj.beta2)*(db.^2);
            obj.kernel=obj.kernel-(obj.alpha*sqrt(1-obj.beta2.^epoch)./...
                (1-obj.beta1.^epoch))*obj.momentumW./(sqrt(obj.volocityW)+...
                obj.epsilon);
            obj.bias=obj.bias-(obj.alpha*sqrt(1-obj.beta2.^epoch)./...
                (1-obj.beta1.^epoch))*obj.momentumB./(sqrt(obj.volocityB)+...
                obj.epsilon);
        end
        function obj=changeAlpha(obj,newAlpha)
            obj.alpha=newAlpha;
        end
    end
end

