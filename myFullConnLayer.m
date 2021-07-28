classdef myFullConnLayer

    
    properties
        inputSize;
        outputSize;
        alpha=0.001;
        beta1=0.9;
        beta2=0.999;
        epsilon=1e-8;
        weight;
        bias;
        x;
        M;
        momentumW=0;
        momentumB=0;
        volocityW=0;
        volocityB=0;
    end
    
    methods
        function obj = myFullConnLayer(weightSize)
            obj.weight=randn(weightSize)*sqrt(2/weightSize(2));
            obj.bias=ones(weightSize(1),1);
            obj.inputSize=weightSize(2);
            obj.outputSize=weightSize(1);
        end
        
        function [obj,output] = forward(obj,input)
            obj.x=input;
            obj.M=size(input,2);
            output=obj.weight*input+obj.bias;
        end
        function [obj,error]=backward(obj,preError,epoch)
            dw=(preError*obj.x')/obj.M;
            db=mean(preError,2);
            error=obj.weight'*preError;
            obj.momentumW=obj.beta1*obj.momentumW+(1-obj.beta1)*dw;
            obj.momentumB=obj.beta1*obj.momentumB+(1-obj.beta1)*db;
            obj.volocityW=obj.beta2*obj.volocityW+(1-obj.beta2)*(dw.^2);
            obj.volocityB=obj.beta2*obj.volocityB+(1-obj.beta2)*(db.^2);
            obj.weight=obj.weight-(obj.alpha*sqrt(1-obj.beta2.^epoch)/...
                (1-obj.beta1.^epoch))*obj.momentumW./(sqrt(obj.volocityW)+...
                obj.epsilon);
            obj.bias=obj.bias-(obj.alpha*sqrt(1-obj.beta2.^epoch)/...
                (1-obj.beta1.^epoch))*obj.momentumB./(sqrt(obj.volocityB)+...
                obj.epsilon);
        end
        function obj=changeAlpha(obj,newAlpha)
            obj.alpha=newAlpha;
        end
    end
end

