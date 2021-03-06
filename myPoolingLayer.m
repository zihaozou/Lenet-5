classdef myPoolingLayer
    %MYPOOLINGLAYER 此处显示有关此类的摘要
    %   此处显示详细说明
    
    properties
        inputSize;
        outputSize;
        x;
        M;
    end
    
    methods
        function obj=myPoolingLayer(inputSize)
            obj.inputSize=inputSize;
            obj.outputSize=[inputSize(1)/2 inputSize(2)/2 inputSize(3)];
        end
        
        function [obj,output] = forward(obj,input)
            obj.x=input;
            [hi,wi,numIn,obj.M]=size(input);
            output=zeros(hi/2,wi/2,numIn,obj.M);
            [ho,wo]=size(output,[1 2]);
            for m=1:obj.M
                for c=1:numIn
                    for h=1:ho
                        for w=1:wo
                            output(h,w,c,m)=(input(h*2-1,w*2-1,c,m)+...
                                input(h*2,w*2-1,c,m)+...
                                input(h*2-1,w*2,c,m)+...
                                input(h*2,w*2,c,m))/4;
                        end
                    end
                end
            end
            
        end
        
        function [obj,error]=backward(obj,preError,epoch)
            [hi,wi,numIn,~]=size(preError);
            error=zeros(hi*2,wi*2,numIn,obj.M);
            preError=preError/4;
            error(1:2:hi*2-1,1:2:wi*2-1,:,:)=preError;
            error(2:2:hi*2,2:2:wi*2,:,:)=preError;
            error(1:2:hi*2-1,2:2:wi*2,:,:)=preError;
            error(2:2:hi*2,1:2:wi*2-1,:,:)=preError;
        end
    end
end

