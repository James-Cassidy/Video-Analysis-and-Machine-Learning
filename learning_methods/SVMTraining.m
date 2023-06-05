function model = SVMTraining(images, labels)

    % Binary classification
    model.type='binary';
    
    % SVM software requires labels -1 or 1 for the binary problem
    labels(labels==0)=-1;

    % Initilaise and setup SVM parameters
    lambda = 1e-20;  
    C = Inf;
     sigmakernel=10;
     K=svmkernel(images,'gaussian',sigmakernel);
     kerneloption.matrix=K;
     kernel='numerical';

    % Calculate the support vectors
    [xsup,w,w0,pos,tps,alpha] = svmclass(images,labels,C,lambda,kernel,kerneloption,1); 

    % Create a structure encapsulating all the variables composing the model
    model.xsup = xsup;
    model.w = w;
    model.w0 = w0;

    model.param.sigmakernel=sigmakernel;
    model.param.kernel=kernel;
    
end