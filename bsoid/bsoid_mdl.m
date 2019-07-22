function [OF_mdl,CV_amean,CV_asem,acc_fig] = bsoid_mdl(f_10fps,grp,hldout,cv_it,btchsz)
%BSOID_MDL     Build a SVM classifier for open field behavior based on users own dataset.
%   
%   [OF_MDL,CV_AMEAN,CV_ASEM] = BSOID_MDL(F_10FPS,GRP) outputs the open field model based on the features and the grouping labels from bsoid_gmm.
%
%   INPUTS:
%   F_10FPS    Compiled features that were used to cluster, 10fps temporal resolution.
%   GRP    Statistically different groups of actions based on data. Output is 10Hz.
%   HLDOUT    Percentage of data randomly held out for test. Default is 0.20 (80/20 training/testing).
%   CV_IT    Number of times to run cross-validation on. Default is 100.
%   BTCHSZ    Batch size for randsampling, make sure the hold out data is <= CV_IT*BTCHSZ. Default is 200.
%   
%   OUTPUTS:
%   OF_MDL    Support Vector Machine Classifier Model.
%   CV_AMEAN    Cross-validated accuracy mean.
%   CV_ASEM    Cross-validated accuracy standard error fo the mean.
%   ACC_FIG    Accuracy box plot figure showing all the cross validated data points.
%
%   EXAMPLES:
%   load feats
%   [Mdl,CVacc] = bsoid_mdl(f_10fps,grp);
%
%   Created by Alexander Hsu, Date: 071919
%   Contact ahsu2@andrew.cmu.edu
    
    if nargin < 2
        error('Please input feature matrix and the grouping labels!')
    end
    if nargin < 3
        hldout = 0.2;
    end
    if nargin < 4
        cv_it = 100;
    end
    if nargin < 5
        btchsz = 200;
    end
    fprintf('Training an SVM classifier (kernel trick: Gaussian kernel function)... \n');
    f_10fps_T = f_10fps';
    grp_T = grp';
    classOrder = unique(grp_T);
    rng(1); % For reproducibility
	t = templateSVM('Standardize',true,'KernelFunction','gaussian');
    PMdl = fitcecoc(f_10fps_T,grp_T,'Holdout',hldout,'Learners',t,'ClassNames',classOrder);
    OF_mdl = PMdl.Trained{1};           % Extract trained, compact classifier
    testInds = test(PMdl.Partition);  % Extract the test indices
    feat_test = f_10fps_T(testInds,:);
    label_test = grp_T(testInds,:);
    labels = predict(OF_mdl,feat_test);
    idx_samp = randsample(sum(testInds),cv_it*btchsz);
    for it = 1:cv_it
        ACC(it) = length(find((label_test(idx_samp((it-1)*btchsz+1:it*btchsz))-labels(idx_samp((it-1)*btchsz+1:it*btchsz)))==0))...
            /length(labels(idx_samp((it-1)*btchsz+1:it*btchsz)));
    end
    CV_amean = mean(ACC);
    CV_asem = std(ACC)/sqrt(numel(ACC));
    figure; ax = axes;
    acc_fig = boxplot(100*(ACC),'Color','k'); set(acc_fig,{'linew'},{4}); 
    xticklabels('SVM');ylabel('Accuracy'); ylim([min(100*ACC)-2,max(100*ACC)+2]); 
    hold on; scatter(ones(size(ACC)).*(1+(rand(size(ACC))-0.5)/10),100*(ACC),'r','filled');
    title(sprintf('%s%s%s','Model Performance on ',num2str(100*hldout),'% Test')); ytickformat(ax, '%g%%');
    
return