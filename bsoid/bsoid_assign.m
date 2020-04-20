function [f_10fps,tsne_feats,grp,llh,bsoid_fig] = bsoid_assign(data,fps,comp,smth_hstry,smth_futr,kclass,it)
%BSOID_ASSIGN     Behavioral Segmentation of Open-Field Behavior based on Gaussian Mixture Models (GMM) in DeepLabCut. 
%                 This unsupervised learning algorithm parses out action groups based on statistically different feature distribution. 
%                 This includes adaptive t-SNE parameters (perplexity, learning rate, and exaggeration) for larger datasets. 
%   
%   [F_10FPS,TSNE_FEATS,GRP,LLH,BSOID_FIG] = BSOID_ASSIGN(DATA,FPS,COMP,SMTH_HSTRY,SMTH_FUTR,KCLASS,IT) outputs segmented behaviors.
%
%   INPUTS:
%   DATA    6-body parts (x,y) matrix outlining the tetrapod animal over time videotaped from the bottom looking up. Rows represents time.
%           Columns 1 & 2 tracks snout; columns 3 to 6 tracks the two front paws; columns 7 to 10 tracks the two hind paws;
%           columns 11 & 12 tracks the base of the tail.
%   FPS    Rounded frame rate, can use VideoReader/ffmpeg to detect the input video fps.
%   COMP    If you desire 1 segmentation built on multiple animals, set this parameter to 1; 
%           Otherwise, set it to 0 for individual segmentation/.csv file. Default is 1. 
%   SMTH_HSTRY    BOXCAR smoothing using number of frames from before. Default ~40ms before.
%   SMTH_FUTR    BOXCAR smoothing using number of frames from after. Default ~40ms after.
%   KCLASS    Maximum number of assigment classes that the Gaussian Mixture Model will try to parse out. Default 50.
%
%   OUTPUTS:
%   F_10FPS    Compiled features that were used to cluster, 10fps temporal resolution.
%   TSNE_FEATS    A 3-dimensional space where 7 features are embedded using t-Distributed Stochastic Neighbor Embedding.                 
%   GRP    Statistically different groups of actions based on data. Output is 10Hz matching FEATS and TSNE_FEATS.
%   LLH    Log-likelihood to see if the EM algorithm converged.
%   BSOID_FIG    3D colored scatter plot showing the how Gaussian Mixture Models grouped data points in the 3D t-SNE space.
%
%   EXAMPLES:
%   clear data;
%   load MsInOpenField.mat
%   [feats,tsne_feats,grp,llh,bsoid_fig] = bsoid_assign(data,60,1);
%
%   clear data;
%   data{1} = [rand(10000,3),randi([1 4],10000,3),randi([-5 5],10000,3),randn(10000,3)];
%   fps = 60;
%   [feats,tsne_feats,grp,llh,bsoid_fig] = bsoid_assign(data,fps,0,2,1,8);
%
%   clear data;
%   data{1} = [rand(1000,12);randn(1000,12);randi(1000,1000,12)];
%   data{2} = [rand(1000,12);randn(1000,12);randi(1000,1000,12)];
%   fps = 30;
%   [feats,tsne_feats,grp,llh,bsoid_fig] = bsoid_assign(data,fps,1);
%   
%   clear data; load fisheriris.mat; 
%   data{1} = [meas,meas,meas]; 
%   fps = 10;
%   [feats,tsne_feats,grp,llh,bsoid_fig] = bsoid_assign(data,fps,0,0,0,3);
%
%   Created by Alexander Hsu, Date: 021920
%   Contact ahsu2@andrew.cmu.edu
    
    if nargin < 2
        error('Please input dataset AND frame rate!')
    end
    if nargin < 3
        comp=1;
    end
    if nargin < 5
        smth_hstry = round(0.05/(1/fps))-1;
        smth_futr = round(0.05/(1/fps))-1;
    end
    if nargin < 6
        kclass = 30;
    end
    if nargin < 7
        it = 20;
    end
    
    fprintf('Obtaining features from dataset... \n');
    for m = 1:length(data) % For each csv file you uploaded.
        %% Obtain features, 2 physical features and 5 time-varying signals
        clear fpd_norm cfp_pt_norm chp_pt_norm sn_pt_norm sn_pt_ang sn_disp pt_disp;
        fpd = data{m}(:,3:4)-data{m}(:,5:6); % Front Paw distance in x,y
        cfp = [mean([data{m}(:,3),data{m}(:,5)],2),mean([data{m}(:,4),data{m}(:,6)],2)]; % Center of front paws x,y
        cfp_pt = [cfp(:,1) - data{m}(:,11), cfp(:,2) - data{m}(:,12)]; % Center of front paw to proximal tail x,y
        chp = [mean([data{m}(:,7),data{m}(:,9)],2),mean([data{m}(:,8),data{m}(:,10)],2)]; % Center of hind paws x,y
        chp_pt = [chp(:,1) - data{m}(:,11), chp(:,2) - data{m}(:,12)]; % Center of hind paws to proximal tail x,y
        sn_pt = [data{m}(:,1) - data{m}(:,11), data{m}(:,2) - data{m}(:,12)]; % Snout to proximal tail x,y
        for i = 1:length(data{m}) % Euclidean distance of x,y, since position means nothing
            fpd_norm(i) = norm(data{m}(i,3:4)-data{m}(i,5:6)); % Front paw R to L euclidean distance    
            cfp_pt_norm(i) = norm(cfp_pt(i,:)); % Center of front paws to proximal tail euclidean distance
            chp_pt_norm(i) = norm(chp_pt(i,:)); % Center of hind paws to proximal tail euclidean distance
            sn_pt_norm(i) = norm(sn_pt(i,:)); % Snout to proximal tail euclidean distance, i.e. body length
        end
        fpd_norm_smth{m} = movmean(fpd_norm,[smth_hstry,smth_futr]); % Reduce label noise
        sn_cfp_norm_smth{m} = movmean(sn_pt_norm-cfp_pt_norm,[smth_hstry,smth_futr]); % Reduce label noise
        sn_chp_norm_smth{m} = movmean(sn_pt_norm-chp_pt_norm,[smth_hstry,smth_futr]); % Reduce label noise
        sn_pt_norm_smth{m} = movmean(sn_pt_norm,[smth_hstry,smth_futr]); % Reduce label noise
        for k = 1:length(data{m})-1 % Velocity and angle over time
            b_3d = [sn_pt(k+1,:),0]; a_3d = [sn_pt(k,:),0]; c = cross(b_3d,a_3d);
            sn_pt_ang(k) = sign(c(3))*180/pi*atan2(norm(c),dot(sn_pt(k,:),sn_pt(k+1,:))); % Body angle, arctan between body  
            sn_disp(k) = norm(data{m}(k+1,1:2)-data{m}(k,1:2)); % Snout displacement over time
            pt_disp(k) = norm(data{m}(k+1,11:12)-data{m}(k,11:12)); % Proximal tail displacement over time
        end 
        sn_pt_ang_smth{m} = movmean(sn_pt_ang,[smth_hstry,smth_futr]); % Reduce label noise
        sn_disp_smth{m} = movmean(sn_disp,[smth_hstry,smth_futr]); % Reduce label noise
        pt_disp_smth{m} = movmean(pt_disp,[smth_hstry,smth_futr]); % Reduce label noise
        %% Collate 7 features. We will reduce the dimensions using the KL-distance in a stochastic process.
        feats{m} = [sn_cfp_norm_smth{m}(:,2:end); sn_chp_norm_smth{m}(:,2:end); fpd_norm_smth{m}(:,2:end); sn_pt_norm_smth{m}(:,2:end); ...
            sn_pt_ang_smth{m}(:,1:end); sn_disp_smth{m}(:,1:end); pt_disp_smth{m}(:,1:end)];
    end
    if comp == 1
        f_10fps = [];
    end
    for n = 1:length(feats)
        feats1 = [];
        for k = fps/10:fps/10:length(feats{n})
            feats1(:,end+1) = [mean(feats{n}(1:4,k-fps/10+1:k),2);sum(feats{n}(5:7,k-fps/10+1:k),2)];
        end
        if comp == 1
            f_10fps = cat(2,f_10fps,feats1);
        else
            f_10fps{n} = feats1;
            if length(f_10fps{n}) <= 10000
                msg = 'Insufficient data, exiting...';
                error(msg)
            else
                p = round(length(f_10fps{n})/300);
            end
            %% For reproducibility
            rng default; tsne_feats = [];
            %% Run t-Distributed Stochastic Neighbor Embedding (t-SNE)
            fprintf('Running individual datasets through t-SNE collapsing the 7 features onto 3 action space coordinates... \n');
            tsne_feats = tsne(f_10fps{n}(:,1:end)','Standardize',true,'Exaggeration',12,'LearnRate',1000,...
            'Perplexity',p,'NumDimensions',3); % Refer to MATLAB tsne function for arguments
            %% Run a Gaussian Mixture Model Expectation Maximization to group the t-SNE clusters
            X = tsne_feats'; k = kclass; 
            [grp{n},model{n},llh{n}] = em_gmm(X,k,it);
            fprintf('TADA! \n');
            cmap = hsv(length(unique(grp{n})));
            figure; hold on;
            for g = 1:length(unique(grp{n}))
                bsoid_fig{n} = scatter3(tsne_feats(grp{n}==g,1),tsne_feats(grp{n}==g,2),tsne_feats(grp{n}==g,3),15,'filled');
                bsoid_fig{n}.MarkerFaceColor = cmap(g,:);
            end
            legend(string(1:length(unique(grp{n}))));
        end
    end 
    if comp == 1
        if length(f_10fps) <= 10000
            msg = 'Insufficient data, exiting...';
            error(msg)
        else
            p = round(length(f_10fps)/300);
        end
        %% For reproducibility
        rng default; tsne_feats = [];
        fprintf('Running the compiled data through t-SNE collapsing the 7 features onto 3 action space coordinates... \n');
        %% Run t-Distributed Stochastic Neighbor Embedding (t-SNE)
        tsne_feats = tsne(f_10fps(:,1:end)','Standardize',true,'Exaggeration',12,'LearnRate',1000,...
            'Perplexity',p,'NumDimensions',3); % Refer to MATLAB tsne function for arguments
        X = tsne_feats'; k = kclass;
        [grp,~,llh,~] = em_gmm(X,k,it); % EM GMM
        fprintf('TADA! \n');
        cmap = hsv(length(unique(grp)));
        figure; hold on;
        for g = 1:length(unique(grp))
            bsoid_fig = scatter3(tsne_feats(grp==g,1),tsne_feats(grp==g,2),tsne_feats(grp==g,3),15,'filled');
            bsoid_fig.MarkerFaceColor = cmap(g,:);
        end
        legend(string(1:length(unique(grp))));
    end
    
return





