function [t,B,b_ex] = action_gif2(PNGpath,grp,n,n_len,X,filepathOutResults)
%ACTION_GIF     Create a short, randomly sampled video from BSOID grouping. BSOID learns from data, and this is a way for you to subjectively 
%               name the individual groups that were apparently statistically different. 
%   
%   [T,B,B_EX] = ACTION_GIF(VIDEO,GRP,N,N_LEN,FILEPATHOUT) outputs classified behaviors based on DeepLabCut analysis
%   VIDEO.FRAMES.CDATA    The video frame-by-frame output from mmread or videoread.
%   FPS    detect video frame-rate to down sample to 20fps(~50ms) videos snippets
%   GRP_FILL    Statistically different gropus based on unsupervised GMM model fitting of action space data. This is the output from BSOID.
%   COMP If combined for same animal, comp = 1. Default 0;
%   N    Number of sampled video you desire. Default 3. 
%   N_LEN    Lower bound for consecutive frames (10fps, 50ms each) for which the video will be generated from. Default 6 (300ms).
%   X    X times speed. Default 0.5.
%   FILEPATHOUT    Output path to store video. Default current directory.
%
%   T    Compiled features that were used to cluster.
%   B    All behaviors that happened more or equal to the lower bound for consecutive frames.
%   B_EX    N random samples from B. 
%
%   Examples:
%   vidObj = VideoReader('xylophone.mp4'); filepathout = pwd;
%   k = 1;
%   while hasFrame(vidObj)
%       video{1}(k).cdata = readFrame(vidObj);
%       k = k+1;
%   end
%   grp = [ones(1,20),2*ones(1,3),ones(1,17),2*ones(1,20),3*ones(1,5),2*ones(1,35),3*ones(1,15),ones(1,5),3*ones(1,20)];
%   [t,b,b_ex] = action_gif(video,grp,2,10,0.5,filepathout);
%
%
%   Created by Alexander Hsu, Date: 100219
%   Contact ahsu2@andrew.cmu.edu
    if nargin < 2
        error('Please input path to frames, group labels (grp)!')
    end
    if nargin < 3
        n = 3;
    end
    if nargin < 4
        n_len = 6;
    end
    if nargin < 5
        X = 0.5;
    end
    if nargin < 6
        filepathOutResults = pwd;
    end
    fprintf('Computer generating videos... \nCan human tell me what the animal is doing? \n');
    clear i0 ii t ts act_i_frms ieb x ieb_rnd
    i0 = [true;diff(grp')~=0];
    ii = cumsum(i0);
    t = [0,0;grp(i0)',accumarray(ii,1)];
    ts = cat(2,t(:,1),cumsum(t(:,2)));
    for b = 1:length(unique(grp))
        act_i_frms{b} = t(find(t(:,1)==b),:);
        B{b} = cat(2,ts(find(t(:,1)==b & t(:,2)>=n_len),:),ts(find(t(:,1)==b & t(:,2)>=n_len),2)-ts(find(t(:,1)==b & t(:,2)>=n_len)-1,2));   
        if numel(B{b}(:,1)) >= n
            x{b} = randsample(B{b}(:,2),n);
        else
            x{b} = B{b}(:,2);
        end
        for r = 1:length(x{b})
            b_ex{b}(r,:) = [B{b}(find(B{b}(:,2) == x{b}(r)),2)-B{b}(find(B{b}(:,2) == x{b}(r)),3),B{b}(find(B{b}(:,2) == x{b}(r)),2)];
            images = {};
            for i = b_ex{b}(r,1):b_ex{b}(r,2)
                images{end+1} = imread(sprintf('%s%s',PNGpath,'img',num2str(i+1),'.png'));
            end

            % create the video writer with 3 fps
            writerObj = VideoWriter(sprintf('%s%s%s',filepathOutResults,'/','group',num2str(b),'_example_',num2str(r),'.avi'));
            writerObj.FrameRate = X*10;
            % set the frames per image
            secsPerImage = [ones(i,1)];
            % open the video writer
            open(writerObj);
            % write the frames to the video
            for u=1:length(images)
                % convert the image to a frame
                frame = im2frame(images{u});
                for v=1:secsPerImage(u) 
                    writeVideo(writerObj, frame);
                end
            end
            % close the writer object
            close(writerObj);
        end
    end




return