function [t,b,b_ex] = action_gif(video,fps,grp_fill,comp,n,n_len,X,filepathout)
%ACTION_GIF     Create a short, randomly sampled video from BSOID grouping. BSOID learns from data, and this is a way for you to subjectively 
%               name the individual groups that were apparently statistically different. 
%   
%   [T,B,B_EX] = ACTION_GIF(VIDEO,GRP,N,N_LEN,FILEPATHOUT) outputs classified behaviors based on DeepLabCut analysis
%   VIDEO.FRAMES.CDATA    The video frame-by-frame output from mmread or videoread.
%   FPS    detect video frame-rate to down sample to 20fps(~50ms) videos snippets
%   GRP_FILL    Statistically different gropus based on unsupervised GMM model fitting of action space data. This is the output from BSOID.
%   COMP If combined for same animal, comp = 1. Default 0;
%   N    Number of sampled video you desire. Default 3. 
%   N_LEN    Lower bound for consecutive frames (20fps, 50ms each) for which the video will be generated from. Default 6 (300ms).
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
%   Created by Alexander Hsu, Date: 070819
%   Contact ahsu2@andrew.cmu.edu
    if nargin < 3
        error('Please input video, video frame-rate, AND vector of group numbers!')
    end
    if nargin < 4
        comp = 0;
    end
    if nargin < 5
        n = 3;
    end
    if nargin < 6
        n_len = 6;
    end
    if nargin < 7
        X = 0.5;
    end
    if nargin < 8
        filepathout = pwd;
    end
    fprintf('Computer is trying to take over the mice community... \nCan human help? \n');
    grp_1 = grp_fill(1:fps/20:end);
    video_all = []; grp_parse_all = [];
    for j = 1:length(video)
        if comp == 1
           video_all = cat(2,video_all,video{j});
           grp_parse_all = cat(2,grp_parse_all,grp_1((j-1)*length(grp_1)/numel(video)+1:(j-1)*length(grp_1)/numel(video)+length(video{j})));
        else
            clear i0 ii t ts act_idx b x b_ex
            grp_parse{j} = grp_1{j}(1:length(video{j}));
            i0 = [true;diff(grp_parse{j}')~=0];
            ii = cumsum(i0);
            t = [0,0;grp_parse{j}(i0)',accumarray(ii,1)];
            ts = cat(2,t(:,1),cumsum(t(:,2)));
            for g = 1:length(unique(grp_parse{j}))
                act_idx{g} = t((t(:,1) == g),:);
                b{g} = cat(2,ts((t(:,1) == g & t(:,2) >= n_len),:),ts((t(:,1) == g & t(:,2) >= n_len),2)-ts(find(t(:,1) == g & t(:,2) >= n_len)-1,2));   
                if numel(b{g}(:,1)) >= n_len
                   x{g} = randsample(b{g}(:,2),n);
                else
                   x{g} = b{g}(:,2);
                end
                if ~isempty(x{g})
                    for k = 1:length(x{g})
                        b_ex{g}(k,:) = [b{g}((b{g}(:,2) == x{g}(k)),2)-b{g}((b{g}(:,2) == x{g}(k)),3)+1,b{g}((b{g}(:,2) == x{g}(k)),2)];
                        images = {};
                        for i = b_ex{g}(k,1):b_ex{g}(k,2)
                            images{end+1} = video{j}(i+1).cdata;
                        end
                        % Create example videos at X times speed. 
                        writerObj = VideoWriter(sprintf('%s%s%s%s%s',filepathout,'/','animal_',num2str(j),...
                            'group',num2str(g),'_example_',num2str(k),'.avi'));
                        writerObj.FrameRate = X*20;
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
                        close(writerObj);
                    end
                end
            end
        end
    end
    if comp == 1
        clear i0 ii t ts act_idx b x b_ex
        i0 = [true;diff(grp_parse_all')~=0];
        ii = cumsum(i0);
        t = [0,0;grp_parse_all(i0)',accumarray(ii,1)];
        ts = cat(2,t(:,1),cumsum(t(:,2)));
        for g = 1:length(unique(grp_parse_all))
            act_idx{g} = t((t(:,1) == g),:);
            b{g} = cat(2,ts((t(:,1) == g & t(:,2) >= n_len),:),ts((t(:,1) == g & t(:,2) >= n_len),2)-ts(find(t(:,1) == g & t(:,2) >= n_len)-1,2));   
            if numel(b{g}(:,1)) >= n_len
                x{g} = randsample(b{g}(:,2),n);
            else
                x{g} = b{g}(:,2);
            end
            if ~isempty(x{g})
                for k = 1:length(x{g})
                    b_ex{g}(k,:) = [b{g}((b{g}(:,2) == x{g}(k)),2)-b{g}((b{g}(:,2) == x{g}(k)),3)+1,b{g}((b{g}(:,2) == x{g}(k)),2)];
                    images = {};
                    for i = b_ex{g}(k,1):b_ex{g}(k,2)
                        images{end+1} = video_all(i+1).cdata;
                    end
                    % Create example videos at X times speed. 
                    writerObj = VideoWriter(sprintf('%s%s%s',filepathout,'/','group',num2str(g),'_example_',num2str(k),'.avi'));
                    writerObj.FrameRate = X*20;
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
                    close(writerObj);
                end
            end
        end
    end

return