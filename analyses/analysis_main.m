%   This analysis_main script analyzes transition matrix and kinematics using B-SOiD labels and positional csv files.
%   Created by Alexander Hsu, Date: 042920
%   Contact: ahsu2@andrew.cmu.edu

clear all; close all; clc;

%% Create output folder to store results, plots/traces or values.
fprintf('Please select folder to store results'); 
filepathOutResults = uigetdir;
outputfname = '041919_Ctrl_Ms1_';

clear P_all count_all
for sess = 1:length(labels)
    labels_filt = labels{sess};
    % if you want to remove any behaviors that are < 200ms in duration, you can run this
    stchall = find(diff(labels_filt)~=0);
    for i = 1:length(stchall)-1
        if stchall(i+1) - stchall(i) < 3
            labels_filt(stchall(i):stchall(i+1)) = labels_filt(stchall(i));
        end
    end
    labels_filt(1) = labels_filt(2); % Fill back the first one
    labels_filt_all{sess} = labels_filt;
    %% Tallying the number of transitions per frame
    binEdgesall = 1:length(labels_filt); [~,statesall] = histc(nonzeros(labels_filt),binEdgesall); % Histc just asks which bin each pose belongs to
    binTicksall = binEdgesall(1:end-1); % Don't need the last bin edge
    ActMatall = zeros(length(unique(labels_filt))); fromBinNosall = statesall(1:end-1); nextBinNosall = statesall(2:end);
    for i = 1:length(labels_filt)-1
        ActMatall(fromBinNosall(i),nextBinNosall(i)) = ActMatall(fromBinNosall(i),nextBinNosall(i)) + 1; % if goes from 2 -> 3, add 1 to row 2, col 3
    end
    for j = 1:length(unique(labels{sess}))
        P_all{sess}(j,:) = ActMatall(j,:)/sum(ActMatall(j,:)); % probability
        count_all{sess}(j,:) = ActMatall(j,:); % count
    end
end
%% Transition matrix
for i = 1:length(P_all)
    out = P_all{i} - diag(diag(P_all{i})); % remove the diagonal
    tmatall_fig = plot_tmat(out,outputfname,filepathOutResults,hot,'png',300); % plot transition matrix
end
%% Kinematics per group
clear bout_*
for g = 1:length(unique(labels{1}))
    [bout_dur{g},bout_lenS{g},bout_speedS{g},bout_lenRf{g},bout_speedRf{g},bout_lenLf{g},bout_speedLf{g}, ...
        bout_lenRh{g},bout_speedRh{g},bout_lenLh{g},bout_speedLh{g}] = getkin(MsTestingData,labels_filt_all,g,60);
end
%% Analyze by session/animal, and plot cdf
%%% for example, lets look at peak stride speed in right forepaw (bout_speedRf) during locomotion (group 8)
[bout_lenRf_grp10,cdf_fig] = kin_analysis(bout_lenRf,10);
%%% or a for loop for all behavioral groups
for k = 1:length(unique(labels{1}))
    [bout_speedRf_grpall{k},cdf_fig] = kin_analysis(bout_speedRf,k);
end


