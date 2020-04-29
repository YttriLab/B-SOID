function [bout_kin_grpn,cdf_fig] = kin_analysis(bout_kin,group_num)

    %% Concatenate the individual bouts into one vector, per group
    bout_kin_grpn = {};
    for i = 1:length(bout_kin{group_num})
        for n = 1:length(bout_kin{group_num}{i})
            if n == 1
                bout_kin_grpn{i} = bout_kin{group_num}{i}{1};
            else
                bout_kin_grpn{i} = cat(2,bout_kin_grpn{i},bout_kin{group_num}{i}{n});
            end
        end
    end
    %% Plot with legends
    Legend=cell(length(bout_kin_grpn),1);
    cmap = hsv(length(bout_kin_grpn));
    figure; hold on;
    for j = 1:length(bout_kin_grpn)
        Legend{j} = ['Session ',num2str(j)];
        cdf_fig{j} = cdfplot(bout_kin_grpn{j});
        cdf_fig{j}.LineWidth = 2.5; cdf_fig{j}.Color = cmap(j,:);
        kin_min(j) = min(bout_kin_grpn{j});
        kin_max(j) = max(bout_kin_grpn{j});
    end
    legend(Legend,'Location','Northwest');
    xlim([min(kin_min),max(kin_max)*0.9]);
    
    
return



