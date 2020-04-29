
cmap = hsv(length(unique(grp)));
filepathOutResults = uigetdir;
figure; hold on;
for ii = 1:length(unique(grp))
    subplot(length(unique(grp)),1,ii);
    [n1, x1] = hist(feats(1,grp==ii),[linspace(mean(feats(1,:))-3.1*std(feats(1,:)),mean(feats(1,:))+3.1*std(feats(1,:)),100)]); %// use two-output version of hist to get values
    n_normalized = n1/numel(feats(1,grp==ii)); %// normalize to unit area
    a = area(x1(2:99),n_normalized(2:99),'LineWidth',2,'EdgeColor','k'); 
    a.FaceColor = cmap(ii,:); a.EdgeColor = 'k'; a.FaceAlpha = 0.8;
    xlim([mean(feats(1,:))-3*std(feats(1,:)),mean(feats(1,:))+3*std(feats(1,:))]);
    set(gca, 'FontName', 'Arial','Fontsize',24);
    xticks([linspace(mean(feats(1,:))-3.1*std(feats(1,:)),mean(feats(1,:))+3.1*std(feats(1,:)),8)]); 
    xticklabels([round(linspace(mean(feats(1,:))-3.1*std(feats(1,:)),mean(feats(1,:))+3.1*std(feats(1,:)),8),2)]);
    if ii < length(unique(grp))
        set(gca,'xticklabel',{[]},'xtick',[])
    end
end
xlabel('Relative snout to forepaws placement (px)','FontName', 'Arial','Fontsize',52);
set(gcf, 'PaperUnits', 'inches'); set(gcf, 'PaperPosition', [0 0 15 16]); 
print(gcf,sprintf('%s%s%s',filepathOutResults,'/','feat1.png'),sprintf('%s%s','-d','png'),sprintf('%s%s','-r',num2str(300)));

figure; hold on;
for ii = 1:length(unique(grp))
    subplot(length(unique(grp)),1,ii);
    [n1, x1] = hist(feats(2,grp==ii),[linspace(mean(feats(2,:))-3.1*std(feats(2,:)),mean(feats(2,:))+3.1*std(feats(2,:)),100)]); %// use two-output version of hist to get values
    n_normalized = n1/numel(feats(2,grp==ii)); %// normalize to unit area
    a = area(x1(2:99),n_normalized(2:99),'LineWidth',2,'EdgeColor','k'); 
    a.FaceColor = cmap(ii,:); a.EdgeColor = 'k'; a.FaceAlpha = 0.8;
    xlim([mean(feats(2,:))-3*std(feats(2,:)),mean(feats(2,:))+3*std(feats(2,:))]);
    set(gca, 'FontName', 'Arial','Fontsize',24);
    xticks([linspace(mean(feats(2,:))-3.1*std(feats(2,:)),mean(feats(2,:))+3.1*std(feats(2,:)),8)]); 
    xticklabels(round(([linspace(mean(feats(2,:))-3.1*std(feats(2,:)),mean(feats(2,:))+3.1*std(feats(2,:)),8)]),2));
    if ii < length(unique(grp))
        set(gca,'xticklabel',{[]},'xtick',[])
    end
end
xlabel('Relative snout to hind paws placement (px)','FontName', 'Arial','Fontsize',52);
set(gcf, 'PaperUnits', 'inches'); set(gcf, 'PaperPosition', [0 0 15 16]); 
print(gcf,sprintf('%s%s%s',filepathOutResults,'/','feat2.png'),sprintf('%s%s','-d','png'),sprintf('%s%s','-r',num2str(300)));
  
figure; hold on;
for ii = 1:length(unique(grp))
    subplot(length(unique(grp)),1,ii);
    [n1, x1] = hist(feats(3,grp==ii),[linspace(0,mean(feats(3,:))+3.1*std(feats(3,:)),100)]); %// use two-output version of hist to get values
    n_normalized = n1/numel(feats(3,grp==ii)); %// normalize to unit area
    a = area(x1(1:99),n_normalized(1:99),'LineWidth',2,'EdgeColor','k'); 
    a.FaceColor = cmap(ii,:); a.EdgeColor = 'k'; a.FaceAlpha = 0.8;
    xlim([0,mean(feats(3,:))+3*std(feats(3,:))]);
    set(gca, 'FontName', 'Arial','Fontsize',24);
    xticks([linspace(0,mean(feats(3,:))+3.1*std(feats(3,:)),8)]); 
    xticklabels(round(([linspace(0,mean(feats(3,:))+3.1*std(feats(3,:)),8)]),2));
    if ii < length(unique(grp))
        set(gca,'xticklabel',{[]},'xtick',[])
    end
end
xlabel('Inter-forepaw distance (px)','FontName', 'Arial','Fontsize',52);
set(gcf, 'PaperUnits', 'inches'); set(gcf, 'PaperPosition', [0 0 15 16]); 
print(gcf,sprintf('%s%s%s',filepathOutResults,'/','feat3.png'),sprintf('%s%s','-d','png'),sprintf('%s%s','-r',num2str(300)));

figure; hold on;
for ii = 1:length(unique(grp))
    subplot(length(unique(grp)),1,ii);
    [n1, x1] = hist(feats(4,grp==ii),[linspace(0,mean(feats(4,:))+3.1*std(feats(4,:)),100)]); %// use two-output version of hist to get values
    n_normalized = n1/numel(feats(4,grp==ii)); %// normalize to unit area
    a = area(x1(1:99),n_normalized(1:99),'LineWidth',2,'EdgeColor','k'); 
    a.FaceColor = cmap(ii,:); a.EdgeColor = 'k'; a.FaceAlpha = 0.8;
    xlim([0,mean(feats(4,:))+3*std(feats(4,:))]);
    set(gca, 'FontName', 'Arial','Fontsize',24);
    xticks([linspace(0,mean(feats(4,:))+3.1*std(feats(4,:)),8)]); 
    xticklabels(round(([linspace(0,mean(feats(4,:))+3.1*std(feats(4,:)),8)]),2));
    if ii < length(unique(grp))
        set(gca,'xticklabel',{[]},'xtick',[])
    end
end
xlabel('Body length (px)','FontName', 'Arial','Fontsize',52);
set(gcf, 'PaperUnits', 'inches'); set(gcf, 'PaperPosition', [0 0 15 16]); 
print(gcf,sprintf('%s%s%s',filepathOutResults,'/','feat4.png'),sprintf('%s%s','-d','png'),sprintf('%s%s','-r',num2str(300)));

figure; hold on;
for ii = 1:length(unique(grp))
    subplot(length(unique(grp)),1,ii);
    [n1, x1] = hist(feats(5,grp==ii),[linspace(mean(feats(5,:))-3.1*std(feats(5,:)),mean(feats(5,:))+3.1*std(feats(5,:)),100)]); %// use two-output version of hist to get values
    n_normalized = n1/numel(feats(5,grp==ii)); %// normalize to unit area
    a = area(x1(2:99),n_normalized(2:99),'LineWidth',2,'EdgeColor','k'); 
    a.FaceColor = cmap(ii,:); a.EdgeColor = 'k'; a.FaceAlpha = 0.8;
    xlim([mean(feats(5,:))-3*std(feats(5,:)),mean(feats(5,:))+3*std(feats(5,:))]);
    set(gca, 'FontName', 'Arial','Fontsize',24);
    if ii < length(unique(grp))
        set(gca,'xticklabel',{[]},'xtick',[])
    end
end
xlabel(['Body angle (' char(176) ')'],'FontName', 'Arial','Fontsize',52);
set(gcf, 'PaperUnits', 'inches'); set(gcf, 'PaperPosition', [0 0 15 16]); 
print(gcf,sprintf('%s%s%s',filepathOutResults,'/','feat5.png'),sprintf('%s%s','-d','png'),sprintf('%s%s','-r',num2str(300)));
    
figure; hold on;
for ii = 1:length(unique(grp))
    subplot(length(unique(grp)),1,ii);
    [n1, x1] = hist(feats(6,grp==ii),[linspace(0,mean(feats(6,:))+3.1*std(feats(6,:)),100)]); %// use two-output version of hist to get values
    n_normalized = n1/numel(feats(6,grp==ii)); %// normalize to unit area
    a = area(x1(1:99),n_normalized(1:99),'LineWidth',2,'EdgeColor','k'); 
    a.FaceColor = cmap(ii,:); a.EdgeColor = 'k'; a.FaceAlpha = 0.8;
    xlim([0,mean(feats(6,:))+3*std(feats(6,:))]);
    set(gca, 'FontName', 'Arial','Fontsize',24);
    xticks([linspace(0,mean(feats(6,:))+3.1*std(feats(6,:)),8)]); xticklabels(round(([linspace(0,mean(feats(6,:))+3.1*std(feats(6,:)),8)])/2.35126,2));
    if ii < length(unique(grp))
        set(gca,'xticklabel',{[]},'xtick',[])
    end
end
xlabel('Snout displacement (px/frame)','FontName', 'Arial','Fontsize',52);
set(gcf, 'PaperUnits', 'inches'); set(gcf, 'PaperPosition', [0 0 15 16]); 
print(gcf,sprintf('%s%s%s',filepathOutResults,'/','feat6.png'),sprintf('%s%s','-d','png'),sprintf('%s%s','-r',num2str(300)));

figure; hold on;
for ii = 1:length(unique(grp))
    subplot(length(unique(grp)),1,ii);
    [n1, x1] = hist(feats(7,grp==ii),[linspace(0,mean(feats(7,:))+3.1*std(feats(7,:)),100)]); %// use two-output version of hist to get values
    n_normalized = n1/numel(feats(7,grp==ii)); %// normalize to unit area
    a = area(x1(1:99),n_normalized(1:99),'LineWidth',2,'EdgeColor','k'); 
    a.FaceColor = cmap(ii,:); a.EdgeColor = 'k'; a.FaceAlpha = 0.8;
    xlim([0,mean(feats(7,:))+3*std(feats(7,:))]);
    set(gca, 'FontName', 'Arial','Fontsize',24);
    xticks([linspace(0,mean(feats(7,:))+3.1*std(feats(7,:)),8)]); xticklabels(round(([linspace(0,mean(feats(7,:))+3.1*std(feats(7,:)),8)])/2.35126,2));
    if ii < length(unique(grp))
        set(gca,'xticklabel',{[]},'xtick',[])
    end
end
xlabel('Tail-base displacement (px/frame)','FontName', 'Arial','Fontsize',52);
set(gcf, 'PaperUnits', 'inches'); set(gcf, 'PaperPosition', [0 0 15 16]); 
print(gcf,sprintf('%s%s%s',filepathOutResults,'/','feat7.png'),sprintf('%s%s','-d','png'),sprintf('%s%s','-r',num2str(300)));


