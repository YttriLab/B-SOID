function tmat_fig = plot_tmat(P,file,filepathOutResults,tmatcolors,format,dpi)

%PLOT_TMAT     Plot transition matrix based on Markov assumption
%   tmat_fig = plot_mat(P,invert_img,filepathOutResults,tmatcolors,format,dpi) generates transition matrix plot based on Markov assumption
%   P    transition matrix with current state in rows, next state in columns.
%   file    filename of output graph
%   filepathOutResults     specifies output folder.
%   format    image format in string ('')
%   dpi     denotes png quality.
%
%   Created by Alexander Hsu, Date: 051519
%   Contact ahsu2@andrew.cmu.edu

    Y = discretize(P,[0,0.0025,0.005,0.0075,0.01,0.02,0.03,0.05,0.1,1]);
    figure; tmat_fig = imagesc(Y);
    c = colorbar('Ticks',[1:9],'TickLabels',{'< 0.25%','0.25-0.5%','0.5-0.75%','0.75-1%','1-2%','2-3%','3-5%','5-10%','> 10%'},'Color','k');    
    xlab = xlabel('Next frame of action');
    xticks(1:16);
    ylab = ylabel('Current frame of action');
    yticks(1:16);
    colormap(tmatcolors); c.Label.String= 'Probability'; c.Label.Rotation = 270; 
    pos = get(c,'Position');
    c.Label.Position = [6.5 4]; % to change its position
    x1=get(gca,'position'); x=get(c,'Position'); x(3)=0.03; set(c,'Position',x); set(gca,'position',x1);
    set(gca,'TickLength', [0 0]); set(gcf, 'PaperUnits', 'inches'); set(gcf, 'PaperPosition', [0 0 16 12]);
    set(gca, 'FontName', 'Helvetica','Fontsize',28); c.Label.FontSize = 32; xlab.FontSize = 32; ylab.FontSize = 32;
    print(gcf,sprintf('%s%s%s%s%s',filepathOutResults,'/',file(1:end),'TransitionMatrix.',format),sprintf('%s%s','-d',format),sprintf('%s%s','-r',num2str(dpi)));
    
return
    