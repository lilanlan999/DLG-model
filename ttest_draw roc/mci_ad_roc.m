[X3,Y3,~,AUC1] = perfcurve(True_test_label_mciad,SCORE_mciad(:,5),1); 
figure(1)
set(gcf,'Position',[100 100 550 460]);
set(gcf,'color','w')
p3 = plot([0 1],[0 1],'--k');hold on
p1 = plot(mci_ad_18(:,1),mci_ad_18(:,2),'-go','LineWidth',1.5,'MarkerSize',1);hold on
p11 = plot(mci_ad_34(:,1),mci_ad_34(:,2),'-bo','LineWidth',1.5,'MarkerSize',1);hold on
p12 = plot(mci_ad_cnn(:,1),mci_ad_cnn(:,2),'-mo','LineWidth',1.5,'MarkerSize',1);hold on
p13 = plot(X3,Y3,'-ro','LineWidth',1.5,'MarkerSize',1);hold on
set(gca,'Linewidth',1);
set(gca,'xtick',[0:0.2:1.0],'FontSize',16)
set(gca,'ytick',[0:0.2:1.0],'FontSize',16)
set(gca,'tickdir','out');
xlabel('False Positive Rate','FontSize',17);
ylabel('True Positive Rate','FontSize',17);
title('ROC curve between AD and MCI groups','FontSize',17)
% legend(p13,'GWAS(AUC=0.569)','Location','SouthEast');
% legend(p12,'CNN(AUC=0.840)','Location','SouthEast');
% legend(p1,'ResNet18(AUC=0.972)','Location','SouthEast');
% legend(p11,'ResNet34(AUC=0.981)','Location','SouthEast');
h = legend([p13 p12 p1 p11],'GWAS(AUC=0.569)','CNN(AUC=0.840)','ResNet18(AUC=0.972)','ResNet34(AUC=0.981)','Location','SouthEast');
set(h,'FontSize',16)
%legend([p1 p2],'MCE:AUC = 0.875','PES :AUC = 0.781');
box off
ax2 = axes('Position',get(gca,'Position'),...
           'XAxisLocation','top',...
           'YAxisLocation','right',...
           'Color','none',...
           'XColor','k','YColor','k');
set(ax2,'YTick', []);
set(ax2,'XTick', []);
set(ax2,'Linewidth',1);
box on