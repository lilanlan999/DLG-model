[X3,Y3,~,AUC2] = perfcurve(True_test_label_hcmci,SCORE_hcmci(:,1),1); 
figure(1)
set(gcf,'Position',[100 100 550 460]);
set(gcf,'color','w')
p3 = plot([0 1],[0 1],'--k');hold on
p1 = plot(hc_mci_18(:,1),hc_mci_18(:,2),'-go','LineWidth',1.5,'MarkerSize',1);hold on
p11 = plot(hc_mci_34(:,1),hc_mci_34(:,2),'-bo','LineWidth',1.5,'MarkerSize',1);hold on
p12 = plot(hc_mci_cnn(:,1),hc_mci_cnn(:,2),'-mo','LineWidth',1.5,'MarkerSize',1);hold on
p13 = plot(X3,Y3,'-ro','LineWidth',1.5,'MarkerSize',1);hold on
set(gca,'Linewidth',1);
set(gca,'xtick',[0:0.2:1.0],'FontSize',16)
set(gca,'ytick',[0:0.2:1.0],'FontSize',16)
set(gca,'tickdir','out');
xlabel('False Positive Rate','FontSize',17);
ylabel('True Positive Rate','FontSize',17);
title('ROC curve of between HC and MCI groups','FontSize',17)
% legend(p13,'GWAS(AUC=0.610)','Location','SouthEast');
% legend(p12,'CNN(AUC=0.852)','Location','SouthEast');
% legend(p1,'ResNet18(AUC=0.966)','Location','SouthEast');
% legend(p11,'ResNet34(AUC=0.986)','Location','SouthEast');
h = legend([p13 p12 p1 p11],'GWAS(AUC=0.610)','CNN(AUC=0.852)','ResNet18(AUC=0.966)','ResNet34(AUC=0.986)','Location','SouthEast');
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