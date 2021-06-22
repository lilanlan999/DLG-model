clc;clear;
%%  读取各种特征  
%读取影像组学的特征
% load hc_fmri
% load scd_fmri
load HC_gwas
load MCI_gwas
load AD_gwas
load test_hcad
load test_admci
load test_hcmci
%%
% NC = [hc(1:67,:);hc(69:71,:);hc(73,:);hc(75:77,:)];
NC = MCI_gwas;
SCD =AD_gwas;
M=100;%重复试验次数
Kfold = 5;%5折交叉验证

features=[NC;SCD];
num_NC=size(NC,1);
num_SCD=size(SCD,1);
num = num_NC + num_SCD; %总的样本数
labels=[zeros(num_NC,1);ones(num_SCD,1)];
num_features = size(SCD,2);  %每个样本提取的影像组学的特征数
a_index = zeros(500,200);
 
test_data_feature = test_admci(:,1:2);
test_data_label = test_admci(:,3);
num_test_subjects = size(test_data_feature,1);
subjects_test= randperm(num_test_subjects);
feature_test = test_data_feature(subjects_test,1:2);
True_test_label = test_data_label(subjects_test);
% ACC_train = zeros(M*Kfold,3); %模型一在测试集中的分类的准确率，敏感性，特异性
% ACC_test = zeros(M*Kfold,3); %模型一在训练集中的分类的准确率，敏感性，特异性
%  for i=1:num_features
%     [Train_data(:,i),beta_new] = line_regression_covariate_1(Train_data(:,i),BaseInfo_new);
%     [Test_data(:,i),beta_old] = line_regression_covariate_1(Test_data(:,i),BaseInfo_old);
% end
for ind = 1:M
    disp(['ind------->',num2str(ind),'     (',num2str(M) ' times in total)'])
    [m,N] = size(features);
    indices = crossvalind('Kfold',m, Kfold);
    for i = 1:Kfold
        test_index = (indices == i);
        train_index = ~test_index;
        
        Kfold_Train_data  = features(train_index,:);
        Kfold_Train_label = labels(train_index,:);
        Kfold_Test_data = features(test_index,:);
        Kfold_Test_label = labels(test_index,:);
         %%  特征筛选--t检验
%         r1=Kfold_Train_data((Kfold_Train_label==0),:);
%         r2=Kfold_Train_data((Kfold_Train_label==1),:);
% %         for j= 1: N
% %             [H(j,:),P(j,:)]=ttest2(r1(:,j),r2(:,j));
% %         end
%         [H,P] = ttest2(r1,r2);
%         Index_SigDiff=find(P <= 0.05)';      %列向量(每一种feature)对应的index
%         Ttest_Train_data=Kfold_Train_data(:,Index_SigDiff);
% %         Ttest_Valid_data=Kfold_Valid_data(:,Index_SigDiff);
%         Ttest_Train_num=size(Ttest_Train_data,2);%the number of features after t-test
%          %%    特征筛选----特征自相关
%         corr_index = corr(Ttest_Train_data);
%         deleteindex = findCorrelation(corr_index,0.8);         %去除平均相关性大于0.8的特征列
%         Corr_Train_data = Ttest_Train_data;
%         Corr_Train_data(:,deleteindex)=[];%经过自相关筛选后的特征值                     
%         Corr_Train_num = setdiff(1:Ttest_Train_num,deleteindex);%筛选后留下的个数索引
% 
%         Corr_Train_index=Index_SigDiff(Corr_Train_num,:);%经过自相关后留下的相对于总体特征值中的索引
%         num_a = size(Corr_Train_index,1);
%         a_index (1:num_a,i + Kfold*(ind-1)) = Corr_Train_index;
%         
%         %%   fisher score
%           [features_rank] = FScoreCalculate(Corr_Train_data,Kfold_Train_label);
%            %%    
%          Final_Fea_index=Corr_Train_index(features_rank.DescendList);
%          Final_Fea_Train=Kfold_Train_data(:,Final_Fea_index);
%          Final_Fea_Valid=Kfold_Test_data(:,Final_Fea_index);
%          Final_Fea_Test=feature_test(:,Final_Fea_index);
%          [Final_Fea_Train, settings] = mapminmax( Final_Fea_Train',-1,1);%对每一种特征进行归一化，该函数对行归一化，所以需要转置
%          Final_Fea_Train= Final_Fea_Train';
% 
%          Final_Fea_Valid = mapminmax('apply', Final_Fea_Valid', settings);
%          Final_Fea_Valid = Final_Fea_Valid';
% %             
%          Final_Fea_Test = mapminmax('apply', Final_Fea_Test', settings);
%          Final_Fea_Test = Final_Fea_Test';             
                   %%
% % % %            分类模型 SVM         
        cmd = [' -s 0 -b 1 -t 0 -c 2^(15) '];
        model = svmtrain(Kfold_Train_label, Kfold_Train_data,cmd);
%         [predict_valid, accuracy_valid, score_valid] = svmpredict( Kfold_Test_label,Final_Fea_Valid, model,'-b 1');
%         [predict_test, accuracy_test, score] = svmpredict(True_test_label, Final_Fea_Test, model,'-b 1');
        [predict_valid, accuracy_valid, score_valid] = svmpredict( Kfold_Test_label,Kfold_Test_data, model,'-b 1');
        [predict_test, accuracy_test, score] = svmpredict(True_test_label,feature_test, model,'-b 1');
        [Acc_vaild, Sen_valid, Spe_valid] = cal_Acc_Sens_Spec( Kfold_Test_label, predict_valid);
        [Acc_test, Sen_test, Spe_test] = cal_Acc_Sens_Spec(True_test_label, predict_test);
        ACC(i + Kfold*(ind-1)) = Acc_test;
        SEN(i + Kfold*(ind-1)) = Sen_test;
        SPE(i + Kfold*(ind-1)) = Spe_test;
        
        SCORE(:,i) = score(:,2); 
        auc_lable(:,i) = True_test_label;
        [X2,Y2,~,AUC1] = perfcurve(True_test_label,predict_test,1);
         AUC(i + Kfold*(ind-1)) = AUC1;

         
            
    end  
end
        [mean(ACC) mean(SEN) mean(SPE) mean(AUC)]
        [std(ACC) std(SEN) std(SPE)]
        sco = SCORE(:); 
        auc_lable =auc_lable(:);
        auc = plot_roc(sco,auc_lable) ;
        hold on

        
       