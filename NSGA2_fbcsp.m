clc
clear 
close all

tic

% BCI Competition III - Dataset IVa - 100Hz

%% Loading data from BCI Competition III dataset

% "#tr": number of training (labelled) trials   
% "#te": number of test (unlabelled) trials

% Uncomment the desired subject

% load('data_set_IVa_aa.mat');     % #tr=168   &   #te=112    1=80 , 2=88
% load('true_labels_aa.mat');
% Params.LabeldTrials='168';
% NoLabeldTrials=168;
% Params.UnLabeldTrials='112';
% Params.firstClassTrials='80';
% Params.secondClassTrials='88';


load('data_set_IVa_al.mat');     % #tr=224   &   #te=56     1=112,2=112
load('true_labels_al.mat');
Params.LabeldTrials='224';
NoLabeldTrials=224;
Params.UnLabeldTrials='56';
Params.firstClassTrials='112';
Params.secondClassTrials='112';


% load('data_set_IVa_av.mat');     % #tr=84    &   #te=196
% load('true_labels_av.mat');
% Params.LabeldTrials='84';
% NoLabeldTrials=84;
% Params.UnLabeldTrials='196';
% Params.firstClassTrials='?';
% Params.secondClassTrials='?';

% load('data_set_IVa_aw.mat');     % #tr=56    &   #te=224    1=30 , 2=26
% load('true_labels_aw.mat');
% Params.LabeldTrials='56';
% NoLabeldTrials=56;
% Params.UnLabeldTrials='224';
% Params.firstClassTrials='30';
% Params.secondClassTrials='26';


% load('data_set_IVa_ay.mat');     % #tr=28    &   #te=252
% load('true_labels_ay.mat');
% Params.LabeldTrials='28';
% NoLabeldTrials=28;
% Params.UnLabeldTrials='252';
% Params.firstClassTrials='?';
% Params.secondClassTrials='?';

% 1= right, 2=foot, or 0 for test trials 

cnt = double(cnt)*0.1;        % Microvolt

rng(1); % For reproducibility

%% Parameters
fs = nfo.fs;             % Sampling Rate
Params.SamplingFrequency=nfo.fs;

%% Channel Locations
x = nfo.xpos;
y = nfo.ypos;
% channellabels = nfo.clab;     %cell matrix
% figure
% plot(x,y,'ro','markersize',10,'linewidth',2,'markerfacecolor','k')       %Capital initials for older versions
% grid on
% text(x+0.02,y,channellabels,'fontsize',15,'color','b')
% title('Channel Locations')


%% Preprocessing with Spatial Filters 
type=questdlg('what type of spatial filter do you want to use?','Spatial Filter',...
    'CAR','Low Laplacian','High Laplacian','CAR');

ch = (1:118); 
points = [x';y'];

% figure
[cnt_f] = myspatialfilter(cnt,type,points);    % EEG signal after Preprocessing with spatial filters
% figure
% plot(cnt_f(:,1),'r')     
% title('EEG Signal of a Single Channel after Pre-Processing with Spatial Filters')
       
% % 52=C3  ;  56=C4;
% figure
% subplot(2,1,1)
% plot(cnt(:,52),'r','linewidth',0.5)
% title('EEG signal acquired from C3 channel','fontsize',15)
% subplot(2,1,2)
% % plot(cntf(:,52),'b','linewidth',0.5)
% plot(cnt_f(:,52),'b','linewidth',0.5)
% title('Filtered EEG signal of C3 channel','fontsize',15)
% 
% figure
% subplot(2,1,1)
% plot(cnt(:,56),'r','linewidth',0.5)
% title('EEG signal acquired from C4 channel','fontsize',15)
% subplot(2,1,2)
% % plot(cntf(:,56),'b','linewidth',0.5)
% plot(cnt_f(:,56),'b','linewidth',0.5)
% title('Filtered EEG signal of C4 channel','fontsize',15)


%% Bandpass Filtering to get mu , beta and gamma band information

%%%%%% FBCSP %%%%%%%%%%%%

FrequencyIntervals=questdlg('which frequency band do you want to use?','FrequencyBand',...
    'mu & beta 2OL1','mu & beta 4OL2','mu & beta & gamma','mu & beta & gamma');

% FrequencyBand=questdlg('which frequency band do you want to use?','Frequency Band',...
%     'mu','mu & beta','mu & beta & gamma','mu & beta');

Usegamma=strcmp(FrequencyIntervals,'mu & beta & gamma');
Usetwo=strcmp(FrequencyIntervals,'mu & beta 2OL1');
Usefour=strcmp(FrequencyIntervals,'mu & beta 4OL2');

% Usemu=strcmp(FrequencyBand,'mu');
% Usemubeta=strcmp(FrequencyBand,'mu & beta');
% Usemubetagamma=strcmp(FrequencyBand,'mu & beta & gamma');

if Usegamma
%         bands=[8:1:29;9:1:30];  
%         bands=[8:4:44;12:4:48];
%         bands=[8:2:44;12:2:48];
%         bands=[8:1:47;10:1:49];
        bands=[8:1:38;10:1:40];
end

if Usetwo
%         bands=[8:2:28;10:2:30]; 
        bands=[8:1:28;10:1:30]; 
end

if Usefour
%         bands=[8:4:28;12:4:32]; 
        bands=[8:2:28;12:2:32];
end

% if Usemu
%         fl = 8;     % beginning of mu
%         fh = 13;    % end of mu                  
% end
% 
% if Usemubeta
%         fl = 8;     % beginning of mu
%         fh = 30;    % end of beta                  
% end
% 
% if Usemubetagamma
%         fl = 8;     % beginning of mu
%         fh = 49.9;    % end of gamma     %not 60-70 because fs=100 => fs/2=50                  
% end

% order = 5;
% wn = [fl fh]/(fs/2);
% type = 'bandpass';
% [b,a] = butter(order,wn,type);
% cntf = filtfilt(b,a,cnt_f);

% figure
% subplot(2,1,1)
% plot(cnt(:,1),'r','linewidth',0.5)
% title('EEG signal acquired from the first channel','fontsize',15)
% subplot(2,1,2)
% plot(cntf(:,1),'b','linewidth',0.5)
% title('BandPass Filtered EEG signal of the first channel','fontsize',15)

% % m = length(cnt);
% % r = linspace(0,fs/2,round(m/2));
% % 
% % figure 
% % 
% % f1 = fft(cnt);
% % f1 = abs(f1);
% % f1 = f1(1:round(m/2));
% % subplot(2,1,1)
% % stem(r,f1,'marker','non','linewidth',1)
% % title('Original EEG Coef.','fontsize',15)
% % 
% % f2 = fft(cntf);
% % f2 = abs(f2);
% % f2 = f2(1:round(m/2));
% % subplot(2,1,2)
% % stem(r,f2,'marker','non','linewidth',1)
% % title('Filtered EEG Coef.','fontsize',15)

%%
       featuretestreal=[];
%         datatestreal=[];
%         featuretest1= [];
%         featuretest2 = [];
        featuretrain1= [];
        featuretrain2= [];
        
        prompt='Enter value of mm';
        dlgtitle='Input';
        dims=[1 40];
        answer=inputdlg(prompt,dlgtitle,dims);
        mm=str2num(answer{1});

 for bn= 1:size(bands,2)    
    wn= bands(:,bn);   
      

        %%%% Butterworth
        % order = 3;
        order = 5;
        Params.ButterworthOrder=order;
%         type = 'bandpass';
        [b,a] = butter(order,[wn]/(fs/2),'bandpass');    
        cntf = filtfilt(b,a,cnt_f);            % Filtered EEG signal


%% Extracting Trials from EEG & Separating them by Labels
Params.TrialLength='3.5 sec';
trlen = 3.5*fs;            % Trial Length (No. of samples in a single trial)
trlab = mrk.y;           % Trial Labels
trad = mrk.pos;          % Trial Address (beginning)

c1 = 0;                               % We use a counter instaed of "i" to avoid zero ....
c2 = 0;
for i=1:length(trlab)
    index= trad(i):trad(i)+trlen-1;   % Addresses of samples in a trial (+349 not 350)
    trial = cntf(index,:);             % EEG of a single trial in all channels
    if trlab(i) == 1                   % 1=right
        c1=c1+1;
        data1(:,:,c1) = trial;
    elseif trlab(i) == 2               % 2=foot 
        c2=c2+1;
        data2(:,:,c2) = trial;
    end
end


datatrain1=data1(:,:,:);    % datatrain1=data1
datatrain2=data2(:,:,:);


%% Applying Spatial filter of CSP on each single trial and putting them in a loop
% figure
% prompt = " m first and last columns of  ...?";
% mm = input(prompt)


% mm=1;                               %No. of spatial filters for each class
% mm=2; 
% mm=3; 
Params.NumberOfCSPs1stapproach = 2*mm;

[w] = myCSP(datatrain1,datatrain2,mm);
% ww = reshape(w',[],236);
% Params.NumberOfCSPs2ndapproach = numel(ww);

for i=1:size(datatrain1,3)         %#al    %No. of loops = No. of trials  
    x1=datatrain1(:,:,i)';
    x2=datatrain2(:,:,i)';
    
    
%     subplot(2,2,1)
%     plot(x1(52,:),x1(56,:),'r.');                %53=C1  ;  55=C2;   52=C3  ;  56=C4;
%     hold on
%     plot(x2(52,:),x2(56,:),'b.');
%     title('EEG before CSP');
%     drawnow

%     rez1=ww'*x1(52,:);
%     rez2=ww'*x2(52,:);  

    y1=w'*x1;
    y2=w'*x2;
    
    % First approach to feature extraction for variance
    
    %     temptrain1(:,i)=var(rez1');
    %     temptrain2(:,i)=var(rez2');
    
%         featuretrain1(:,i)=var(rez1');
%         featuretrain2(:,i)=var(rez2');
        
        temptrain1(:,i)=var(y1');
        temptrain2(:,i)=var(y2');
        
%     featuretrain1(:,i)=var(y1');
%     featuretrain2(:,i)=var(y2');
      
%     
%     % Second approach to feature extraction for variance
% %     featuretrain_1(:,i)=log10(var(y1')/sum(var(y1')));
% %     featuretrain_2(:,i)=log10(var(y2')/sum(var(y2')));
%       
%     subplot(2,2,2)
%     plot(y1(1,:),y1(2,:),'r.'); 
%     hold on
%     plot(y2(1,:),y2(2,:),'b.');
%     title('EEG after CSP');
    
%         subplot(2,2,2)
%     plot(rez1(1,:),rez1(2,:),'r.'); 
%     hold on
%     plot(rez2(1,:),rez2(2,:),'b.');
%     title('EEG after CSP');
    
    
%      subplot(2,2,3)
%     plot(featuretrain1(1,:),featuretrain1(2,:),'rs','linewidth',2,'markersize',8);
%     hold on
%     plot(featuretrain2(1,:),featuretrain2(2,:),'bo','linewidth',2,'markersize',8);
%     title('Variance of each EEG Channel');
% 
%      subplot(2,2,4)
% %     plot(featuretrain_1(1,:),featuretrain_1(2,:),'rs','linewidth',2,'markersize',8);
% %     hold on
% %     plot(featuretrain_2(1,:),featuretrain_2(2,:),'bo','linewidth',2,'markersize',8);
% %     title('Logarithm of Variance of each EEG Channel');

end


%% Applying CSP on Unlabeld Test Trials

c3=0;                            % We use a counter instaed of "i" to avoid zero ....
unlbtst=NoLabeldTrials+1;
for i=unlbtst:280
    index= trad(i):trad(i)+trlen-1;   % Addresses of samples in a trial (+349 not 350)
    trial = cntf(index,:);             % EEG of a single trial in all channels
    c3=c3+1;
    datatestreal(:,:,c3) = trial;  
end

for i=1:size(datatestreal,3)          %No. of loops = No. of trials
    xreal=datatestreal(:,:,i)';

%     rezreal=ww'*xreal(52,:);

    yreal=w'*xreal;
    
    %First approach to feature extraction for variance
    
    %     temptestreal(:,i)=var(rezreal');
    
            temptestreal(:,i)=var(yreal');
    
%         featuretestreal(:,i)=var(rezreal');
    
%     featuretestreal(:,i)=var(yreal');
 
end

    featuretrain1= [featuretrain1;temptrain1];
    featuretrain2= [featuretrain2;temptrain2];
%     featuretest1=  [featuretest1;temptest1];
%     featuretest2 = [featuretest2;temptest2];
    featuretestreal = [featuretestreal;temptestreal];
            
    temptrain1= [];
    temptrain2 = [];
% %     temptest1= [];
% %     temptest2= [];
    temptestreal= [];
    
 end


%% feature selection using Fischer Discriminant Ratio (FDR)
%         for fn= 1:size(featuretrain1,1)
%             m1= mean(featuretrain1(fn,:));
%             m2= mean(featuretrain2(fn,:));
%             s1= var(featuretrain1(fn,:));
%             s2= var(featuretrain2(fn,:));
%             fdr(fn) = ((m1-m2)^2) / (s1+s2);
%         end
%         numf= 5;
%         [fdr,ind]= sort(fdr,'descend');
%         sel_ind= ind(1:numf);
%         
%         featuretrain1=featuretrain1(sel_ind,:);
%         featuretrain2=featuretrain2(sel_ind,:);
%         
%         featuretest1=featuretest1(sel_ind,:);
%         featuretest2=featuretest2(sel_ind,:);
         
%         featuretestreal=featuretestreal(sel_ind,:);

%%
global xxx ttt datatrain labeltrain datatestreal labeltestreal NoLabeldTrials trlab Compactmdl1
    
%%
datatrain=[featuretrain1,featuretrain2];
labeltrain=[ones(1,size(featuretrain1,2)),2*ones(1,size(featuretrain2,2))];

%%
xxx=datatrain;
ttt=double([labeltrain==1; labeltrain==2]);
% 
    data.nx=size(xxx,1);
    data.nt=size(ttt,1);
    data.nSample=size(xxx,2);
    
    data.xxx=xxx;
    data.ttt=ttt;
    
    %%
% datatest=[featuretest1,featuretest2];
% labeltest=[ones(1,size(featuretest1,2)),2*ones(1,size(featuretest2,2))];

%%
truelabels=true_y;
datatestreal=[featuretestreal];
labeltestreal=[truelabels(unlbtst:280)];

%%%

CostFunction=@(s) FeatureSelectionCost(s,data);     % Cost Function

nVar=data.nx;            % Number of Decision Variables (Number of bits in a chromosome)

VarSize=[1 nVar];   % Decision Variables Matrix Size

% Number of Objective Functions
nObj=numel(CostFunction(randi([0 1],VarSize)));

% x0=randi([0 1],VarSize);
% nObj=numel(CostFunction(x0));


% NSGA-II Parameters

% MaxIt=100;      % Maximum Number of Iterations
% MaxIt=20;
% MaxIt=3;
MaxIt=1;

% nPop=50;        % Population Size
% nPop=30;        % Population Size
nPop=10;        % Population Size

pc=0.7;                 % Crossover Percentage
nc=2*round(pc*nPop/2);  % Number of Offsprings (Parnets)

pm=0.4;                 % Mutation Percentage
nm=round(pm*nPop);      % Number of Mutants

mu=0.1;         % Mutation Rate ('0' -> no bits, '1' -> all bits), lower mu means more exploration
% mu=0.01;

% Initialization

disp('Initialization ...');

empty_individual.Position=[];
empty_individual.Cost=[];
empty_individual.out=[];
empty_individual.Rank=[];
empty_individual.DominationSet=[];       % Sp
empty_individual.DominatedCount=[];      % np
empty_individual.CrowdingDistance=[];


pop=repmat(empty_individual,nPop,1);

for i=1:nPop
    
    % Initialize Position
    if i~=1
        pop(i).Position=randi([0 1],VarSize);
    else
        pop(i).Position=ones(VarSize);   % The solution that containes all the features
    end

    % Evaluation
    [pop(i).Cost ,pop(i).out]=CostFunction(pop(i).Position);
%     [pop(i).Cost]=CostFunction(pop(i).Position);
    
end

% Non-Dominated Sorting
[pop ,F]=NonDominatedSorting(pop);

% Calculate Crowding Distance
pop=CalcCrowdingDistance(pop,F);

% Sort Population
[pop ,F]=SortPopulation(pop);


% NSGA-II Main Loop

for it=1:MaxIt
    
    disp(['Starting Iteration ' num2str(it) '...']);
    
    
    % Crossover
    popc=repmat(empty_individual,nc/2,2);
    for k=1:nc/2
        
        i1=randi([1 nPop]);
        p1=pop(i1);
        
        i2=randi([1 nPop]);
        p2=pop(i2);

        
        % Apply Crossover
        [popc(k,1).Position ,popc(k,2).Position]=Crossover(p1.Position,p2.Position);
        
        % Evaluate Offsprings
%         [popc(k,1).Cost, popc(k,1).out]=CostFunction(popc(k,1).Position);
%         [popc(k,2).Cost, popc(k,2).out]=CostFunction(popc(k,2).Position);

        while any(popc(k,1).Position)==0 | any(popc(k,2).Position)==0   
            disp(['kc ' num2str(k)])
            [popc(k,1).Position ,popc(k,2).Position]=Crossover(p1.Position,p2.Position);
        end
        
        [popc(k,1).Cost, popc(k,1).out]=CostFunction(popc(k,1).Position);
        [popc(k,2).Cost, popc(k,2).out]=CostFunction(popc(k,2).Position);
        
    end
    popc=popc(:);
    
    
    % Mutation
    popm=repmat(empty_individual,nm,1);
    for k=1:nm
        
        % Select Parent
        i=randi([1 nPop]);
        p=pop(i);
        
        % Apply Mutation
        popm(k).Position=Mutate(p.Position,mu);
        
        while any(popm(k).Position)==0         
            disp(['km ' num2str(k)])
            popm(k).Position=Mutate(p.Position,mu);
        end
        
        % Evaluate Mutant
        [popm(k).Cost, popm(k).out]=CostFunction(popm(k).Position);
        
    end
    
    % Create Merged Population
    pop=[pop
         popc
         popm];   
     
       % Non-Dominated Sorting
    [pop ,F]=NonDominatedSorting(pop);

    % Calculate Crowding Distance
    pop=CalcCrowdingDistance(pop,F);

    % Sort Population
    [pop ,F]=SortPopulation(pop); %#ok
    
    % Truncate
    pop=pop(1:nPop);
    
    % Non-Dominated Sorting
    [pop ,F]=NonDominatedSorting(pop);

    % Calculate Crowding Distance
    pop=CalcCrowdingDistance(pop,F);

    % Sort Population
    [pop ,F]=SortPopulation(pop);
    
    % Store F1
    F1=pop(F{1});
    F1=GetUniqueMembers(F1);

    
    F1san{it,:}={F1.Cost};

    
%     Z1=[F1.Cost];
%     NF1=Z1(1,:);          % First row of Z1
%     F1=F1(NF1>0);         % for omitting the conditions in which no feature is selected

    
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': Number of F1 Members = ' num2str(numel(F1))]);
    
    % Plot F1 Costs
    figure(1);
    PlotCosts(F1);
    pause(0.1);
    
end

global mdl1


%% Train classifier using train data and label
%%%%%%   SVM   %%%%%
% mdl1=fitcsvm(datatrain',labeltrain,'KFold',10);
% % CVmdl1=crossval(mdl1);
% % output1=predict(mdl1,datatest');
% % output1real=kfoldPredict(mdl1);
% % output1real=predict(mdl1,datatestreal');
% % output1real=predict(CVmdl1,datatestreal');
% % accuracy1= sum(labeltest==output1')/numel(output1)*100;
% % accuracy1real= sum(labeltestreal==output1real')/numel(output1real)*100;
% % Acck(iter,1)=accuracy1;
% % Acckreal(iter,1)= accuracy1real;
% % Acc(bn,1)=accuracy1;
% % Accreal(bn,1)=accuracy1;

% classLoss = kfoldLoss(CVmdl1); 


        % %%%%%%% Classification SVM  %%%%%        
%         mdl1=fitcsvm(datatrain',labeltrain,'Crossval','on');    %, 'Crossval','on','ClassNames',{'1','2'},'Standardize',true
%         classLoss = kfoldLoss(mdl1);    %By default, kfoldLoss returns the classification error
%                                         % out-of-sample misclassification rate or generalization rate
%         acctrain = (1- classLoss)*100
% %         output1real=predict(mdl1.Trained{1},datatestreal');
% %         accuracy1real= sum(labeltestreal==output1real')/numel(output1real)*100;
%          EE=classLoss*100;


%% Final Evaluation of the Classifier on the Train Data

          X = datatrain';

        cc = cvpartition(NoLabeldTrials,'KFold',10)
%         fold1 = test(cc,1);
        
        Compactmdl1 = mdl1.Trained{6}; % Extract trained, compact classifier
        testInds=test(cc,6);   % Extract the test indices
        XTest = X(testInds,:);

        output1 = predict(Compactmdl1,XTest);
        
        labeltest=[trlab(testInds)];
        
        accuracy1= sum(labeltest==output1')/numel(output1)*100


%% Final Evaluation of the Classifier on the Test Data

%         output1real=predict(mdl1.Trained{6},datatestreal')    %.Trained{10}
          output1real=predict(Compactmdl1,datatestreal');
        
%         output1real=predict(Compactmdl1,datatestreal')   
%         output1real=predict(mdl1,datatestreal')  
        accuracy1real= sum(labeltestreal==output1real')/numel(output1real)*100
        
        C = confusionmat(output1real,labeltestreal)
        confusionchart(C)
     

%         output1=predict(mdl1,XTest)  
%             output1real=predict(mdl1.Trained{10},datatestreal')    %.Trained{10}
% %         output1real=predict(Compactmdl1,datatestreal')   
% %         output1real=predict(mdl1,datatestreal')  
%         accuracy1real= sum(labeltestreal==output1real')/numel(output1real)*100
%         sanaz1 = confusionmat(labeltestreal,output1real)
%         sanaz2 = confusionchart(labeltestreal,output1real)
        
% %         output3real=predict(mdl3.Trained{1},datatestreal');
% %         accuracy3real= sum(labeltestreal==output3real')/numel(output3real)*100;        
% %         sanaz1 = confusionmat(labeltestreal,output3real)
% %         sanaz2 = confusionchart(labeltestreal,output3real)



%% Train classifier using train data and label
% %%%%%%   SVM   %%%%%
% mdl1=fitcsvm(datatrain',labeltrain);
% output1=predict(mdl1,datatest');
% output1real=predict(mdl1,datatestreal');
% accuracy1= sum(labeltest==output1')/numel(output1)*100;
% accuracy1real= sum(labeltestreal==output1real')/numel(output1real)*100;
% Acck(iter,1)=accuracy1;
% Acckreal(iter,1)= accuracy1real;
% % Acc(bn,1)=accuracy1;
% % Accreal(bn,1)=accuracy1;
% 

% %%%%%%   Nonlinear SVM   %%%%%%%
% mdl2=fitcsvm(datatrain',labeltrain,'Standardize',true,'KernelFunction','rbf');
% % ,...
% %      'KernelScale','auto');
% % Train an SVM classifier using the radial basis kernel. Let the software find a scale 
% % value for the kernel function. Standardize the predictors.
% output2=predict(mdl2,datatest');
% output2real=predict(mdl2,datatestreal');
% accuracy2= sum(labeltest==output2')/numel(output2)*100;
% accuracy2real= sum(labeltestreal==output1real')/numel(output1real)*100;
% Acck(iter,2)=accuracy2;
% Acckreal(iter,2)= accuracy2real;
% 

% %%%%%%   KNN   %%%%%%
% mdl3=fitcknn(datatrain',labeltrain,'NumNeighbors',5);
% output3=predict(mdl3,datatest');
% output3real=predict(mdl3,datatestreal');
% accuracy3= sum(labeltest==output3')/numel(output3)*100;
% accuracy3real= sum(labeltestreal==output3real')/numel(output3real)*100;
% Acck(iter,3)=accuracy3;
% Acckreal(iter,3)= accuracy3real;
% 
% 
% %%%%%%   LDA   %%%%%%%
% mdl4=fitcdiscr(datatrain',labeltrain);
% output4=predict(mdl4,datatest');
% output4real=predict(mdl4,datatestreal');
% accuracy4= sum(labeltest==output4')/numel(output4)*100;
% accuracy4real= sum(labeltestreal==output4real')/numel(output4real)*100;
% Acck(iter,4)=accuracy4;
% Acckreal(iter,4)= accuracy4real;

    
%     end
    
%%
% Acc=mean(Acck);
% Accreal=mean(Acckreal);
% 
% disp(['total Accuracy (SVM) : ',num2str(Acc(1)),' %'])
% disp(['total Accuracy (SVM-2) : ',num2str(Acc(2)),' %'])
% disp(['total Accuracy (KNN) : ',num2str(Acc(3)),' %'])
% disp(['total Accuracy (LDA) : ',num2str(Acc(4)),' %'])
% 
% disp(['total Accuracyreal (SVM) : ',num2str(Accreal(1)),' %'])
% disp(['total Accuracyreal (SVM-2) : ',num2str(Accreal(2)),' %'])
% disp(['total Accuracyreal (KNN) : ',num2str(Accreal(3)),' %'])
% disp(['total Accuracyreal (LDA) : ',num2str(Accreal(4)),' %'])
% 
% figure
% boxplot(Acck,'Labels',{'SVM','SVM-2','KNN','LDA'})
% title('Box plot of average classification accuracy by different classifiers')
% xlabel('Classification Algorithm')
% ylabel('Classification Accuracy % ')
% % ylim([80 105])
% 
% figure
% boxplot(Acckreal,'Labels',{'SVM','SVM-2','KNN','LDA'})
% title('Box plot of average classification accuracy by different classifiers on unseen test data')
% xlabel('Classification Algorithm')
% ylabel('Classification Accuracy % ')
% % ylim([90 102])

% Results.Acc=Acc;
% Results.Accreal=Accreal;

toc

