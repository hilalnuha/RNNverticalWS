clc
clear all
close all
%M=csvread("WS_10m.csv",1,3);
%M=csvread("WS_hr.csv",2,3,[2 3 3000 3]);
M=csvread("WS_KFUPM_10m_2015.csv",1,2);
% [10 20 30 40 60 80 100 120 140 160 180]
% [10 20 30 40 x50 60 x70 80 x90 100 x110 120 x130 140 x150 160 x170 180]
%M=M(:,[2 4 5 6 7 8 10 

days=360;
%numdat=6*24*days;
inputsize=4;
%M=M(1:(numdat),:);
N=length(M);

for k=1:11
for i=1:N  
    if M(i,k)> 20 %CLEAN THE DATA if 9999, then replace with previous value
        M(i,k)=M(i-1,k);
    elseif M(i,k)<= 0
    M(i,k)=M(i-1,k);
    end
end
end

M=fliplr(M);

ii=1;
for i=1:N
    diff0=M(i,2:11)-M(i,1:10);
    lt0=sum(find(diff0<0.1));
    if lt0==0 
        MN(ii,:)=M(i,:);
        ii=ii+1;
    end
end

mt15=find(MN(:,6)<=15);   
M=MN(mt15,:);
mt10=find(M(:,3)<=10);   
M=M(mt10,:);

M=[M(:,[1 2 3 4]) (M(:,4)+M(:,5))/2 M(:,5) (M(:,5)+M(:,6))/2 M(:,6) (M(:,6)+M(:,7))/2 M(:,7) (M(:,7)+M(:,8))/2 M(:,8) (M(:,8)+M(:,9))/2 M(:,9) (M(:,9)+M(:,10))/2 M(:,10) (M(:,10)+M(:,11))/2 M(:,11)];
% [10 20 30 40 60 80 100 120 140 160 180]
% [10 20 30 40 x50 60 x70 80 x90 100 x110 120 x130 140 x150 160 x170 180]
%

perc=100;
numdat=length(M);

%R=6; % Every 6 makes an hour
%mm=floor(N/R);
%for i=1:mm
%    j=(i-1)*R+1;
%    MD(i,1)=mean(M(j:j+R-1));
%end

trainingnum=floor(0.8*numdat); % Num of training samples
maxx=max(max(M(1:trainingnum,1:inputsize)));
training=M(1:trainingnum,:);

series=training/maxx;
datasize=size(series);
nex=1;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testing=M((trainingnum+1):end,:);

seriesT=testing/maxx;
%numdata=max(datasize)-(inputsize+ahead-1);
testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

P50 = traininginput';
Y50 = trainingtarget';
Ptest50 = testinginput';
Ytest50 = testingtarget';
testingtarget50=Ytest50'*maxx;
%
%Create NN
numhid=20;
pastval=[0 1];
nn = [inputsize numhid 1];
dIn = [pastval];
dIntern=[];
dOut=[];
miter=20;
net = CreateNN(nn,dIn,dIntern,dOut);
netLM = train_LM(P50,Y50,net,miter,1e-5);
%Calculate Output of trained net (LM) for training and Test Data
outval = NNOut(P50,netLM);

%outval = netMLP(P);
outvalmax=outval'*maxx;
RNNOutf50train=outvalmax;

trainingtargetmax=trainingtarget*maxx;

msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = NNOut(Ptest50,netLM);

outvaltestmax=outvaltest'*maxx;
RNNOutf50=outvaltestmax;
testingtargetmax=testingtarget*maxx;
msetest50=mse(RNNOutf50,testingtarget50);
mapetest50=mape(RNNOutf50,testingtarget50);
mbetest50=mbe(RNNOutf50,testingtarget50);
r2test50=rsquare(RNNOutf50,testingtarget50);
RNNperf50=[msetest50 mapetest50 mbetest50 r2test50];
target50=testingtarget50;

%
figure
PtestMax50=Ptest50'*maxx;
height50=[10 20 30 40 50];
plot([PtestMax50'; RNNOutf50' ],height50);
rang50=[0 13];
xlim(rang50)
%ylim([-0.4 0.8])
title(['RNN 50 m Testing MSE=' num2str(msetest50) ',MAPE=' num2str(mapetest50) ',MBE=' num2str(mbetest50) ',R^2=' num2str(r2test50)]);
%
figure
rl50=[1:13];
plot( RNNOutf50, target50,'ob',rl50,rl50,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl50)+0.5]);
ylim([0 max(rl50)+0.5]);
title(['RNN 50 m Testing ']);
text(2,12,['R^2=' num2str(round(r2test50*perc,2)) ' %'])

%xlim([0 13])
%ylim([-0.4 0.8])

%camroll(90)

%
MLPP50 = traininginput';
MLPY50 = trainingtarget';
MLPPtest50 = testinginput';
MLPYtest50 = testingtarget';
MLPtestingtarget50=MLPYtest50'*maxx;

%Create NN

numiter=20;
numhid=20;
netMLP=feedforwardnet(numhid);
netMLP.divideFcn='divideind';
%netMLP.trainParam.showWindow = 0;
netMLP.divideParam.trainInd=1:max(size(MLPP50)); %training_index
netMLP.trainParam.epochs=numiter;
netMLP.trainParam.mu=3;
netMLP.trainParam.mu_dec=0.1;
netMLP.trainParam.mu_inc =10;
netMLP.trainParam.showWindow = 0; 
[netMLP,tr,Y,E] = train(netMLP,MLPP50,MLPY50);

outval = netMLP(MLPP50);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netMLP(MLPPtest50);
outvaltestmax=outvaltest'*maxx;
MLPOutf50=outvaltestmax;
MLPtestingtargetmax50=testingtarget50;
MLPmsetest50=mse(MLPOutf50,MLPtestingtargetmax50);
MLPmapetest50=mape(MLPOutf50,MLPtestingtargetmax50);
MLPmbetest50=mbe(MLPOutf50,MLPtestingtargetmax50);
MLPr2test50=rsquare(MLPOutf50,MLPtestingtargetmax50);
MLPperf50=[MLPmsetest50 MLPmapetest50 MLPmbetest50 MLPr2test50];
MLPOutf50train=outvalmax;
MLPPtestMax50=MLPPtest50'*maxx;
%
figure

plot([MLPPtestMax50'; MLPOutf50' ],height50);
xlim(rang50)
%ylim([-0.4 0.8])
title(['MLP 50m Testing MSE=' num2str(MLPmsetest50) ',MAPE=' num2str(MLPmapetest50) ',MBE=' num2str(MLPmbetest50)  ',R^2=' num2str(MLPr2test50)]);
%camroll(90)
% [10 20 30 40 50 60 70 80 90 100 110 120 130 140 150 160 170 180]
%
figure
%rl=[1:13];
plot( MLPOutf50, testingtargetmax,'ob',rl50,rl50,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl50)+0.5]);
ylim([0 max(rl50)+0.5]);
title(['MLP 50 m Testing ']);
text(2,12,['R^2=' num2str(round(MLPr2test50*perc,2)) ' %'])
% 1/7 WSE

LWSEP50 = traininginput';
LWSEY50 = trainingtarget';
LWSEPtest50 = testinginput';
LWSEYtest50 = testingtarget';
LWSEtestingtarget50=LWSEYtest50'*maxx;


alpha=1/3;
outval=LWSEP50(4,:)*((50/40)^(alpha));
outvalmax=outval*maxx;
LWSEOutf50train=outvalmax;


outvaltest=LWSEPtest50(4,:)*((50/40)^(alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget50;
LWSEOutf50=outvaltestmax;
LWSEmsetest50=mse(LWSEOutf50,testingtarget50);
LWSEmapetest50=mape(LWSEOutf50,testingtarget50);
LWSEmbetest50=mbe(LWSEOutf50,testingtarget50);
LWSEr2test50=rsquare(LWSEOutf50,testingtarget50);
LWSEperf50=[LWSEmsetest50 LWSEmapetest50 LWSEmbetest50 LWSEr2test50];
LWSEPtestMax50=LWSEPtest50'*maxx;

figure
interv=1:200;
plot( [testingtarget50(interv) ],'b');
hold on
plot( [RNNOutf50(interv)],'r');
plot( [MLPOutf50(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','MLP est','Location','northwest')
meantarget50=mean([seriesT(:,1:inputsize)'*maxx; testingtarget50' ]');
meanRNN50=mean([PtestMax50'; RNNOutf50' ]');
meanMLP50=mean([MLPPtestMax50'; MLPOutf50' ]');
meanLWSE50=mean([LWSEPtestMax50'; LWSEOutf50' ]');
figure
plot(meantarget50,height50,'k');
hold on
plot(meanRNN50,height50,'b-s');
plot(meanMLP50,height50,'--r');
plot(meanLWSE50,height50,'-.g');

hold off
title('average')
legend('measured','RNN est','MLP est','1/7 est','Location','northwest')

display ('50 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf50]
[MLPperf50]
[LWSEperf50]

%% 60

nex=2;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

P60 = [P50; RNNOutf50train'/maxx];
Y60 = trainingtarget';
Ptest60 = [Ptest50; RNNOutf50'/maxx];
Ytest60 = testingtarget';
testingtarget60=Ytest60'*maxx;


%
%Create NN
numhid=20;
pastval=[0 1];
nn = [inputsize+(nex-1) numhid 1];
dIn = [pastval];
dIntern=[];
dOut=[];
miter=20;
net = CreateNN(nn,dIn,dIntern,dOut);
netLM = train_LM(P60,Y60,net,miter,1e-5);
%Calculate Output of trained net (LM) for training and Test Data
outval = NNOut(P60,netLM);

%outval = netMLP(P);
outvalmax=outval'*maxx;
RNNOutf60train=outvalmax;

trainingtargetmax=trainingtarget*maxx;

msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = NNOut(Ptest60,netLM);

outvaltestmax=outvaltest'*maxx;
RNNOutf60=outvaltestmax;
testingtargetmax=testingtarget*maxx;
msetest60=mse(RNNOutf60,testingtarget60);
mapetest60=mape(RNNOutf60,testingtarget60);
mbetest60=mbe(RNNOutf60,testingtarget60);
r2test60=rsquare(RNNOutf60,testingtarget60);
RNNperf60=[msetest60 mapetest60 mbetest60 r2test60];
target60=testingtarget60;

%
figure
PtestMax60=Ptest60'*maxx;
height60=[height50 60];
plot([PtestMax60'; RNNOutf60' ],height60);
rang60=[0 13.5];
xlim(rang60)
%ylim([-0.4 0.8])
title(['RNN 60 m Testing MSE=' num2str(msetest60) ',MAPE=' num2str(mapetest60) ',MBE=' num2str(mbetest60) ',R^2=' num2str(r2test60)]);
%
figure
rl60=[1:13.5];
plot( RNNOutf60, target60,'ob',rl60,rl60,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl60)+0.5]);
ylim([0 max(rl60)+0.5]);
title(['RNN 60 m Testing ']);
text(2,12,['R^2=' num2str(round(r2test60*perc,2)) ' %'])

%xlim([0 13])
%ylim([-0.4 0.8])

%camroll(90)
%
MLPP60= [MLPP50; MLPOutf50train'/maxx];
MLPY60 = trainingtarget';
MLPPtest60 = [MLPPtest50; MLPOutf50'/maxx];
MLPYtest60 = testingtarget';
MLPtestingtarget60=MLPYtest60'*maxx;

%Create NN

numiter=20;
numhid=20;
netMLP=feedforwardnet(numhid);
netMLP.divideFcn='divideind';
%netMLP.trainParam.showWindow = 0;
netMLP.divideParam.trainInd=1:max(size(MLPP60)); %training_index
netMLP.trainParam.epochs=numiter;
netMLP.trainParam.mu=3;
netMLP.trainParam.mu_dec=0.1;
netMLP.trainParam.mu_inc =10;
[netMLP,tr,Y,E] = train(netMLP,MLPP60,MLPY60);

outval = netMLP(MLPP60);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netMLP(MLPPtest60);
outvaltestmax=outvaltest'*maxx;
MLPOutf60=outvaltestmax;
MLPtestingtargetmax60=testingtarget60;
MLPmsetest60=mse(MLPOutf60,MLPtestingtargetmax60);
MLPmapetest60=mape(MLPOutf60,MLPtestingtargetmax60);
MLPmbetest60=mbe(MLPOutf60,MLPtestingtargetmax60);
MLPr2test60=rsquare(MLPOutf60,MLPtestingtargetmax60);
MLPperf60=[MLPmsetest60 MLPmapetest60 MLPmbetest60 MLPr2test60];
MLPOutf60train=outvalmax;
%
figure
MLPPtestMax60=MLPPtest60'*maxx;

plot([MLPPtestMax60'; MLPOutf60' ],height60);
xlim(rang60)
%ylim([-0.4 0.8])
title(['MLP 60m Testing MSE=' num2str(MLPmsetest60) ',MAPE=' num2str(MLPmapetest60) ',MBE=' num2str(MLPmbetest60)  ',R^2=' num2str(MLPr2test60)]);
%camroll(90)
% [10 20 30 40 60 60 70 80 90 100 110 120 130 140 160 160 170 180]
%
figure
%rl=[1:13];
plot( MLPOutf60, testingtargetmax,'ob',rl60,rl60,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl60)+0.5]);
ylim([0 max(rl60)+0.5]);
title(['MLP 60 m Testing ']);
text(2,12,['R^2=' num2str(round(MLPr2test60*perc,2)) ' %'])


% 1/7 WSE
%
LWSEP60 = [LWSEP50; LWSEOutf50train/maxx];
LWSEY60 = trainingtarget';
LWSEPtest60 = [LWSEPtest50; LWSEOutf50'/maxx];
LWSEYtest60 = testingtarget';
LWSEtestingtarget60=LWSEYtest60'*maxx;


alpha=1/3;
outval=LWSEP60(5,:)*((60/50)^(alpha));
outvalmax=outval*maxx;
LWSEOutf60train=outvalmax;


outvaltest=LWSEPtest60(5,:)*((60/50)^(alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget60;
LWSEOutf60=outvaltestmax;
LWSEmsetest60=mse(LWSEOutf60,testingtarget60);
LWSEmapetest60=mape(LWSEOutf60,testingtarget60);
LWSEmbetest60=mbe(LWSEOutf60,testingtarget60);
LWSEr2test60=rsquare(LWSEOutf60,testingtarget60);
LWSEperf60=[LWSEmsetest60 LWSEmapetest60 LWSEmbetest60 LWSEr2test60];
LWSEPtestMax60=LWSEPtest60'*maxx;

figure
interv=1:200;
plot( [testingtarget60(interv) ],'b');
hold on
plot( [RNNOutf60(interv)],'r');
plot( [MLPOutf60(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','MLP est','Location','northwest')

meantarget60=[meantarget50 mean(testingtarget60)];
meanRNN60=[meanRNN50 mean(RNNOutf60)];
meanMLP60=[meanMLP50 mean(MLPOutf60)];
meanLWSE60=[meanLWSE50 mean(LWSEOutf60)];

figure
plot(meantarget60,height60,'k');
hold on
plot(meanRNN60,height60,'b-s');
plot(meanMLP60,height60,'--r');
plot(meanLWSE60,height60,'-.g');

hold off
title('average')
legend('measured','RNN est','MLP est','1/7 est','Location','northwest')

display ('60 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf60]
[MLPperf60]
[LWSEperf60]

%% 70

nex=3;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

P70 = [P60; RNNOutf60train'/maxx];
Y70 = trainingtarget';
Ptest70 = [Ptest60; RNNOutf60'/maxx];
Ytest70 = testingtarget';
testingtarget70=Ytest70'*maxx;


%
%Create NN
numhid=20;
pastval=[0 1];
nn = [inputsize+(nex-1) numhid 1];
dIn = [pastval];
dIntern=[];
dOut=[];
miter=20;
net = CreateNN(nn,dIn,dIntern,dOut);
netLM = train_LM(P70,Y70,net,miter,1e-5);
%Calculate Output of trained net (LM) for training and Test Data
outval = NNOut(P70,netLM);

%outval = netMLP(P);
outvalmax=outval'*maxx;
RNNOutf70train=outvalmax;

trainingtargetmax=trainingtarget*maxx;

msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = NNOut(Ptest70,netLM);

outvaltestmax=outvaltest'*maxx;
RNNOutf70=outvaltestmax;
testingtargetmax=testingtarget*maxx;
msetest70=mse(RNNOutf70,testingtarget70);
mapetest70=mape(RNNOutf70,testingtarget70);
mbetest70=mbe(RNNOutf70,testingtarget70);
r2test70=rsquare(RNNOutf70,testingtarget70);
RNNperf70=[msetest70 mapetest70 mbetest70 r2test70];
target70=testingtarget70;

%
figure
PtestMax70=Ptest70'*maxx;
height70=[height60 70];
plot([PtestMax70'; RNNOutf70' ],height70);
rang70=[0 14];
xlim(rang70)
%ylim([-0.4 0.8])
title(['RNN 70 m Testing MSE=' num2str(msetest70) ',MAPE=' num2str(mapetest70) ',MBE=' num2str(mbetest70) ',R^2=' num2str(r2test70)]);
%
figure
rl70=[1:14];
plot( RNNOutf70, target70,'ob',rl70,rl70,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl70)+0.5]);
ylim([0 max(rl70)+0.5]);
title(['RNN 70 m Testing ']);
text(2,12,['R^2=' num2str(round(r2test70*perc,2)) ' %'])

%xlim([0 13])
%ylim([-0.4 0.8])

%camroll(90)
%
MLPP70= [MLPP60; MLPOutf60train'/maxx];
MLPY70 = trainingtarget';
MLPPtest70 = [MLPPtest60; MLPOutf60'/maxx];
MLPYtest70 = testingtarget';
MLPtestingtarget70=MLPYtest70'*maxx;

%Create NN

numiter=20;
numhid=20;
netMLP=feedforwardnet(numhid);
netMLP.divideFcn='divideind';
%netMLP.trainParam.showWindow = 0;
netMLP.divideParam.trainInd=1:max(size(MLPP70)); %training_index
netMLP.trainParam.epochs=numiter;
netMLP.trainParam.mu=3;
netMLP.trainParam.mu_dec=0.1;
netMLP.trainParam.mu_inc =10;
[netMLP,tr,Y,E] = train(netMLP,MLPP70,MLPY70);

outval = netMLP(MLPP70);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netMLP(MLPPtest70);
outvaltestmax=outvaltest'*maxx;
MLPOutf70=outvaltestmax;
MLPtestingtargetmax70=testingtarget70;
MLPmsetest70=mse(MLPOutf70,MLPtestingtargetmax70);
MLPmapetest70=mape(MLPOutf70,MLPtestingtargetmax70);
MLPmbetest70=mbe(MLPOutf70,MLPtestingtargetmax70);
MLPr2test70=rsquare(MLPOutf70,MLPtestingtargetmax70);
MLPperf70=[MLPmsetest70 MLPmapetest70 MLPmbetest70 MLPr2test70];
MLPOutf70train=outvalmax;
%
figure
MLPPtestMax70=MLPPtest70'*maxx;

plot([MLPPtestMax70'; MLPOutf70' ],height70);
xlim(rang70)
%ylim([-0.4 0.8])
title(['MLP 70m Testing MSE=' num2str(MLPmsetest70) ',MAPE=' num2str(MLPmapetest70) ',MBE=' num2str(MLPmbetest70)  ',R^2=' num2str(MLPr2test70)]);
%camroll(90)
% [10 20 30 40 70 70 70 80 90 100 110 120 130 140 170 170 170 180]
%
figure
%rl=[1:13];
plot( MLPOutf70, testingtargetmax,'ob',rl70,rl70,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl70)+0.5]);
ylim([0 max(rl70)+0.5]);
title(['MLP 70 m Testing ']);
text(2,12,['R^2=' num2str(round(MLPr2test70*perc,2)) ' %'])


% 1/7 WSE
%
LWSEP70 = [LWSEP60; LWSEOutf60train/maxx];
LWSEY70 = trainingtarget';
LWSEPtest70 = [LWSEPtest60; LWSEOutf60'/maxx];
LWSEYtest70 = testingtarget';
LWSEtestingtarget70=LWSEYtest70'*maxx;


alpha=1/3;
outval=LWSEP70(6,:)*((70/60)^(alpha));
outvalmax=outval*maxx;
LWSEOutf70train=outvalmax;


outvaltest=LWSEPtest70(6,:)*((70/60)^(alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget70;
LWSEOutf70=outvaltestmax;
LWSEmsetest70=mse(LWSEOutf70,testingtarget70);
LWSEmapetest70=mape(LWSEOutf70,testingtarget70);
LWSEmbetest70=mbe(LWSEOutf70,testingtarget70);
LWSEr2test70=rsquare(LWSEOutf70,testingtarget70);
LWSEperf70=[LWSEmsetest70 LWSEmapetest70 LWSEmbetest70 LWSEr2test70];
LWSEPtestMax70=LWSEPtest70'*maxx;

figure
interv=1:200;
plot( [testingtarget70(interv) ],'b');
hold on
plot( [RNNOutf70(interv)],'r');
plot( [MLPOutf70(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','MLP est','Location','northwest')

meantarget70=[meantarget60 mean(testingtarget70)];
meanRNN70=[meanRNN60 mean(RNNOutf70)];
meanMLP70=[meanMLP60 mean(MLPOutf70)];
meanLWSE70=[meanLWSE60 mean(LWSEOutf70)];

figure
plot(meantarget70,height70,'k');
hold on
plot(meanRNN70,height70,'b-s');
plot(meanMLP70,height70,'--r');
plot(meanLWSE70,height70,'-.g');

hold off
title('average')
legend('measured','RNN est','MLP est','1/7 est','Location','northwest')

display ('70 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf70]
[MLPperf70]
[LWSEperf70]

%% 80

nex=4;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

P80 = [P70; RNNOutf70train'/maxx];
Y80 = trainingtarget';
Ptest80 = [Ptest70; RNNOutf70'/maxx];
Ytest80 = testingtarget';
testingtarget80=Ytest80'*maxx;


%
%Create NN
numhid=20;
pastval=[0 1];
nn = [inputsize+(nex-1) numhid 1];
dIn = [pastval];
dIntern=[];
dOut=[];
miter=20;
net = CreateNN(nn,dIn,dIntern,dOut);
netLM = train_LM(P80,Y80,net,miter,1e-5);
%Calculate Output of trained net (LM) for training and Test Data
outval = NNOut(P80,netLM);

%outval = netMLP(P);
outvalmax=outval'*maxx;
RNNOutf80train=outvalmax;

trainingtargetmax=trainingtarget*maxx;

msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = NNOut(Ptest80,netLM);

outvaltestmax=outvaltest'*maxx;
RNNOutf80=outvaltestmax;
testingtargetmax=testingtarget*maxx;
msetest80=mse(RNNOutf80,testingtarget80);
mapetest80=mape(RNNOutf80,testingtarget80);
mbetest80=mbe(RNNOutf80,testingtarget80);
r2test80=rsquare(RNNOutf80,testingtarget80);
RNNperf80=[msetest80 mapetest80 mbetest80 r2test80];
target80=testingtarget80;

%
figure
PtestMax80=Ptest80'*maxx;
height80=[height70 80];
plot([PtestMax80'; RNNOutf80' ],height80);
rang80=[0 14.5];
xlim(rang80)
%ylim([-0.4 0.8])
title(['RNN 80 m Testing MSE=' num2str(msetest80) ',MAPE=' num2str(mapetest80) ',MBE=' num2str(mbetest80) ',R^2=' num2str(r2test80)]);
%
figure
rl80=[1:14.5];
plot( RNNOutf80, target80,'ob',rl80,rl80,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl80)+0.5]);
ylim([0 max(rl80)+0.5]);
title(['RNN 80 m Testing ']);
text(2,12,['R^2=' num2str(round(r2test80*perc,2)) ' %'])

%
MLPP80= [MLPP70; MLPOutf70train'/maxx];
MLPY80 = trainingtarget';
MLPPtest80 = [MLPPtest70; MLPOutf70'/maxx];
MLPYtest80 = testingtarget';
MLPtestingtarget80=MLPYtest80'*maxx;

%Create NN

numiter=20;
numhid=20;
netMLP=feedforwardnet(numhid);
netMLP.divideFcn='divideind';
%netMLP.trainParam.showWindow = 0;
netMLP.divideParam.trainInd=1:max(size(MLPP80)); %training_index
netMLP.trainParam.epochs=numiter;
netMLP.trainParam.mu=3;
netMLP.trainParam.mu_dec=0.1;
netMLP.trainParam.mu_inc =10;
[netMLP,tr,Y,E] = train(netMLP,MLPP80,MLPY80);

outval = netMLP(MLPP80);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netMLP(MLPPtest80);
outvaltestmax=outvaltest'*maxx;
MLPOutf80=outvaltestmax;
MLPtestingtargetmax80=testingtarget80;
MLPmsetest80=mse(MLPOutf80,MLPtestingtargetmax80);
MLPmapetest80=mape(MLPOutf80,MLPtestingtargetmax80);
MLPmbetest80=mbe(MLPOutf80,MLPtestingtargetmax80);
MLPr2test80=rsquare(MLPOutf80,MLPtestingtargetmax80);
MLPperf80=[MLPmsetest80 MLPmapetest80 MLPmbetest80 MLPr2test80];
MLPOutf80train=outvalmax;
%
figure
MLPPtestMax80=MLPPtest80'*maxx;

plot([MLPPtestMax80'; MLPOutf80' ],height80);
xlim(rang80)
%ylim([-0.4 0.8])
title(['MLP 80m Testing MSE=' num2str(MLPmsetest80) ',MAPE=' num2str(MLPmapetest80) ',MBE=' num2str(MLPmbetest80)  ',R^2=' num2str(MLPr2test80)]);
%camroll(90)
% [10 20 30 40 80 80 80 80 90 100 110 120 130 140 180 180 180 180]
%
figure
%rl=[1:13];
plot( MLPOutf80, testingtargetmax,'ob',rl80,rl80,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl80)+0.5]);
ylim([0 max(rl80)+0.5]);
title(['MLP 80 m Testing ']);
text(2,12,['R^2=' num2str(round(MLPr2test80*perc,2)) ' %'])


% 1/7 WSE
%
LWSEP80 = [LWSEP70; LWSEOutf70train/maxx];
LWSEY80 = trainingtarget';
LWSEPtest80 = [LWSEPtest70; LWSEOutf70'/maxx];
LWSEYtest80 = testingtarget';
LWSEtestingtarget80=LWSEYtest80'*maxx;


alpha=1/3;
outval=LWSEP80(7,:)*((80/70)^(alpha));
outvalmax=outval*maxx;
LWSEOutf80train=outvalmax;


outvaltest=LWSEPtest80(7,:)*((80/70)^(alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget80;
LWSEOutf80=outvaltestmax;
LWSEmsetest80=mse(LWSEOutf80,testingtarget80);
LWSEmapetest80=mape(LWSEOutf80,testingtarget80);
LWSEmbetest80=mbe(LWSEOutf80,testingtarget80);
LWSEr2test80=rsquare(LWSEOutf80,testingtarget80);
LWSEperf80=[LWSEmsetest80 LWSEmapetest80 LWSEmbetest80 LWSEr2test80];
LWSEPtestMax80=LWSEPtest80'*maxx;

figure
interv=1:200;
plot( [testingtarget80(interv) ],'b');
hold on
plot( [RNNOutf80(interv)],'r');
plot( [MLPOutf80(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','MLP est','Location','northwest')

meantarget80=[meantarget70 mean(testingtarget80)];
meanRNN80=[meanRNN70 mean(RNNOutf80)];
meanMLP80=[meanMLP70 mean(MLPOutf80)];
meanLWSE80=[meanLWSE70 mean(LWSEOutf80)];

figure
plot(meantarget80,height80,'k');
hold on
plot(meanRNN80,height80,'b-s');
plot(meanMLP80,height80,'--r');
plot(meanLWSE80,height80,'-.g');

hold off
title('average')
legend('measured','RNN est','MLP est','1/7 est','Location','northwest')

display ('80 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf80]
[MLPperf80]
[LWSEperf80]

%% 90

nex=5;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

P90 = [P80; RNNOutf80train'/maxx];
Y90 = trainingtarget';
Ptest90 = [Ptest80; RNNOutf80'/maxx];
Ytest90 = testingtarget';
testingtarget90=Ytest90'*maxx;


%
%Create NN
numhid=20;
pastval=[0 1];
nn = [inputsize+(nex-1) numhid 1];
dIn = [pastval];
dIntern=[];
dOut=[];
miter=20;
net = CreateNN(nn,dIn,dIntern,dOut);
netLM = train_LM(P90,Y90,net,miter,1e-5);
%Calculate Output of trained net (LM) for training and Test Data
outval = NNOut(P90,netLM);

%outval = netMLP(P);
outvalmax=outval'*maxx;
RNNOutf90train=outvalmax;

trainingtargetmax=trainingtarget*maxx;

msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = NNOut(Ptest90,netLM);

outvaltestmax=outvaltest'*maxx;
RNNOutf90=outvaltestmax;
testingtargetmax=testingtarget*maxx;
msetest90=mse(RNNOutf90,testingtarget90);
mapetest90=mape(RNNOutf90,testingtarget90);
mbetest90=mbe(RNNOutf90,testingtarget90);
r2test90=rsquare(RNNOutf90,testingtarget90);
RNNperf90=[msetest90 mapetest90 mbetest90 r2test90];
target90=testingtarget90;

%
figure
PtestMax90=Ptest90'*maxx;
height90=[height80 90];
plot([PtestMax90'; RNNOutf90' ],height90);
mxr=12.5+nex*0.5;
rang90=[0 mxr];
xlim(rang90)
%ylim([-0.4 0.8])
title(['RNN 90 m Testing MSE=' num2str(msetest90) ',MAPE=' num2str(mapetest90) ',MBE=' num2str(mbetest90) ',R^2=' num2str(r2test90)]);
%
figure
rl90=[1:mxr];
plot( RNNOutf90, target90,'ob',rl90,rl90,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl90)+0.5]);
ylim([0 max(rl90)+0.5]);
title(['RNN 90 m Testing ']);
text(2,12,['R^2=' num2str(round(r2test90*perc,2)) ' %'])

%
MLPP90= [MLPP80; MLPOutf80train'/maxx];
MLPY90 = trainingtarget';
MLPPtest90 = [MLPPtest80; MLPOutf80'/maxx];
MLPYtest90 = testingtarget';
MLPtestingtarget90=MLPYtest90'*maxx;

%Create NN

numiter=20;
numhid=20;
netMLP=feedforwardnet(numhid);
netMLP.divideFcn='divideind';
%netMLP.trainParam.showWindow = 0;
netMLP.divideParam.trainInd=1:max(size(MLPP90)); %training_index
netMLP.trainParam.epochs=numiter;
netMLP.trainParam.mu=3;
netMLP.trainParam.mu_dec=0.1;
netMLP.trainParam.mu_inc =10;
[netMLP,tr,Y,E] = train(netMLP,MLPP90,MLPY90);

outval = netMLP(MLPP90);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netMLP(MLPPtest90);
outvaltestmax=outvaltest'*maxx;
MLPOutf90=outvaltestmax;
MLPtestingtargetmax90=testingtarget90;
MLPmsetest90=mse(MLPOutf90,MLPtestingtargetmax90);
MLPmapetest90=mape(MLPOutf90,MLPtestingtargetmax90);
MLPmbetest90=mbe(MLPOutf90,MLPtestingtargetmax90);
MLPr2test90=rsquare(MLPOutf90,MLPtestingtargetmax90);
MLPperf90=[MLPmsetest90 MLPmapetest90 MLPmbetest90 MLPr2test90];
MLPOutf90train=outvalmax;
%
figure
MLPPtestMax90=MLPPtest90'*maxx;

plot([MLPPtestMax90'; MLPOutf90' ],height90);
xlim(rang90)
%ylim([-0.4 0.8])
title(['MLP 90m Testing MSE=' num2str(MLPmsetest90) ',MAPE=' num2str(MLPmapetest90) ',MBE=' num2str(MLPmbetest90)  ',R^2=' num2str(MLPr2test90)]);
%camroll(90)
% [10 20 30 40 90 90 90 90 90 100 110 120 130 140 190 190 190 190]
%
figure
%rl=[1:13];
plot( MLPOutf90, testingtargetmax,'ob',rl90,rl90,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl90)+0.5]);
ylim([0 max(rl90)+0.5]);
title(['MLP 90 m Testing ']);
text(2,12,['R^2=' num2str(round(MLPr2test90*perc,2)) ' %'])


% 1/7 WSE
%
LWSEP90 = [LWSEP80; LWSEOutf80train/maxx];
LWSEY90 = trainingtarget';
LWSEPtest90 = [LWSEPtest80; LWSEOutf80'/maxx];
LWSEYtest90 = testingtarget';
LWSEtestingtarget90=LWSEYtest90'*maxx;


alpha=1/3;
outval=LWSEP90(8,:)*((90/80)^(alpha));
outvalmax=outval*maxx;
LWSEOutf90train=outvalmax;


outvaltest=LWSEPtest90(8,:)*((90/80)^(alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget90;
LWSEOutf90=outvaltestmax;
LWSEmsetest90=mse(LWSEOutf90,testingtarget90);
LWSEmapetest90=mape(LWSEOutf90,testingtarget90);
LWSEmbetest90=mbe(LWSEOutf90,testingtarget90);
LWSEr2test90=rsquare(LWSEOutf90,testingtarget90);
LWSEperf90=[LWSEmsetest90 LWSEmapetest90 LWSEmbetest90 LWSEr2test90];
LWSEPtestMax90=LWSEPtest90'*maxx;

figure
interv=1:200;
plot( [testingtarget90(interv) ],'b');
hold on
plot( [RNNOutf90(interv)],'r');
plot( [MLPOutf90(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','MLP est','Location','northwest')

meantarget90=[meantarget80 mean(testingtarget90)];
meanRNN90=[meanRNN80 mean(RNNOutf90)];
meanMLP90=[meanMLP80 mean(MLPOutf90)];
meanLWSE90=[meanLWSE80 mean(LWSEOutf90)];

figure
plot(meantarget90,height90,'k');
hold on
plot(meanRNN90,height90,'b-s');
plot(meanMLP90,height90,'--r');
plot(meanLWSE90,height90,'-.g');

hold off
title('average')
legend('measured','RNN est','MLP est','1/7 est','Location','northwest')

display ('90 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf90]
[MLPperf90]
[LWSEperf90]

%% 100

nex=6;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

P100 = [P90; RNNOutf90train'/maxx];
Y100 = trainingtarget';
Ptest100 = [Ptest90; RNNOutf90'/maxx];
Ytest100 = testingtarget';
testingtarget100=Ytest100'*maxx;


%
%Create NN
numhid=20;
pastval=[0 1];
nn = [inputsize+(nex-1) numhid 1];
dIn = [pastval];
dIntern=[];
dOut=[];
miter=20;
net = CreateNN(nn,dIn,dIntern,dOut);
netLM = train_LM(P100,Y100,net,miter,1e-5);
%Calculate Output of trained net (LM) for training and Test Data
outval = NNOut(P100,netLM);

%outval = netMLP(P);
outvalmax=outval'*maxx;
RNNOutf100train=outvalmax;

trainingtargetmax=trainingtarget*maxx;

msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = NNOut(Ptest100,netLM);

outvaltestmax=outvaltest'*maxx;
RNNOutf100=outvaltestmax;
testingtargetmax=testingtarget*maxx;
msetest100=mse(RNNOutf100,testingtarget100);
mapetest100=mape(RNNOutf100,testingtarget100);
mbetest100=mbe(RNNOutf100,testingtarget100);
r2test100=rsquare(RNNOutf100,testingtarget100);
RNNperf100=[msetest100 mapetest100 mbetest100 r2test100];
target100=testingtarget100;

%
figure
PtestMax100=Ptest100'*maxx;
height100=[height90 100];
plot([PtestMax100'; RNNOutf100' ],height100);
mxr=12.5+nex*0.5;
rang100=[0 mxr];
xlim(rang100)
%ylim([-0.4 0.8])
title(['RNN 100 m Testing MSE=' num2str(msetest100) ',MAPE=' num2str(mapetest100) ',MBE=' num2str(mbetest100) ',R^2=' num2str(r2test100)]);
%
figure
rl100=[1:mxr];
plot( RNNOutf100, target100,'ob',rl100,rl100,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl100)+0.5]);
ylim([0 max(rl100)+0.5]);
title(['RNN 100 m Testing ']);
text(2,12,['R^2=' num2str(round(r2test100*perc,2)) ' %'])

%
MLPP100= [MLPP90; MLPOutf90train'/maxx];
MLPY100 = trainingtarget';
MLPPtest100 = [MLPPtest90; MLPOutf90'/maxx];
MLPYtest100 = testingtarget';
MLPtestingtarget100=MLPYtest100'*maxx;

%Create NN

numiter=20;
numhid=20;
netMLP=feedforwardnet(numhid);
netMLP.divideFcn='divideind';
%netMLP.trainParam.showWindow = 0;
netMLP.divideParam.trainInd=1:max(size(MLPP100)); %training_index
netMLP.trainParam.epochs=numiter;
netMLP.trainParam.mu=3;
netMLP.trainParam.mu_dec=0.1;
netMLP.trainParam.mu_inc =10;
[netMLP,tr,Y,E] = train(netMLP,MLPP100,MLPY100);

outval = netMLP(MLPP100);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netMLP(MLPPtest100);
outvaltestmax=outvaltest'*maxx;
MLPOutf100=outvaltestmax;
MLPtestingtargetmax100=testingtarget100;
MLPmsetest100=mse(MLPOutf100,MLPtestingtargetmax100);
MLPmapetest100=mape(MLPOutf100,MLPtestingtargetmax100);
MLPmbetest100=mbe(MLPOutf100,MLPtestingtargetmax100);
MLPr2test100=rsquare(MLPOutf100,MLPtestingtargetmax100);
MLPperf100=[MLPmsetest100 MLPmapetest100 MLPmbetest100 MLPr2test100];
MLPOutf100train=outvalmax;
%
figure
MLPPtestMax100=MLPPtest100'*maxx;

plot([MLPPtestMax100'; MLPOutf100' ],height100);
xlim(rang100)
%ylim([-0.4 0.8])
title(['MLP 100m Testing MSE=' num2str(MLPmsetest100) ',MAPE=' num2str(MLPmapetest100) ',MBE=' num2str(MLPmbetest100)  ',R^2=' num2str(MLPr2test100)]);
%camroll(100)
% [10 20 30 40 100 100 100 100 100 100 110 120 130 140 1100 1100 1100 1100]
%
figure
%rl=[1:13];
plot( MLPOutf100, testingtargetmax,'ob',rl100,rl100,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl100)+0.5]);
ylim([0 max(rl100)+0.5]);
title(['MLP 100 m Testing ']);
text(2,12,['R^2=' num2str(round(MLPr2test100*perc,2)) ' %'])


% 1/7 WSE
%
LWSEP100 = [LWSEP90; LWSEOutf90train/maxx];
LWSEY100 = trainingtarget';
LWSEPtest100 = [LWSEPtest90; LWSEOutf90'/maxx];
LWSEYtest100 = testingtarget';
LWSEtestingtarget100=LWSEYtest100'*maxx;


alpha=1/3;
outval=LWSEP100(9,:)*((100/90)^(alpha));
outvalmax=outval*maxx;
LWSEOutf100train=outvalmax;


outvaltest=LWSEPtest100(9,:)*((100/90)^(alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget100;
LWSEOutf100=outvaltestmax;
LWSEmsetest100=mse(LWSEOutf100,testingtarget100);
LWSEmapetest100=mape(LWSEOutf100,testingtarget100);
LWSEmbetest100=mbe(LWSEOutf100,testingtarget100);
LWSEr2test100=rsquare(LWSEOutf100,testingtarget100);
LWSEperf100=[LWSEmsetest100 LWSEmapetest100 LWSEmbetest100 LWSEr2test100];
LWSEPtestMax100=LWSEPtest100'*maxx;

figure
interv=1:200;
plot( [testingtarget100(interv) ],'b');
hold on
plot( [RNNOutf100(interv)],'r');
plot( [MLPOutf100(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','MLP est','Location','northwest')

meantarget100=[meantarget90 mean(testingtarget100)];
meanRNN100=[meanRNN90 mean(RNNOutf100)];
meanMLP100=[meanMLP90 mean(MLPOutf100)];
meanLWSE100=[meanLWSE90 mean(LWSEOutf100)];

figure
plot(meantarget100,height100,'k');
hold on
plot(meanRNN100,height100,'b-s');
plot(meanMLP100,height100,'--r');
plot(meanLWSE100,height100,'-.g');

hold off
title('average')
legend('measured','RNN est','MLP est','1/7 est','Location','northwest')

display ('100 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf100]
[MLPperf100]
[LWSEperf100]

%% 110

nex=7;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

P110 = [P100; RNNOutf100train'/maxx];
Y110 = trainingtarget';
Ptest110 = [Ptest100; RNNOutf100'/maxx];
Ytest110 = testingtarget';
testingtarget110=Ytest110'*maxx;


%
%Create NN
numhid=20;
pastval=[0 1];
nn = [inputsize+(nex-1) numhid 1];
dIn = [pastval];
dIntern=[];
dOut=[];
miter=20;
net = CreateNN(nn,dIn,dIntern,dOut);
netLM = train_LM(P110,Y110,net,miter,1e-5);
%Calculate Output of trained net (LM) for training and Test Data
outval = NNOut(P110,netLM);

%outval = netMLP(P);
outvalmax=outval'*maxx;
RNNOutf110train=outvalmax;

trainingtargetmax=trainingtarget*maxx;

msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = NNOut(Ptest110,netLM);

outvaltestmax=outvaltest'*maxx;
RNNOutf110=outvaltestmax;
testingtargetmax=testingtarget*maxx;
msetest110=mse(RNNOutf110,testingtarget110);
mapetest110=mape(RNNOutf110,testingtarget110);
mbetest110=mbe(RNNOutf110,testingtarget110);
r2test110=rsquare(RNNOutf110,testingtarget110);
RNNperf110=[msetest110 mapetest110 mbetest110 r2test110];
target110=testingtarget110;

%
figure
PtestMax110=Ptest110'*maxx;
height110=[height100 110];
plot([PtestMax110'; RNNOutf110' ],height110);
mxr=12.5+nex*0.5;
rang110=[0 mxr];
xlim(rang110)
%ylim([-0.4 0.8])
title(['RNN 110 m Testing MSE=' num2str(msetest110) ',MAPE=' num2str(mapetest110) ',MBE=' num2str(mbetest110) ',R^2=' num2str(r2test110)]);
%
figure
rl110=[1:mxr];
plot( RNNOutf110, target110,'ob',rl110,rl110,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl110)+0.5]);
ylim([0 max(rl110)+0.5]);
title(['RNN 110 m Testing ']);
text(2,12,['R^2=' num2str(round(r2test110*perc,2)) ' %'])

%
MLPP110= [MLPP100; MLPOutf100train'/maxx];
MLPY110 = trainingtarget';
MLPPtest110 = [MLPPtest100; MLPOutf100'/maxx];
MLPYtest110 = testingtarget';
MLPtestingtarget110=MLPYtest110'*maxx;

%Create NN

numiter=20;
numhid=20;
netMLP=feedforwardnet(numhid);
netMLP.divideFcn='divideind';
%netMLP.trainParam.showWindow = 0;
netMLP.divideParam.trainInd=1:max(size(MLPP110)); %training_index
netMLP.trainParam.epochs=numiter;
netMLP.trainParam.mu=3;
netMLP.trainParam.mu_dec=0.1;
netMLP.trainParam.mu_inc =10;
[netMLP,tr,Y,E] = train(netMLP,MLPP110,MLPY110);

outval = netMLP(MLPP110);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netMLP(MLPPtest110);
outvaltestmax=outvaltest'*maxx;
MLPOutf110=outvaltestmax;
MLPtestingtargetmax110=testingtarget110;
MLPmsetest110=mse(MLPOutf110,MLPtestingtargetmax110);
MLPmapetest110=mape(MLPOutf110,MLPtestingtargetmax110);
MLPmbetest110=mbe(MLPOutf110,MLPtestingtargetmax110);
MLPr2test110=rsquare(MLPOutf110,MLPtestingtargetmax110);
MLPperf110=[MLPmsetest110 MLPmapetest110 MLPmbetest110 MLPr2test110];
MLPOutf110train=outvalmax;
%
figure
MLPPtestMax110=MLPPtest110'*maxx;

plot([MLPPtestMax110'; MLPOutf110' ],height110);
xlim(rang110)
%ylim([-0.4 0.8])
title(['MLP 110m Testing MSE=' num2str(MLPmsetest110) ',MAPE=' num2str(MLPmapetest110) ',MBE=' num2str(MLPmbetest110)  ',R^2=' num2str(MLPr2test110)]);
%camroll(110)
% [10 20 30 40 110 110 110 110 110 110 110 120 130 140 1110 1110 1110 1110]
%
figure
%rl=[1:13];
plot( MLPOutf110, testingtargetmax,'ob',rl110,rl110,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl110)+0.5]);
ylim([0 max(rl110)+0.5]);
title(['MLP 110 m Testing ']);
text(2,12,['R^2=' num2str(round(MLPr2test110*perc,2)) ' %'])


% 1/7 WSE
%
LWSEP110 = [LWSEP100; LWSEOutf100train/maxx];
LWSEY110 = trainingtarget';
LWSEPtest110 = [LWSEPtest100; LWSEOutf100'/maxx];
LWSEYtest110 = testingtarget';
LWSEtestingtarget110=LWSEYtest110'*maxx;


alpha=1/3;
indl=10;
outval=LWSEP110(indl,:)*((110/100)^(alpha));
outvalmax=outval*maxx;
LWSEOutf110train=outvalmax;


outvaltest=LWSEPtest110(indl,:)*((110/100)^(alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget110;
LWSEOutf110=outvaltestmax;
LWSEmsetest110=mse(LWSEOutf110,testingtarget110);
LWSEmapetest110=mape(LWSEOutf110,testingtarget110);
LWSEmbetest110=mbe(LWSEOutf110,testingtarget110);
LWSEr2test110=rsquare(LWSEOutf110,testingtarget110);
LWSEperf110=[LWSEmsetest110 LWSEmapetest110 LWSEmbetest110 LWSEr2test110];
LWSEPtestMax110=LWSEPtest110'*maxx;

figure
interv=1:200;
plot( [testingtarget110(interv) ],'b');
hold on
plot( [RNNOutf110(interv)],'r');
plot( [MLPOutf110(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','MLP est','Location','northwest')

meantarget110=[meantarget100 mean(testingtarget110)];
meanRNN110=[meanRNN100 mean(RNNOutf110)];
meanMLP110=[meanMLP100 mean(MLPOutf110)];
meanLWSE110=[meanLWSE100 mean(LWSEOutf110)];

figure
plot(meantarget110,height110,'k');
hold on
plot(meanRNN110,height110,'b-s');
plot(meanMLP110,height110,'--r');
plot(meanLWSE110,height110,'-.g');

hold off
title('average')
legend('measured','RNN est','MLP est','1/7 est','Location','northwest')

display ('110 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf110]
[MLPperf110]
[LWSEperf110]

%% 120

nex=8;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

P120 = [P110; RNNOutf110train'/maxx];
Y120 = trainingtarget';
Ptest120 = [Ptest110; RNNOutf110'/maxx];
Ytest120 = testingtarget';
testingtarget120=Ytest120'*maxx;


%
%Create NN
numhid=20;
pastval=[0 1];
nn = [inputsize+(nex-1) numhid 1];
dIn = [pastval];
dIntern=[];
dOut=[];
miter=20;
net = CreateNN(nn,dIn,dIntern,dOut);
netLM = train_LM(P120,Y120,net,miter,1e-5);
%Calculate Output of trained net (LM) for training and Test Data
outval = NNOut(P120,netLM);

%outval = netMLP(P);
outvalmax=outval'*maxx;
RNNOutf120train=outvalmax;

trainingtargetmax=trainingtarget*maxx;

msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = NNOut(Ptest120,netLM);

outvaltestmax=outvaltest'*maxx;
RNNOutf120=outvaltestmax;
testingtargetmax=testingtarget*maxx;
msetest120=mse(RNNOutf120,testingtarget120);
mapetest120=mape(RNNOutf120,testingtarget120);
mbetest120=mbe(RNNOutf120,testingtarget120);
r2test120=rsquare(RNNOutf120,testingtarget120);
RNNperf120=[msetest120 mapetest120 mbetest120 r2test120];
target120=testingtarget120;

%
figure
PtestMax120=Ptest120'*maxx;
height120=[height110 120];
plot([PtestMax120'; RNNOutf120' ],height120);
mxr=12.5+nex*0.5;
rang120=[0 mxr];
xlim(rang120)
%ylim([-0.4 0.8])
title(['RNN 120 m Testing MSE=' num2str(msetest120) ',MAPE=' num2str(mapetest120) ',MBE=' num2str(mbetest120) ',R^2=' num2str(r2test120)]);
%
figure
rl120=[1:mxr];
plot( RNNOutf120, target120,'ob',rl120,rl120,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl120)+0.5]);
ylim([0 max(rl120)+0.5]);
title(['RNN 120 m Testing ']);
text(2,12,['R^2=' num2str(round(r2test120*perc,2)) ' %'])

%
MLPP120= [MLPP110; MLPOutf110train'/maxx];
MLPY120 = trainingtarget';
MLPPtest120 = [MLPPtest110; MLPOutf110'/maxx];
MLPYtest120 = testingtarget';
MLPtestingtarget120=MLPYtest120'*maxx;

%Create NN

numiter=20;
numhid=20;
netMLP=feedforwardnet(numhid);
netMLP.divideFcn='divideind';
%netMLP.trainParam.showWindow = 0;
netMLP.divideParam.trainInd=1:max(size(MLPP120)); %training_index
netMLP.trainParam.epochs=numiter;
netMLP.trainParam.mu=3;
netMLP.trainParam.mu_dec=0.1;
netMLP.trainParam.mu_inc =10;
[netMLP,tr,Y,E] = train(netMLP,MLPP120,MLPY120);

outval = netMLP(MLPP120);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netMLP(MLPPtest120);
outvaltestmax=outvaltest'*maxx;
MLPOutf120=outvaltestmax;
MLPtestingtargetmax120=testingtarget120;
MLPmsetest120=mse(MLPOutf120,MLPtestingtargetmax120);
MLPmapetest120=mape(MLPOutf120,MLPtestingtargetmax120);
MLPmbetest120=mbe(MLPOutf120,MLPtestingtargetmax120);
MLPr2test120=rsquare(MLPOutf120,MLPtestingtargetmax120);
MLPperf120=[MLPmsetest120 MLPmapetest120 MLPmbetest120 MLPr2test120];
MLPOutf120train=outvalmax;
%
figure
MLPPtestMax120=MLPPtest120'*maxx;

plot([MLPPtestMax120'; MLPOutf120' ],height120);
xlim(rang120)
%ylim([-0.4 0.8])
title(['MLP 120m Testing MSE=' num2str(MLPmsetest120) ',MAPE=' num2str(MLPmapetest120) ',MBE=' num2str(MLPmbetest120)  ',R^2=' num2str(MLPr2test120)]);
%camroll(120)
% [10 20 30 40 120 120 120 120 120 120 120 120 130 140 1120 1120 1120 1120]
%
figure
%rl=[1:13];
plot( MLPOutf120, testingtargetmax,'ob',rl120,rl120,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl120)+0.5]);
ylim([0 max(rl120)+0.5]);
title(['MLP 120 m Testing ']);
text(2,12,['R^2=' num2str(round(MLPr2test120*perc,2)) ' %'])


% 1/7 WSE
%
LWSEP120 = [LWSEP110; LWSEOutf110train/maxx];
LWSEY120 = trainingtarget';
LWSEPtest120 = [LWSEPtest110; LWSEOutf110'/maxx];
LWSEYtest120 = testingtarget';
LWSEtestingtarget120=LWSEYtest120'*maxx;


alpha=1/3;
indl=nex+3;
outval=LWSEP120(indl,:)*((120/110)^(alpha));
outvalmax=outval*maxx;
LWSEOutf120train=outvalmax;


outvaltest=LWSEPtest120(indl,:)*((120/110)^(alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget120;
LWSEOutf120=outvaltestmax;
LWSEmsetest120=mse(LWSEOutf120,testingtarget120);
LWSEmapetest120=mape(LWSEOutf120,testingtarget120);
LWSEmbetest120=mbe(LWSEOutf120,testingtarget120);
LWSEr2test120=rsquare(LWSEOutf120,testingtarget120);
LWSEperf120=[LWSEmsetest120 LWSEmapetest120 LWSEmbetest120 LWSEr2test120];
LWSEPtestMax120=LWSEPtest120'*maxx;

figure
interv=1:200;
plot( [testingtarget120(interv) ],'b');
hold on
plot( [RNNOutf120(interv)],'r');
plot( [MLPOutf120(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','MLP est','Location','northwest')

meantarget120=[meantarget110 mean(testingtarget120)];
meanRNN120=[meanRNN110 mean(RNNOutf120)];
meanMLP120=[meanMLP110 mean(MLPOutf120)];
meanLWSE120=[meanLWSE110 mean(LWSEOutf120)];

figure
plot(meantarget120,height120,'k');
hold on
plot(meanRNN120,height120,'b-s');
plot(meanMLP120,height120,'--r');
plot(meanLWSE120,height120,'-.g');

hold off
title('average')
legend('measured','RNN est','MLP est','1/7 est','Location','northwest')

display ('120 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf120]
[MLPperf120]
[LWSEperf120]

%% 130

nex=9;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

P130 = [P120; RNNOutf120train'/maxx];
Y130 = trainingtarget';
Ptest130 = [Ptest120; RNNOutf120'/maxx];
Ytest130 = testingtarget';
testingtarget130=Ytest130'*maxx;


%
%Create NN
numhid=20;
pastval=[0 1];
nn = [inputsize+(nex-1) numhid 1];
dIn = [pastval];
dIntern=[];
dOut=[];
miter=20;
net = CreateNN(nn,dIn,dIntern,dOut);
netLM = train_LM(P130,Y130,net,miter,1e-5);
%Calculate Output of trained net (LM) for training and Test Data
outval = NNOut(P130,netLM);

%outval = netMLP(P);
outvalmax=outval'*maxx;
RNNOutf130train=outvalmax;

trainingtargetmax=trainingtarget*maxx;

msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = NNOut(Ptest130,netLM);

outvaltestmax=outvaltest'*maxx;
RNNOutf130=outvaltestmax;
testingtargetmax=testingtarget*maxx;
msetest130=mse(RNNOutf130,testingtarget130);
mapetest130=mape(RNNOutf130,testingtarget130);
mbetest130=mbe(RNNOutf130,testingtarget130);
r2test130=rsquare(RNNOutf130,testingtarget130);
RNNperf130=[msetest130 mapetest130 mbetest130 r2test130];
target130=testingtarget130;

%
figure
PtestMax130=Ptest130'*maxx;
height130=[height120 130];
plot([PtestMax130'; RNNOutf130' ],height130);
mxr=12.5+nex*0.5;
rang130=[0 mxr];
xlim(rang130)
%ylim([-0.4 0.8])
title(['RNN 130 m Testing MSE=' num2str(msetest130) ',MAPE=' num2str(mapetest130) ',MBE=' num2str(mbetest130) ',R^2=' num2str(r2test130)]);
%
figure
rl130=[1:mxr];
plot( RNNOutf130, target130,'ob',rl130,rl130,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl130)+0.5]);
ylim([0 max(rl130)+0.5]);
title(['RNN 130 m Testing ']);
text(2,12,['R^2=' num2str(round(r2test130*perc,2)) ' %'])

%
MLPP130= [MLPP120; MLPOutf120train'/maxx];
MLPY130 = trainingtarget';
MLPPtest130 = [MLPPtest120; MLPOutf120'/maxx];
MLPYtest130 = testingtarget';
MLPtestingtarget130=MLPYtest130'*maxx;

%Create NN

numiter=20;
numhid=20;
netMLP=feedforwardnet(numhid);
netMLP.divideFcn='divideind';
%netMLP.trainParam.showWindow = 0;
netMLP.divideParam.trainInd=1:max(size(MLPP130)); %training_index
netMLP.trainParam.epochs=numiter;
netMLP.trainParam.mu=3;
netMLP.trainParam.mu_dec=0.1;
netMLP.trainParam.mu_inc =10;
[netMLP,tr,Y,E] = train(netMLP,MLPP130,MLPY130);

outval = netMLP(MLPP130);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netMLP(MLPPtest130);
outvaltestmax=outvaltest'*maxx;
MLPOutf130=outvaltestmax;
MLPtestingtargetmax130=testingtarget130;
MLPmsetest130=mse(MLPOutf130,MLPtestingtargetmax130);
MLPmapetest130=mape(MLPOutf130,MLPtestingtargetmax130);
MLPmbetest130=mbe(MLPOutf130,MLPtestingtargetmax130);
MLPr2test130=rsquare(MLPOutf130,MLPtestingtargetmax130);
MLPperf130=[MLPmsetest130 MLPmapetest130 MLPmbetest130 MLPr2test130];
MLPOutf130train=outvalmax;
%
figure
MLPPtestMax130=MLPPtest130'*maxx;

plot([MLPPtestMax130'; MLPOutf130' ],height130);
xlim(rang130)
%ylim([-0.4 0.8])
title(['MLP 130m Testing MSE=' num2str(MLPmsetest130) ',MAPE=' num2str(MLPmapetest130) ',MBE=' num2str(MLPmbetest130)  ',R^2=' num2str(MLPr2test130)]);
%camroll(130)
% [10 20 30 40 130 130 130 130 130 130 130 130 130 140 1130 1130 1130 1130]
%
figure
%rl=[1:13];
plot( MLPOutf130, testingtargetmax,'ob',rl130,rl130,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl130)+0.5]);
ylim([0 max(rl130)+0.5]);
title(['MLP 130 m Testing ']);
text(2,12,['R^2=' num2str(round(MLPr2test130*perc,2)) ' %'])


% 1/7 WSE
%
LWSEP130 = [LWSEP120; LWSEOutf120train/maxx];
LWSEY130 = trainingtarget';
LWSEPtest130 = [LWSEPtest120; LWSEOutf120'/maxx];
LWSEYtest130 = testingtarget';
LWSEtestingtarget130=LWSEYtest130'*maxx;


alpha=1/3;
indl=nex+3;
outval=LWSEP130(indl,:)*((130/120)^(alpha));
outvalmax=outval*maxx;
LWSEOutf130train=outvalmax;


outvaltest=LWSEPtest130(indl,:)*((130/120)^(alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget130;
LWSEOutf130=outvaltestmax;
LWSEmsetest130=mse(LWSEOutf130,testingtarget130);
LWSEmapetest130=mape(LWSEOutf130,testingtarget130);
LWSEmbetest130=mbe(LWSEOutf130,testingtarget130);
LWSEr2test130=rsquare(LWSEOutf130,testingtarget130);
LWSEperf130=[LWSEmsetest130 LWSEmapetest130 LWSEmbetest130 LWSEr2test130];
LWSEPtestMax130=LWSEPtest130'*maxx;

figure
interv=1:200;
plot( [testingtarget130(interv) ],'b');
hold on
plot( [RNNOutf130(interv)],'r');
plot( [MLPOutf130(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','MLP est','Location','northwest')

meantarget130=[meantarget120 mean(testingtarget130)];
meanRNN130=[meanRNN120 mean(RNNOutf130)];
meanMLP130=[meanMLP120 mean(MLPOutf130)];
meanLWSE130=[meanLWSE120 mean(LWSEOutf130)];

figure
plot(meantarget130,height130,'k');
hold on
plot(meanRNN130,height130,'b-s');
plot(meanMLP130,height130,'--r');
plot(meanLWSE130,height130,'-.g');

hold off
title('average')
legend('measured','RNN est','MLP est','1/7 est','Location','northwest')

display ('130 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf130]
[MLPperf130]
[LWSEperf130]

%% 140

nex=10;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

P140 = [P130; RNNOutf130train'/maxx];
Y140 = trainingtarget';
Ptest140 = [Ptest130; RNNOutf130'/maxx];
Ytest140 = testingtarget';
testingtarget140=Ytest140'*maxx;


%
%Create NN
numhid=20;
pastval=[0 1];
nn = [inputsize+(nex-1) numhid 1];
dIn = [pastval];
dIntern=[];
dOut=[];
miter=20;
net = CreateNN(nn,dIn,dIntern,dOut);
netLM = train_LM(P140,Y140,net,miter,1e-5);
%Calculate Output of trained net (LM) for training and Test Data
outval = NNOut(P140,netLM);

%outval = netMLP(P);
outvalmax=outval'*maxx;
RNNOutf140train=outvalmax;

trainingtargetmax=trainingtarget*maxx;

msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = NNOut(Ptest140,netLM);

outvaltestmax=outvaltest'*maxx;
RNNOutf140=outvaltestmax;
testingtargetmax=testingtarget*maxx;
msetest140=mse(RNNOutf140,testingtarget140);
mapetest140=mape(RNNOutf140,testingtarget140);
mbetest140=mbe(RNNOutf140,testingtarget140);
r2test140=rsquare(RNNOutf140,testingtarget140);
RNNperf140=[msetest140 mapetest140 mbetest140 r2test140];
target140=testingtarget140;

%
figure
PtestMax140=Ptest140'*maxx;
height140=[height130 140];
plot([PtestMax140'; RNNOutf140' ],height140);
mxr=12.5+nex*0.5;
rang140=[0 mxr];
xlim(rang140)
%ylim([-0.4 0.8])
title(['RNN 140 m Testing MSE=' num2str(msetest140) ',MAPE=' num2str(mapetest140) ',MBE=' num2str(mbetest140) ',R^2=' num2str(r2test140)]);
%
figure
rl140=[1:mxr];
plot( RNNOutf140, target140,'ob',rl140,rl140,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl140)+0.5]);
ylim([0 max(rl140)+0.5]);
title(['RNN 140 m Testing ']);
text(2,12,['R^2=' num2str(round(r2test140*perc,2)) ' %'])

%
MLPP140= [MLPP130; MLPOutf130train'/maxx];
MLPY140 = trainingtarget';
MLPPtest140 = [MLPPtest130; MLPOutf130'/maxx];
MLPYtest140 = testingtarget';
MLPtestingtarget140=MLPYtest140'*maxx;

%Create NN

numiter=20;
numhid=20;
netMLP=feedforwardnet(numhid);
netMLP.divideFcn='divideind';
%netMLP.trainParam.showWindow = 0;
netMLP.divideParam.trainInd=1:max(size(MLPP140)); %training_index
netMLP.trainParam.epochs=numiter;
netMLP.trainParam.mu=3;
netMLP.trainParam.mu_dec=0.1;
netMLP.trainParam.mu_inc =10;
[netMLP,tr,Y,E] = train(netMLP,MLPP140,MLPY140);

outval = netMLP(MLPP140);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netMLP(MLPPtest140);
outvaltestmax=outvaltest'*maxx;
MLPOutf140=outvaltestmax;
MLPtestingtargetmax140=testingtarget140;
MLPmsetest140=mse(MLPOutf140,MLPtestingtargetmax140);
MLPmapetest140=mape(MLPOutf140,MLPtestingtargetmax140);
MLPmbetest140=mbe(MLPOutf140,MLPtestingtargetmax140);
MLPr2test140=rsquare(MLPOutf140,MLPtestingtargetmax140);
MLPperf140=[MLPmsetest140 MLPmapetest140 MLPmbetest140 MLPr2test140];
MLPOutf140train=outvalmax;
%
figure
MLPPtestMax140=MLPPtest140'*maxx;

plot([MLPPtestMax140'; MLPOutf140' ],height140);
xlim(rang140)
%ylim([-0.4 0.8])
title(['MLP 140m Testing MSE=' num2str(MLPmsetest140) ',MAPE=' num2str(MLPmapetest140) ',MBE=' num2str(MLPmbetest140)  ',R^2=' num2str(MLPr2test140)]);
%camroll(140)
% [10 20 30 40 140 140 140 140 140 140 140 140 140 140 1140 1140 1140 1140]
%
figure
%rl=[1:13];
plot( MLPOutf140, testingtargetmax,'ob',rl140,rl140,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl140)+0.5]);
ylim([0 max(rl140)+0.5]);
title(['MLP 140 m Testing ']);
text(2,12,['R^2=' num2str(round(MLPr2test140*perc,2)) ' %'])


% 1/7 WSE
%
LWSEP140 = [LWSEP130; LWSEOutf130train/maxx];
LWSEY140 = trainingtarget';
LWSEPtest140 = [LWSEPtest130; LWSEOutf130'/maxx];
LWSEYtest140 = testingtarget';
LWSEtestingtarget140=LWSEYtest140'*maxx;


alpha=1/3;
indl=nex+3;
outval=LWSEP140(indl,:)*((140/130)^(alpha));
outvalmax=outval*maxx;
LWSEOutf140train=outvalmax;


outvaltest=LWSEPtest140(indl,:)*((140/130)^(alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget140;
LWSEOutf140=outvaltestmax;
LWSEmsetest140=mse(LWSEOutf140,testingtarget140);
LWSEmapetest140=mape(LWSEOutf140,testingtarget140);
LWSEmbetest140=mbe(LWSEOutf140,testingtarget140);
LWSEr2test140=rsquare(LWSEOutf140,testingtarget140);
LWSEperf140=[LWSEmsetest140 LWSEmapetest140 LWSEmbetest140 LWSEr2test140];
LWSEPtestMax140=LWSEPtest140'*maxx;

figure
interv=1:200;
plot( [testingtarget140(interv) ],'b');
hold on
plot( [RNNOutf140(interv)],'r');
plot( [MLPOutf140(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','MLP est','Location','northwest')

meantarget140=[meantarget130 mean(testingtarget140)];
meanRNN140=[meanRNN130 mean(RNNOutf140)];
meanMLP140=[meanMLP130 mean(MLPOutf140)];
meanLWSE140=[meanLWSE130 mean(LWSEOutf140)];

figure
plot(meantarget140,height140,'k');
hold on
plot(meanRNN140,height140,'b-s');
plot(meanMLP140,height140,'--r');
plot(meanLWSE140,height140,'-.g');

hold off
title('average')
legend('measured','RNN est','MLP est','1/7 est','Location','northwest')

display ('140 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf140]
[MLPperf140]
[LWSEperf140]


%% 150

nex=11;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

P150 = [P140; RNNOutf140train'/maxx];
Y150 = trainingtarget';
Ptest150 = [Ptest140; RNNOutf140'/maxx];
Ytest150 = testingtarget';
testingtarget150=Ytest150'*maxx;


%
%Create NN
numhid=20;
pastval=[0 1];
nn = [inputsize+(nex-1) numhid 1];
dIn = [pastval];
dIntern=[];
dOut=[];
miter=20;
net = CreateNN(nn,dIn,dIntern,dOut);
netLM = train_LM(P150,Y150,net,miter,1e-5);
%Calculate Output of trained net (LM) for training and Test Data
outval = NNOut(P150,netLM);

%outval = netMLP(P);
outvalmax=outval'*maxx;
RNNOutf150train=outvalmax;

trainingtargetmax=trainingtarget*maxx;

msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = NNOut(Ptest150,netLM);

outvaltestmax=outvaltest'*maxx;
RNNOutf150=outvaltestmax;
testingtargetmax=testingtarget*maxx;
msetest150=mse(RNNOutf150,testingtarget150);
mapetest150=mape(RNNOutf150,testingtarget150);
mbetest150=mbe(RNNOutf150,testingtarget150);
r2test150=rsquare(RNNOutf150,testingtarget150);
RNNperf150=[msetest150 mapetest150 mbetest150 r2test150];
target150=testingtarget150;

%
figure
PtestMax150=Ptest150'*maxx;
height150=[height140 150];
plot([PtestMax150'; RNNOutf150' ],height150);
mxr=12.5+nex*0.5;
rang150=[0 mxr];
xlim(rang150)
%ylim([-0.4 0.8])
title(['RNN 150 m Testing MSE=' num2str(msetest150) ',MAPE=' num2str(mapetest150) ',MBE=' num2str(mbetest150) ',R^2=' num2str(r2test150)]);
%
figure
rl150=[1:mxr];
plot( RNNOutf150, target150,'ob',rl150,rl150,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl150)+0.5]);
ylim([0 max(rl150)+0.5]);
title(['RNN 150 m Testing ']);
text(2,12,['R^2=' num2str(round(r2test150*perc,2)) ' %'])

%
MLPP150= [MLPP140; MLPOutf140train'/maxx];
MLPY150 = trainingtarget';
MLPPtest150 = [MLPPtest140; MLPOutf140'/maxx];
MLPYtest150 = testingtarget';
MLPtestingtarget150=MLPYtest150'*maxx;

%Create NN

numiter=20;
numhid=20;
netMLP=feedforwardnet(numhid);
netMLP.divideFcn='divideind';
%netMLP.trainParam.showWindow = 0;
netMLP.divideParam.trainInd=1:max(size(MLPP150)); %training_index
netMLP.trainParam.epochs=numiter;
netMLP.trainParam.mu=3;
netMLP.trainParam.mu_dec=0.1;
netMLP.trainParam.mu_inc =10;
[netMLP,tr,Y,E] = train(netMLP,MLPP150,MLPY150);

outval = netMLP(MLPP150);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netMLP(MLPPtest150);
outvaltestmax=outvaltest'*maxx;
MLPOutf150=outvaltestmax;
MLPtestingtargetmax150=testingtarget150;
MLPmsetest150=mse(MLPOutf150,MLPtestingtargetmax150);
MLPmapetest150=mape(MLPOutf150,MLPtestingtargetmax150);
MLPmbetest150=mbe(MLPOutf150,MLPtestingtargetmax150);
MLPr2test150=rsquare(MLPOutf150,MLPtestingtargetmax150);
MLPperf150=[MLPmsetest150 MLPmapetest150 MLPmbetest150 MLPr2test150];
MLPOutf150train=outvalmax;
%
figure
MLPPtestMax150=MLPPtest150'*maxx;

plot([MLPPtestMax150'; MLPOutf150' ],height150);
xlim(rang150)
%ylim([-0.4 0.8])
title(['MLP 150m Testing MSE=' num2str(MLPmsetest150) ',MAPE=' num2str(MLPmapetest150) ',MBE=' num2str(MLPmbetest150)  ',R^2=' num2str(MLPr2test150)]);
%camroll(150)
% [10 20 30 40 150 150 150 150 150 150 150 150 150 150 1150 1150 1150 1150]
%
figure
%rl=[1:13];
plot( MLPOutf150, testingtargetmax,'ob',rl150,rl150,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl150)+0.5]);
ylim([0 max(rl150)+0.5]);
title(['MLP 150 m Testing ']);
text(2,12,['R^2=' num2str(round(MLPr2test150*perc,2)) ' %'])


% 1/7 WSE
%
LWSEP150 = [LWSEP140; LWSEOutf140train/maxx];
LWSEY150 = trainingtarget';
LWSEPtest150 = [LWSEPtest140; LWSEOutf140'/maxx];
LWSEYtest150 = testingtarget';
LWSEtestingtarget150=LWSEYtest150'*maxx;


alpha=1/3;
indl=nex+3;
outval=LWSEP150(indl,:)*((150/140)^(alpha));
outvalmax=outval*maxx;
LWSEOutf150train=outvalmax;


outvaltest=LWSEPtest150(indl,:)*((150/140)^(alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget150;
LWSEOutf150=outvaltestmax;
LWSEmsetest150=mse(LWSEOutf150,testingtarget150);
LWSEmapetest150=mape(LWSEOutf150,testingtarget150);
LWSEmbetest150=mbe(LWSEOutf150,testingtarget150);
LWSEr2test150=rsquare(LWSEOutf150,testingtarget150);
LWSEperf150=[LWSEmsetest150 LWSEmapetest150 LWSEmbetest150 LWSEr2test150];
LWSEPtestMax150=LWSEPtest150'*maxx;

figure
interv=1:200;
plot( [testingtarget150(interv) ],'b');
hold on
plot( [RNNOutf150(interv)],'r');
plot( [MLPOutf150(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','MLP est','Location','northwest')

meantarget150=[meantarget140 mean(testingtarget150)];
meanRNN150=[meanRNN140 mean(RNNOutf150)];
meanMLP150=[meanMLP140 mean(MLPOutf150)];
meanLWSE150=[meanLWSE140 mean(LWSEOutf150)];

figure
plot(meantarget150,height150,'k');
hold on
plot(meanRNN150,height150,'b-s');
plot(meanMLP150,height150,'--r');
plot(meanLWSE150,height150,'-.g');

hold off
title('average')
legend('measured','RNN est','MLP est','1/7 est','Location','northwest')

display ('150 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf150]
[MLPperf150]
[LWSEperf150]


%% 160

nex=12;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

P160 = [P150; RNNOutf150train'/maxx];
Y160 = trainingtarget';
Ptest160 = [Ptest150; RNNOutf150'/maxx];
Ytest160 = testingtarget';
testingtarget160=Ytest160'*maxx;


%
%Create NN
numhid=20;
pastval=[0 1];
nn = [inputsize+(nex-1) numhid 1];
dIn = [pastval];
dIntern=[];
dOut=[];
miter=20;
net = CreateNN(nn,dIn,dIntern,dOut);
netLM = train_LM(P160,Y160,net,miter,1e-5);
%Calculate Output of trained net (LM) for training and Test Data
outval = NNOut(P160,netLM);

%outval = netMLP(P);
outvalmax=outval'*maxx;
RNNOutf160train=outvalmax;

trainingtargetmax=trainingtarget*maxx;

msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = NNOut(Ptest160,netLM);

outvaltestmax=outvaltest'*maxx;
RNNOutf160=outvaltestmax;
testingtargetmax=testingtarget*maxx;
msetest160=mse(RNNOutf160,testingtarget160);
mapetest160=mape(RNNOutf160,testingtarget160);
mbetest160=mbe(RNNOutf160,testingtarget160);
r2test160=rsquare(RNNOutf160,testingtarget160);
RNNperf160=[msetest160 mapetest160 mbetest160 r2test160];
target160=testingtarget160;

%
figure
PtestMax160=Ptest160'*maxx;
height160=[height150 160];
plot([PtestMax160'; RNNOutf160' ],height160);
mxr=12.5+nex*0.5;
rang160=[0 mxr];
xlim(rang160)
%ylim([-0.4 0.8])
title(['RNN 160 m Testing MSE=' num2str(msetest160) ',MAPE=' num2str(mapetest160) ',MBE=' num2str(mbetest160) ',R^2=' num2str(r2test160)]);
%
figure
rl160=[1:mxr];
plot( RNNOutf160, target160,'ob',rl160,rl160,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl160)+0.5]);
ylim([0 max(rl160)+0.5]);
title(['RNN 160 m Testing ']);
text(2,12,['R^2=' num2str(round(r2test160*perc,2)) ' %'])

%
MLPP160= [MLPP150; MLPOutf150train'/maxx];
MLPY160 = trainingtarget';
MLPPtest160 = [MLPPtest150; MLPOutf150'/maxx];
MLPYtest160 = testingtarget';
MLPtestingtarget160=MLPYtest160'*maxx;

%Create NN

numiter=20;
numhid=20;
netMLP=feedforwardnet(numhid);
netMLP.divideFcn='divideind';
%netMLP.trainParam.showWindow = 0;
netMLP.divideParam.trainInd=1:max(size(MLPP160)); %training_index
netMLP.trainParam.epochs=numiter;
netMLP.trainParam.mu=3;
netMLP.trainParam.mu_dec=0.1;
netMLP.trainParam.mu_inc =10;
[netMLP,tr,Y,E] = train(netMLP,MLPP160,MLPY160);

outval = netMLP(MLPP160);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netMLP(MLPPtest160);
outvaltestmax=outvaltest'*maxx;
MLPOutf160=outvaltestmax;
MLPtestingtargetmax160=testingtarget160;
MLPmsetest160=mse(MLPOutf160,MLPtestingtargetmax160);
MLPmapetest160=mape(MLPOutf160,MLPtestingtargetmax160);
MLPmbetest160=mbe(MLPOutf160,MLPtestingtargetmax160);
MLPr2test160=rsquare(MLPOutf160,MLPtestingtargetmax160);
MLPperf160=[MLPmsetest160 MLPmapetest160 MLPmbetest160 MLPr2test160];
MLPOutf160train=outvalmax;
%
figure
MLPPtestMax160=MLPPtest160'*maxx;

plot([MLPPtestMax160'; MLPOutf160' ],height160);
xlim(rang160)
%ylim([-0.4 0.8])
title(['MLP 160m Testing MSE=' num2str(MLPmsetest160) ',MAPE=' num2str(MLPmapetest160) ',MBE=' num2str(MLPmbetest160)  ',R^2=' num2str(MLPr2test160)]);
%camroll(160)
% [10 20 30 40 160 160 160 160 160 160 160 160 160 160 1160 1160 1160 1160]
%
figure
%rl=[1:13];
plot( MLPOutf160, testingtargetmax,'ob',rl160,rl160,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl160)+0.5]);
ylim([0 max(rl160)+0.5]);
title(['MLP 160 m Testing ']);
text(2,12,['R^2=' num2str(round(MLPr2test160*perc,2)) ' %'])


% 1/7 WSE
%
LWSEP160 = [LWSEP150; LWSEOutf150train/maxx];
LWSEY160 = trainingtarget';
LWSEPtest160 = [LWSEPtest150; LWSEOutf150'/maxx];
LWSEYtest160 = testingtarget';
LWSEtestingtarget160=LWSEYtest160'*maxx;


alpha=1/3;
indl=nex+3;
outval=LWSEP160(indl,:)*((160/150)^(alpha));
outvalmax=outval*maxx;
LWSEOutf160train=outvalmax;


outvaltest=LWSEPtest160(indl,:)*((160/150)^(alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget160;
LWSEOutf160=outvaltestmax;
LWSEmsetest160=mse(LWSEOutf160,testingtarget160);
LWSEmapetest160=mape(LWSEOutf160,testingtarget160);
LWSEmbetest160=mbe(LWSEOutf160,testingtarget160);
LWSEr2test160=rsquare(LWSEOutf160,testingtarget160);
LWSEperf160=[LWSEmsetest160 LWSEmapetest160 LWSEmbetest160 LWSEr2test160];
LWSEPtestMax160=LWSEPtest160'*maxx;

figure
interv=1:200;
plot( [testingtarget160(interv) ],'b');
hold on
plot( [RNNOutf160(interv)],'r');
plot( [MLPOutf160(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','MLP est','Location','northwest')

meantarget160=[meantarget150 mean(testingtarget160)];
meanRNN160=[meanRNN150 mean(RNNOutf160)];
meanMLP160=[meanMLP150 mean(MLPOutf160)];
meanLWSE160=[meanLWSE150 mean(LWSEOutf160)];

figure
plot(meantarget160,height160,'k');
hold on
plot(meanRNN160,height160,'b-s');
plot(meanMLP160,height160,'--r');
plot(meanLWSE160,height160,'-.g');

hold off
title('average')
legend('measured','RNN est','MLP est','1/7 est','Location','northwest')

display ('160 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf160]
[MLPperf160]
[LWSEperf160]

%% 170

nex=13;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

P170 = [P160; RNNOutf160train'/maxx];
Y170 = trainingtarget';
Ptest170 = [Ptest160; RNNOutf160'/maxx];
Ytest170 = testingtarget';
testingtarget170=Ytest170'*maxx;


%
%Create NN
numhid=20;
pastval=[0 1];
nn = [inputsize+(nex-1) numhid 1];
dIn = [pastval];
dIntern=[];
dOut=[];
miter=20;
net = CreateNN(nn,dIn,dIntern,dOut);
netLM = train_LM(P170,Y170,net,miter,1e-5);
%Calculate Output of trained net (LM) for training and Test Data
outval = NNOut(P170,netLM);

%outval = netMLP(P);
outvalmax=outval'*maxx;
RNNOutf170train=outvalmax;

trainingtargetmax=trainingtarget*maxx;

msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = NNOut(Ptest170,netLM);

outvaltestmax=outvaltest'*maxx;
RNNOutf170=outvaltestmax;
testingtargetmax=testingtarget*maxx;
msetest170=mse(RNNOutf170,testingtarget170);
mapetest170=mape(RNNOutf170,testingtarget170);
mbetest170=mbe(RNNOutf170,testingtarget170);
r2test170=rsquare(RNNOutf170,testingtarget170);
RNNperf170=[msetest170 mapetest170 mbetest170 r2test170];
target170=testingtarget170;

%
figure
PtestMax170=Ptest170'*maxx;
height170=[height160 170];
plot([PtestMax170'; RNNOutf170' ],height170);
mxr=12.5+nex*0.5;
rang170=[0 mxr];
xlim(rang170)
%ylim([-0.4 0.8])
title(['RNN 170 m Testing MSE=' num2str(msetest170) ',MAPE=' num2str(mapetest170) ',MBE=' num2str(mbetest170) ',R^2=' num2str(r2test170)]);
%
figure
rl170=[1:mxr];
plot( RNNOutf170, target170,'ob',rl170,rl170,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl170)+0.5]);
ylim([0 max(rl170)+0.5]);
title(['RNN 170 m Testing ']);
text(2,12,['R^2=' num2str(round(r2test170*perc,2)) ' %'])

%
MLPP170= [MLPP160; MLPOutf160train'/maxx];
MLPY170 = trainingtarget';
MLPPtest170 = [MLPPtest160; MLPOutf160'/maxx];
MLPYtest170 = testingtarget';
MLPtestingtarget170=MLPYtest170'*maxx;

%Create NN

numiter=20;
numhid=20;
netMLP=feedforwardnet(numhid);
netMLP.divideFcn='divideind';
%netMLP.trainParam.showWindow = 0;
netMLP.divideParam.trainInd=1:max(size(MLPP170)); %training_index
netMLP.trainParam.epochs=numiter;
netMLP.trainParam.mu=3;
netMLP.trainParam.mu_dec=0.1;
netMLP.trainParam.mu_inc =10;
[netMLP,tr,Y,E] = train(netMLP,MLPP170,MLPY170);

outval = netMLP(MLPP170);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netMLP(MLPPtest170);
outvaltestmax=outvaltest'*maxx;
MLPOutf170=outvaltestmax;
MLPtestingtargetmax170=testingtarget170;
MLPmsetest170=mse(MLPOutf170,MLPtestingtargetmax170);
MLPmapetest170=mape(MLPOutf170,MLPtestingtargetmax170);
MLPmbetest170=mbe(MLPOutf170,MLPtestingtargetmax170);
MLPr2test170=rsquare(MLPOutf170,MLPtestingtargetmax170);
MLPperf170=[MLPmsetest170 MLPmapetest170 MLPmbetest170 MLPr2test170];
MLPOutf170train=outvalmax;
%
figure
MLPPtestMax170=MLPPtest170'*maxx;

plot([MLPPtestMax170'; MLPOutf170' ],height170);
xlim(rang170)
%ylim([-0.4 0.8])
title(['MLP 170m Testing MSE=' num2str(MLPmsetest170) ',MAPE=' num2str(MLPmapetest170) ',MBE=' num2str(MLPmbetest170)  ',R^2=' num2str(MLPr2test170)]);
%camroll(170)
% [10 20 30 40 170 170 170 170 170 170 170 170 170 170 1170 1170 1170 1170]
%
figure
%rl=[1:13];
plot( MLPOutf170, testingtargetmax,'ob',rl170,rl170,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl170)+0.5]);
ylim([0 max(rl170)+0.5]);
title(['MLP 170 m Testing ']);
text(2,12,['R^2=' num2str(round(MLPr2test170*perc,2)) ' %'])


% 1/7 WSE
%
LWSEP170 = [LWSEP160; LWSEOutf160train/maxx];
LWSEY170 = trainingtarget';
LWSEPtest170 = [LWSEPtest160; LWSEOutf160'/maxx];
LWSEYtest170 = testingtarget';
LWSEtestingtarget170=LWSEYtest170'*maxx;


alpha=1/3;
indl=nex+3;
outval=LWSEP170(indl,:)*((170/160)^(alpha));
outvalmax=outval*maxx;
LWSEOutf170train=outvalmax;


outvaltest=LWSEPtest170(indl,:)*((170/160)^(alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget170;
LWSEOutf170=outvaltestmax;
LWSEmsetest170=mse(LWSEOutf170,testingtarget170);
LWSEmapetest170=mape(LWSEOutf170,testingtarget170);
LWSEmbetest170=mbe(LWSEOutf170,testingtarget170);
LWSEr2test170=rsquare(LWSEOutf170,testingtarget170);
LWSEperf170=[LWSEmsetest170 LWSEmapetest170 LWSEmbetest170 LWSEr2test170];
LWSEPtestMax170=LWSEPtest170'*maxx;

figure
interv=1:200;
plot( [testingtarget170(interv) ],'b');
hold on
plot( [RNNOutf170(interv)],'r');
plot( [MLPOutf170(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','MLP est','Location','northwest')

meantarget170=[meantarget160 mean(testingtarget170)];
meanRNN170=[meanRNN160 mean(RNNOutf170)];
meanMLP170=[meanMLP160 mean(MLPOutf170)];
meanLWSE170=[meanLWSE160 mean(LWSEOutf170)];

figure
plot(meantarget170,height170,'k');
hold on
plot(meanRNN170,height170,'b-s');
plot(meanMLP170,height170,'--r');
plot(meanLWSE170,height170,'-.g');

hold off
title('average')
legend('measured','RNN est','MLP est','1/7 est','Location','northwest')

display ('170 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf170]
[MLPperf170]
[LWSEperf170]


%% 180

nex=14;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

P180 = [P170; RNNOutf170train'/maxx];
Y180 = trainingtarget';
Ptest180 = [Ptest170; RNNOutf170'/maxx];
Ytest180 = testingtarget';
testingtarget180=Ytest180'*maxx;


%
%Create NN
numhid=20;
pastval=[0 1];
nn = [inputsize+(nex-1) numhid 1];
dIn = [pastval];
dIntern=[];
dOut=[];
miter=20;
net = CreateNN(nn,dIn,dIntern,dOut);
netLM = train_LM(P180,Y180,net,miter,1e-5);
%Calculate Output of trained net (LM) for training and Test Data
outval = NNOut(P180,netLM);

%outval = netMLP(P);
outvalmax=outval'*maxx;
RNNOutf180train=outvalmax;

trainingtargetmax=trainingtarget*maxx;

msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = NNOut(Ptest180,netLM);

outvaltestmax=outvaltest'*maxx;
RNNOutf180=outvaltestmax;
testingtargetmax=testingtarget*maxx;
msetest180=mse(RNNOutf180,testingtarget180);
mapetest180=mape(RNNOutf180,testingtarget180);
mbetest180=mbe(RNNOutf180,testingtarget180);
r2test180=rsquare(RNNOutf180,testingtarget180);
RNNperf180=[msetest180 mapetest180 mbetest180 r2test180];
target180=testingtarget180;

%
figure
PtestMax180=Ptest180'*maxx;
height180=[height170 180];
plot([PtestMax180'; RNNOutf180' ],height180);
mxr=12.5+nex*0.5;
rang180=[0 mxr];
xlim(rang180)
%ylim([-0.4 0.8])
title(['RNN 180 m Testing MSE=' num2str(msetest180) ',MAPE=' num2str(mapetest180) ',MBE=' num2str(mbetest180) ',R^2=' num2str(r2test180)]);
%
figure
rl180=[1:mxr];
plot( RNNOutf180, target180,'ob',rl180,rl180,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl180)+0.5]);
ylim([0 max(rl180)+0.5]);
title(['RNN 180 m Testing ']);
text(2,12,['R^2=' num2str(round(r2test180*perc,2)) ' %'])

%
MLPP180= [MLPP170; MLPOutf170train'/maxx];
MLPY180 = trainingtarget';
MLPPtest180 = [MLPPtest170; MLPOutf170'/maxx];
MLPYtest180 = testingtarget';
MLPtestingtarget180=MLPYtest180'*maxx;

%Create NN

numiter=20;
numhid=20;
netMLP=feedforwardnet(numhid);
netMLP.divideFcn='divideind';
%netMLP.trainParam.showWindow = 0;
netMLP.divideParam.trainInd=1:max(size(MLPP180)); %training_index
netMLP.trainParam.epochs=numiter;
netMLP.trainParam.mu=3;
netMLP.trainParam.mu_dec=0.1;
netMLP.trainParam.mu_inc =10;
[netMLP,tr,Y,E] = train(netMLP,MLPP180,MLPY180);

outval = netMLP(MLPP180);
outvalmax=outval'*maxx;
trainingtargetmax=trainingtarget*maxx;
msetrain=mse(outvalmax,trainingtargetmax);
mapetrain=mape(outvalmax,trainingtargetmax);
r2train=rsquare(outvalmax,trainingtargetmax);

outvaltest = netMLP(MLPPtest180);
outvaltestmax=outvaltest'*maxx;
MLPOutf180=outvaltestmax;
MLPtestingtargetmax180=testingtarget180;
MLPmsetest180=mse(MLPOutf180,MLPtestingtargetmax180);
MLPmapetest180=mape(MLPOutf180,MLPtestingtargetmax180);
MLPmbetest180=mbe(MLPOutf180,MLPtestingtargetmax180);
MLPr2test180=rsquare(MLPOutf180,MLPtestingtargetmax180);
MLPperf180=[MLPmsetest180 MLPmapetest180 MLPmbetest180 MLPr2test180];
MLPOutf180train=outvalmax;
%
figure
MLPPtestMax180=MLPPtest180'*maxx;

plot([MLPPtestMax180'; MLPOutf180' ],height180);
xlim(rang180)
%ylim([-0.4 0.8])
title(['MLP 180m Testing MSE=' num2str(MLPmsetest180) ',MAPE=' num2str(MLPmapetest180) ',MBE=' num2str(MLPmbetest180)  ',R^2=' num2str(MLPr2test180)]);
%camroll(180)
% [10 20 30 40 180 180 180 180 180 180 180 180 180 180 1180 1180 1180 1180]
%
figure
%rl=[1:13];
plot( MLPOutf180, testingtargetmax,'ob',rl180,rl180,'r');
xlabel('measured')
ylabel('estimated')
xlim([0 max(rl180)+0.5]);
ylim([0 max(rl180)+0.5]);
title(['MLP 180 m Testing ']);
text(2,12,['R^2=' num2str(round(MLPr2test180*perc,2)) ' %'])


% 1/7 WSE
%
LWSEP180 = [LWSEP170; LWSEOutf170train/maxx];
LWSEY180 = trainingtarget';
LWSEPtest180 = [LWSEPtest170; LWSEOutf170'/maxx];
LWSEYtest180 = testingtarget';
LWSEtestingtarget180=LWSEYtest180'*maxx;


alpha=1/3;
indl=nex+3;
outval=LWSEP180(indl,:)*((180/170)^(alpha));
outvalmax=outval*maxx;
LWSEOutf180train=outvalmax;


outvaltest=LWSEPtest180(indl,:)*((180/170)^(alpha));
outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget180;
LWSEOutf180=outvaltestmax;
LWSEmsetest180=mse(LWSEOutf180,testingtarget180);
LWSEmapetest180=mape(LWSEOutf180,testingtarget180);
LWSEmbetest180=mbe(LWSEOutf180,testingtarget180);
LWSEr2test180=rsquare(LWSEOutf180,testingtarget180);
LWSEperf180=[LWSEmsetest180 LWSEmapetest180 LWSEmbetest180 LWSEr2test180];
LWSEPtestMax180=LWSEPtest180'*maxx;

figure
interv=1:200;
plot( [testingtarget180(interv) ],'b');
hold on
plot( [RNNOutf180(interv)],'r');
plot( [MLPOutf180(interv)],'k:');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','MLP est','Location','northwest')

meantarget180=[meantarget170 mean(testingtarget180)];
meanRNN180=[meanRNN170 mean(RNNOutf180)];
meanMLP180=[meanMLP170 mean(MLPOutf180)];
meanLWSE180=[meanLWSE170 mean(LWSEOutf180)];

figure
plot(meantarget180,height180,'k');
hold on
plot(meanRNN180,height180,'b-s');
plot(meanMLP180,height180,'--r');
plot(meanLWSE180,height180,'-.g');

hold off
title('average')
legend('measured','RNN est','MLP est','1/7 est','Location','northwest')

display ('180 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf180]
[MLPperf180]
[LWSEperf180]
