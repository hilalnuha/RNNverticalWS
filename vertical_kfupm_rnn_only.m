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

% 50 m
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
figure
interv=1:200;
plot( [testingtarget50(interv) ],'b');
hold on
plot( [RNNOutf50(interv)],'r');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','Location','northwest')

meantarget50=mean([seriesT(:,1:inputsize)'*maxx; testingtarget50' ]');
meanRNN50=mean([PtestMax50'; RNNOutf50' ]');
figure
plot(meantarget50,height50,'k');
hold on
plot(meanRNN50,height50,'b-s');

hold off
title('average')
legend('measured','RNN est','Location','northwest')

display ('50 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf50]

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
figure
interv=1:200;
plot( [testingtarget60(interv) ],'b');
hold on
plot( [RNNOutf60(interv)],'r');
hold off
title('estimated and hourly measured')
legend('measured','RNN','Location','northwest')

meantarget60=[meantarget50 mean(testingtarget60)];
meanRNN60=[meanRNN50 mean(RNNOutf60)];

figure
plot(meantarget60,height60,'k');
hold on
plot(meanRNN60,height60,'b-s');

hold off
title('average')
legend('measured','RNN est','Location','northwest')

display ('60 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf60]

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

figure
interv=1:200;
plot( [testingtarget70(interv) ],'b');
hold on
plot( [RNNOutf70(interv)],'r');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','Location','northwest')

meantarget70=[meantarget60 mean(testingtarget70)];
meanRNN70=[meanRNN60 mean(RNNOutf70)];

figure
plot(meantarget70,height70,'k');
hold on
plot(meanRNN70,height70,'b-s');

hold off
title('average')
legend('measured','RNN est','Location','northwest')

display ('70 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf70]

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

figure
interv=1:200;
plot( [testingtarget80(interv) ],'b');
hold on
plot( [RNNOutf80(interv)],'r');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','Location','northwest')

meantarget80=[meantarget70 mean(testingtarget80)];
meanRNN80=[meanRNN70 mean(RNNOutf80)];

figure
plot(meantarget80,height80,'k');
hold on
plot(meanRNN80,height80,'b-s');

hold off
title('average')
legend('measured','RNN est','Location','northwest')

display ('80 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf80]

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


figure
interv=1:200;
plot( [testingtarget90(interv) ],'b');
hold on
plot( [RNNOutf90(interv)],'r');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','Location','northwest')

meantarget90=[meantarget80 mean(testingtarget90)];
meanRNN90=[meanRNN80 mean(RNNOutf90)];

figure
plot(meantarget90,height90,'k');
hold on
plot(meanRNN90,height90,'b-s');

hold off
title('average')
legend('measured','RNN est','Location','northwest')

display ('90 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf90]

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


figure
interv=1:200;
plot( [testingtarget100(interv) ],'b');
hold on
plot( [RNNOutf100(interv)],'r');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','Location','northwest')

meantarget100=[meantarget90 mean(testingtarget100)];
meanRNN100=[meanRNN90 mean(RNNOutf100)];

figure
plot(meantarget100,height100,'k');
hold on
plot(meanRNN100,height100,'b-s');

hold off
title('average')
legend('measured','RNN est','Location','northwest')

display ('100 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf100]

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

figure
interv=1:200;
plot( [testingtarget110(interv) ],'b');
hold on
plot( [RNNOutf110(interv)],'r');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','Location','northwest')

meantarget110=[meantarget100 mean(testingtarget110)];
meanRNN110=[meanRNN100 mean(RNNOutf110)];

figure
plot(meantarget110,height110,'k');
hold on
plot(meanRNN110,height110,'b-s');

hold off
title('average')
legend('measured','RNN est','Location','northwest')

display ('110 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf110]

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


figure
interv=1:200;
plot( [testingtarget120(interv) ],'b');
hold on
plot( [RNNOutf120(interv)],'r');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','Location','northwest')

meantarget120=[meantarget110 mean(testingtarget120)];
meanRNN120=[meanRNN110 mean(RNNOutf120)];

figure
plot(meantarget120,height120,'k');
hold on
plot(meanRNN120,height120,'b-s');

hold off
title('average')
legend('measured','RNN est','Location','northwest')

display ('120 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf120]

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


figure
interv=1:200;
plot( [testingtarget130(interv) ],'b');
hold on
plot( [RNNOutf130(interv)],'r');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','Location','northwest')

meantarget130=[meantarget120 mean(testingtarget130)];
meanRNN130=[meanRNN120 mean(RNNOutf130)];

figure
plot(meantarget130,height130,'k');
hold on
plot(meanRNN130,height130,'b-s');

hold off
title('average')
legend('measured','RNN est','Location','northwest')

display ('130 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf130]

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

figure
interv=1:200;
plot( [testingtarget140(interv) ],'b');
hold on
plot( [RNNOutf140(interv)],'r');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','Location','northwest')

meantarget140=[meantarget130 mean(testingtarget140)];
meanRNN140=[meanRNN130 mean(RNNOutf140)];

figure
plot(meantarget140,height140,'k');
hold on
plot(meanRNN140,height140,'b-s');

hold off
title('average')
legend('measured','RNN est','Location','northwest')

display ('140 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf140]


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


figure
interv=1:200;
plot( [testingtarget150(interv) ],'b');
hold on
plot( [RNNOutf150(interv)],'r');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','Location','northwest')

meantarget150=[meantarget140 mean(testingtarget150)];
meanRNN150=[meanRNN140 mean(RNNOutf150)];

figure
plot(meantarget150,height150,'k');
hold on
plot(meanRNN150,height150,'b-s');

hold off
title('average')
legend('measured','RNN est','Location','northwest')

display ('150 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf150]


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

figure
interv=1:200;
plot( [testingtarget160(interv) ],'b');
hold on
plot( [RNNOutf160(interv)],'r');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','Location','northwest')

meantarget160=[meantarget150 mean(testingtarget160)];
meanRNN160=[meanRNN150 mean(RNNOutf160)];

figure
plot(meantarget160,height160,'k');
hold on
plot(meanRNN160,height160,'b-s');

hold off
title('average')
legend('measured','RNN est','Location','northwest')

display ('160 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf160]

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

figure
interv=1:200;
plot( [testingtarget170(interv) ],'b');
hold on
plot( [RNNOutf170(interv)],'r');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','Location','northwest')

meantarget170=[meantarget160 mean(testingtarget170)];
meanRNN170=[meanRNN160 mean(RNNOutf170)];

figure
plot(meantarget170,height170,'k');
hold on
plot(meanRNN170,height170,'b-s');

hold off
title('average')
legend('measured','RNN est','Location','northwest')

display ('170 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf170]


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

figure
interv=1:200;
plot( [testingtarget180(interv) ],'b');
hold on
plot( [RNNOutf180(interv)],'r');
hold off
title('estimated and hourly measured')
legend('measured','RNN est','Location','northwest')

meantarget180=[meantarget170 mean(testingtarget180)];
meanRNN180=[meanRNN170 mean(RNNOutf180)];

figure
plot(meantarget180,height180,'k');
hold on
plot(meanRNN180,height180,'b-s');

hold off
title('average')
legend('measured','RNN est','Location','northwest')

display ('180 m Testing:     MSE     MAPE        MBE        R2')
[RNNperf180]
RNN_perf_all=[RNNperf50; RNNperf60; RNNperf70; RNNperf80; RNNperf90; RNNperf100; RNNperf110; RNNperf120; RNNperf130; RNNperf140; RNNperf150; RNNperf160; RNNperf170; RNNperf180];