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
Nhid=5;
Rrr=0.0000001;
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

%outval = netSVM(P);

trainingtargetmax=trainingtarget*maxx;

height50=[10 20 30 40 50];
rang50=[0 13];
rl50=[1:13];
% SVM WSE

SVMP50 = traininginput';
SVMY50 = trainingtarget';
SVMPtest50 = testinginput';
SVMYtest50 = testingtarget';
SVMtestingtarget50=SVMYtest50'*maxx;

%netSVM = fitrSVM(SVMP50',SVMY50,'OptimizeHyperparameters','epsilon');
%netSVM = fitlm(SVMP50',SVMY50,'interactions'	);
netSVM = fitrsvm(SVMP50',SVMY50);
%netSVM = stepwisefit(SVMP50',SVMY50');
%netSVM = fitrsvm(SVMP50',SVMY50	);
outval = (predict(netSVM,SVMP50'));

outvalmax=outval*maxx;
SVMOutf50train=outvalmax';
%mse(SVMOutf50train,SVMY50*maxx)
%outvaltest=(sigmoid(Ww*SVMPtest50)'*Beta)';
outvaltest=(predict(netSVM,SVMPtest50'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget50;
SVMOutf50=outvaltestmax;
SVMmsetest50=mse(SVMOutf50,testingtarget50);
SVMmapetest50=mape(SVMOutf50,testingtarget50);
SVMmbetest50=mbe(SVMOutf50,testingtarget50);
SVMr2test50=rsquare(SVMOutf50,testingtarget50);
SVMperf50=[SVMmsetest50 SVMmapetest50 SVMmbetest50 SVMr2test50];
SVMPtestMax50=SVMPtest50'*maxx;

meantarget50=mean([seriesT(:,1:inputsize)'*maxx; testingtarget50' ]');
meanSVM50=mean([SVMPtestMax50'; SVMOutf50' ]');
figure
plot(meantarget50,height50,'k');
hold on
plot(meanSVM50,height50,'-.g');

hold off
title('average')
legend('measured','SVM est','Location','northwest')

display ('50 m Testing:     MSE     MAPE        MBE        R2')
[SVMperf50]
SVMperfall=[mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 60

nex=2;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y60 = trainingtarget';
Ytest60 = testingtarget';
testingtarget60=Ytest60'*maxx;

testingtargetmax=testingtarget*maxx;
target60=testingtarget60;

%
height60=[height50 60];
mxr=12.5+nex*0.5;
rang60=[0 mxr];
rl60=[1:mxr];
% SVM WSE
%
SVMP60 = [SVMP50; SVMOutf50train/maxx];
SVMY60 = trainingtarget';
SVMPtest60 = [SVMPtest50; SVMOutf50'/maxx];
SVMYtest60 = testingtarget';
SVMtestingtarget60=SVMYtest60'*maxx;


netSVM = fitrsvm(SVMP60',SVMY60);
outval = (predict(netSVM,SVMP60'));
outvalmax=outval*maxx;
SVMOutf60train=outvalmax';
outvaltest=(predict(netSVM,SVMPtest60'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget60;
SVMOutf60=outvaltestmax;
SVMmsetest60=mse(SVMOutf60,testingtarget60);
SVMmapetest60=mape(SVMOutf60,testingtarget60);
SVMmbetest60=mbe(SVMOutf60,testingtarget60);
SVMr2test60=rsquare(SVMOutf60,testingtarget60);
SVMperf60=[SVMmsetest60 SVMmapetest60 SVMmbetest60 SVMr2test60];
SVMPtestMax60=SVMPtest60'*maxx;

meantarget60=[meantarget50 mean(testingtarget60)];
meanSVM60=[meanSVM50 mean(SVMOutf60)];

figure
plot(meantarget60,height60,'k');
hold on
plot(meanSVM60,height60,'-.g');

hold off
title('average')
legend('measured','SVM est','Location','northwest')

display ('60 m Testing:     MSE     MAPE        MBE        R2')
[SVMperf60]
SVMperfall=[SVMperfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 70

nex=3;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y70 = trainingtarget';
Ytest70 = testingtarget';
testingtarget70=Ytest70'*maxx;

testingtargetmax=testingtarget*maxx;
target70=testingtarget70;

%
height70=[height60 70];
mxr=12.5+nex*0.5;
rang70=[0 mxr];
rl70=[1:mxr];
% SVM WSE
%
SVMP70 = [SVMP60; SVMOutf60train/maxx];
SVMY70 = trainingtarget';
SVMPtest70 = [SVMPtest60; SVMOutf60'/maxx];
SVMYtest70 = testingtarget';
SVMtestingtarget70=SVMYtest70'*maxx;


netSVM = fitrsvm(SVMP70',SVMY70);
outval = (predict(netSVM,SVMP70'));
outvalmax=outval*maxx;
SVMOutf70train=outvalmax';
outvaltest=(predict(netSVM,SVMPtest70'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget70;
SVMOutf70=outvaltestmax;
SVMmsetest70=mse(SVMOutf70,testingtarget70);
SVMmapetest70=mape(SVMOutf70,testingtarget70);
SVMmbetest70=mbe(SVMOutf70,testingtarget70);
SVMr2test70=rsquare(SVMOutf70,testingtarget70);
SVMperf70=[SVMmsetest70 SVMmapetest70 SVMmbetest70 SVMr2test70];
SVMPtestMax70=SVMPtest70'*maxx;

meantarget70=[meantarget60 mean(testingtarget70)];
meanSVM70=[meanSVM60 mean(SVMOutf70)];

figure
plot(meantarget70,height70,'k');
hold on
plot(meanSVM70,height70,'-.g');

hold off
title('average')
legend('measured','SVM est','Location','northwest')

display ('70 m Testing:     MSE     MAPE        MBE        R2')
[SVMperf70]
SVMperfall=[SVMperfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 80

nex=4;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y80 = trainingtarget';
Ytest80 = testingtarget';
testingtarget80=Ytest80'*maxx;

testingtargetmax=testingtarget*maxx;
target80=testingtarget80;

%
height80=[height70 80];
mxr=12.5+nex*0.5;
rang80=[0 mxr];
rl80=[1:mxr];
% SVM WSE
%
SVMP80 = [SVMP70; SVMOutf70train/maxx];
SVMY80 = trainingtarget';
SVMPtest80 = [SVMPtest70; SVMOutf70'/maxx];
SVMYtest80 = testingtarget';
SVMtestingtarget80=SVMYtest80'*maxx;

netSVM = fitrsvm(SVMP80',SVMY80);
outval = (predict(netSVM,SVMP80'));
outvalmax=outval*maxx;
SVMOutf80train=outvalmax';
outvaltest=(predict(netSVM,SVMPtest80'))';


outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget80;
SVMOutf80=outvaltestmax;
SVMmsetest80=mse(SVMOutf80,testingtarget80);
SVMmapetest80=mape(SVMOutf80,testingtarget80);
SVMmbetest80=mbe(SVMOutf80,testingtarget80);
SVMr2test80=rsquare(SVMOutf80,testingtarget80);
SVMperf80=[SVMmsetest80 SVMmapetest80 SVMmbetest80 SVMr2test80];
SVMPtestMax80=SVMPtest80'*maxx;

meantarget80=[meantarget70 mean(testingtarget80)];
meanSVM80=[meanSVM70 mean(SVMOutf80)];

figure
plot(meantarget80,height80,'k');
hold on
plot(meanSVM80,height80,'-.g');

hold off
title('average')
legend('measured','SVM est','Location','northwest')

display ('80 m Testing:     MSE     MAPE        MBE        R2')
[SVMperf80]
SVMperfall=[SVMperfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 90

nex=5;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y90 = trainingtarget';
Ytest90 = testingtarget';
testingtarget90=Ytest90'*maxx;

testingtargetmax=testingtarget*maxx;
target90=testingtarget90;

%
height90=[height80 90];
mxr=12.5+nex*0.5;
rang90=[0 mxr];
rl90=[1:mxr];
% SVM WSE
%
SVMP90 = [SVMP80; SVMOutf80train/maxx];
SVMY90 = trainingtarget';
SVMPtest90 = [SVMPtest80; SVMOutf80'/maxx];
SVMYtest90 = testingtarget';
SVMtestingtarget90=SVMYtest90'*maxx;


netSVM = fitrsvm(SVMP90',SVMY90);
outval = (predict(netSVM,SVMP90'));
outvalmax=outval*maxx;
SVMOutf90train=outvalmax';
outvaltest=(predict(netSVM,SVMPtest90'))';


outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget90;
SVMOutf90=outvaltestmax;
SVMmsetest90=mse(SVMOutf90,testingtarget90);
SVMmapetest90=mape(SVMOutf90,testingtarget90);
SVMmbetest90=mbe(SVMOutf90,testingtarget90);
SVMr2test90=rsquare(SVMOutf90,testingtarget90);
SVMperf90=[SVMmsetest90 SVMmapetest90 SVMmbetest90 SVMr2test90];
SVMPtestMax90=SVMPtest90'*maxx;

meantarget90=[meantarget80 mean(testingtarget90)];
meanSVM90=[meanSVM80 mean(SVMOutf90)];

figure
plot(meantarget90,height90,'k');
hold on
plot(meanSVM90,height90,'-.g');

hold off
title('average')
legend('measured','SVM est','Location','northwest')

display ('90 m Testing:     MSE     MAPE        MBE        R2')
[SVMperf90]
SVMperfall=[SVMperfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 100

nex=6;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y100 = trainingtarget';
Ytest100 = testingtarget';
testingtarget100=Ytest100'*maxx;

testingtargetmax=testingtarget*maxx;
target100=testingtarget100;

%
height100=[height90 100];
mxr=12.5+nex*0.5;
rang100=[0 mxr];
rl100=[1:mxr];
% SVM WSE
%
SVMP100 = [SVMP90; SVMOutf90train/maxx];
SVMY100 = trainingtarget';
SVMPtest100 = [SVMPtest90; SVMOutf90'/maxx];
SVMYtest100 = testingtarget';
SVMtestingtarget100=SVMYtest100'*maxx;

netSVM = fitrsvm(SVMP100',SVMY100);
outval = (predict(netSVM,SVMP100'));
outvalmax=outval*maxx;
SVMOutf100train=outvalmax';
outvaltest=(predict(netSVM,SVMPtest100'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget100;
SVMOutf100=outvaltestmax;
SVMmsetest100=mse(SVMOutf100,testingtarget100);
SVMmapetest100=mape(SVMOutf100,testingtarget100);
SVMmbetest100=mbe(SVMOutf100,testingtarget100);
SVMr2test100=rsquare(SVMOutf100,testingtarget100);
SVMperf100=[SVMmsetest100 SVMmapetest100 SVMmbetest100 SVMr2test100];
SVMPtestMax100=SVMPtest100'*maxx;

meantarget100=[meantarget90 mean(testingtarget100)];
meanSVM100=[meanSVM90 mean(SVMOutf100)];

figure
plot(meantarget100,height100,'k');
hold on
plot(meanSVM100,height100,'-.g');

hold off
title('average')
legend('measured','SVM est','Location','northwest')

display ('100 m Testing:     MSE     MAPE        MBE        R2')
[SVMperf100]
SVMperfall=[SVMperfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 110

nex=7;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y110 = trainingtarget';
Ytest110 = testingtarget';
testingtarget110=Ytest110'*maxx;

testingtargetmax=testingtarget*maxx;
target110=testingtarget110;

%
height110=[height100 110];
mxr=12.5+nex*0.5;
rang110=[0 mxr];
rl110=[1:mxr];
% SVM WSE
%
SVMP110 = [SVMP100; SVMOutf100train/maxx];
SVMY110 = trainingtarget';
SVMPtest110 = [SVMPtest100; SVMOutf100'/maxx];
SVMYtest110 = testingtarget';
SVMtestingtarget110=SVMYtest110'*maxx;

netSVM = fitrsvm(SVMP110',SVMY110);
outval = (predict(netSVM,SVMP110'));
outvalmax=outval*maxx;
SVMOutf110train=outvalmax';
outvaltest=(predict(netSVM,SVMPtest110'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget110;
SVMOutf110=outvaltestmax;
SVMmsetest110=mse(SVMOutf110,testingtarget110);
SVMmapetest110=mape(SVMOutf110,testingtarget110);
SVMmbetest110=mbe(SVMOutf110,testingtarget110);
SVMr2test110=rsquare(SVMOutf110,testingtarget110);
SVMperf110=[SVMmsetest110 SVMmapetest110 SVMmbetest110 SVMr2test110];
SVMPtestMax110=SVMPtest110'*maxx;

meantarget110=[meantarget100 mean(testingtarget110)];
meanSVM110=[meanSVM100 mean(SVMOutf110)];

figure
plot(meantarget110,height110,'k');
hold on
plot(meanSVM110,height110,'-.g');

hold off
title('average')
legend('measured','SVM est','Location','northwest')

display ('110 m Testing:     MSE     MAPE        MBE        R2')
[SVMperf110]
SVMperfall=[SVMperfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 120

nex=8;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y120 = trainingtarget';
Ytest120 = testingtarget';
testingtarget120=Ytest120'*maxx;

testingtargetmax=testingtarget*maxx;
target120=testingtarget120;

%
height120=[height110 120];
mxr=12.5+nex*0.5;
rang120=[0 mxr];
rl120=[1:mxr];
% SVM WSE
%
SVMP120 = [SVMP110; SVMOutf110train/maxx];
SVMY120 = trainingtarget';
SVMPtest120 = [SVMPtest110; SVMOutf110'/maxx];
SVMYtest120 = testingtarget';
SVMtestingtarget120=SVMYtest120'*maxx;

netSVM = fitrsvm(SVMP120',SVMY120);
outval = (predict(netSVM,SVMP120'));
outvalmax=outval*maxx;
SVMOutf120train=outvalmax';
outvaltest=(predict(netSVM,SVMPtest120'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget120;
SVMOutf120=outvaltestmax;
SVMmsetest120=mse(SVMOutf120,testingtarget120);
SVMmapetest120=mape(SVMOutf120,testingtarget120);
SVMmbetest120=mbe(SVMOutf120,testingtarget120);
SVMr2test120=rsquare(SVMOutf120,testingtarget120);
SVMperf120=[SVMmsetest120 SVMmapetest120 SVMmbetest120 SVMr2test120];
SVMPtestMax120=SVMPtest120'*maxx;

meantarget120=[meantarget110 mean(testingtarget120)];
meanSVM120=[meanSVM110 mean(SVMOutf120)];

figure
plot(meantarget120,height120,'k');
hold on
plot(meanSVM120,height120,'-.g');

hold off
title('average')
legend('measured','SVM est','Location','northwest')

display ('120 m Testing:     MSE     MAPE        MBE        R2')
[SVMperf120]
SVMperfall=[SVMperfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 130

nex=9;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y130 = trainingtarget';
Ytest130 = testingtarget';
testingtarget130=Ytest130'*maxx;

testingtargetmax=testingtarget*maxx;
target130=testingtarget130;

%
height130=[height120 130];
mxr=12.5+nex*0.5;
rang130=[0 mxr];
rl130=[1:mxr];
% SVM WSE
%
SVMP130 = [SVMP120; SVMOutf120train/maxx];
SVMY130 = trainingtarget';
SVMPtest130 = [SVMPtest120; SVMOutf120'/maxx];
SVMYtest130 = testingtarget';
SVMtestingtarget130=SVMYtest130'*maxx;

netSVM = fitrsvm(SVMP130',SVMY130);
outval = (predict(netSVM,SVMP130'));
outvalmax=outval*maxx;
SVMOutf130train=outvalmax';
outvaltest=(predict(netSVM,SVMPtest130'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget130;
SVMOutf130=outvaltestmax;
SVMmsetest130=mse(SVMOutf130,testingtarget130);
SVMmapetest130=mape(SVMOutf130,testingtarget130);
SVMmbetest130=mbe(SVMOutf130,testingtarget130);
SVMr2test130=rsquare(SVMOutf130,testingtarget130);
SVMperf130=[SVMmsetest130 SVMmapetest130 SVMmbetest130 SVMr2test130];
SVMPtestMax130=SVMPtest130'*maxx;

meantarget130=[meantarget120 mean(testingtarget130)];
meanSVM130=[meanSVM120 mean(SVMOutf130)];

figure
plot(meantarget130,height130,'k');
hold on
plot(meanSVM130,height130,'-.g');

hold off
title('average')
legend('measured','SVM est','Location','northwest')

display ('130 m Testing:     MSE     MAPE        MBE        R2')
[SVMperf130]
SVMperfall=[SVMperfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 140

nex=10;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y140 = trainingtarget';
Ytest140 = testingtarget';
testingtarget140=Ytest140'*maxx;

testingtargetmax=testingtarget*maxx;
target140=testingtarget140;

%
height140=[height130 140];
mxr=12.5+nex*0.5;
rang140=[0 mxr];
rl140=[1:mxr];
% SVM WSE
%
SVMP140 = [SVMP130; SVMOutf130train/maxx];
SVMY140 = trainingtarget';
SVMPtest140 = [SVMPtest130; SVMOutf130'/maxx];
SVMYtest140 = testingtarget';
SVMtestingtarget140=SVMYtest140'*maxx;

netSVM = fitrsvm(SVMP140',SVMY140);
outval = (predict(netSVM,SVMP140'));
outvalmax=outval*maxx;
SVMOutf140train=outvalmax';
outvaltest=(predict(netSVM,SVMPtest140'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget140;
SVMOutf140=outvaltestmax;
SVMmsetest140=mse(SVMOutf140,testingtarget140);
SVMmapetest140=mape(SVMOutf140,testingtarget140);
SVMmbetest140=mbe(SVMOutf140,testingtarget140);
SVMr2test140=rsquare(SVMOutf140,testingtarget140);
SVMperf140=[SVMmsetest140 SVMmapetest140 SVMmbetest140 SVMr2test140];
SVMPtestMax140=SVMPtest140'*maxx;

meantarget140=[meantarget130 mean(testingtarget140)];
meanSVM140=[meanSVM130 mean(SVMOutf140)];

figure
plot(meantarget140,height140,'k');
hold on
plot(meanSVM140,height140,'-.g');

hold off
title('average')
legend('measured','SVM est','Location','northwest')

display ('140 m Testing:     MSE     MAPE        MBE        R2')
[SVMperf140]
SVMperfall=[SVMperfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 150

nex=11;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y150 = trainingtarget';
Ytest150 = testingtarget';
testingtarget150=Ytest150'*maxx;

testingtargetmax=testingtarget*maxx;
target150=testingtarget150;

%
height150=[height140 150];
mxr=12.5+nex*0.5;
rang150=[0 mxr];
rl150=[1:mxr];
% SVM WSE
%
SVMP150 = [SVMP140; SVMOutf140train/maxx];
SVMY150 = trainingtarget';
SVMPtest150 = [SVMPtest140; SVMOutf140'/maxx];
SVMYtest150 = testingtarget';
SVMtestingtarget150=SVMYtest150'*maxx;


netSVM = fitrsvm(SVMP150',SVMY150);
outval = (predict(netSVM,SVMP150'));
outvalmax=outval*maxx;
SVMOutf150train=outvalmax';
outvaltest=(predict(netSVM,SVMPtest150'))';

outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget150;
SVMOutf150=outvaltestmax;
SVMmsetest150=mse(SVMOutf150,testingtarget150);
SVMmapetest150=mape(SVMOutf150,testingtarget150);
SVMmbetest150=mbe(SVMOutf150,testingtarget150);
SVMr2test150=rsquare(SVMOutf150,testingtarget150);
SVMperf150=[SVMmsetest150 SVMmapetest150 SVMmbetest150 SVMr2test150];
SVMPtestMax150=SVMPtest150'*maxx;

meantarget150=[meantarget140 mean(testingtarget150)];
meanSVM150=[meanSVM140 mean(SVMOutf150)];

figure
plot(meantarget150,height150,'k');
hold on
plot(meanSVM150,height150,'-.g');

hold off
title('average')
legend('measured','SVM est','Location','northwest')

display ('150 m Testing:     MSE     MAPE        MBE        R2')
[SVMperf150]
SVMperfall=[SVMperfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 160

nex=12;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y160 = trainingtarget';
Ytest160 = testingtarget';
testingtarget160=Ytest160'*maxx;

testingtargetmax=testingtarget*maxx;
target160=testingtarget160;

%
height160=[height150 160];
mxr=12.5+nex*0.5;
rang160=[0 mxr];
rl160=[1:mxr];
% SVM WSE
%
SVMP160 = [SVMP150; SVMOutf150train/maxx];
SVMY160 = trainingtarget';
SVMPtest160 = [SVMPtest150; SVMOutf150'/maxx];
SVMYtest160 = testingtarget';
SVMtestingtarget160=SVMYtest160'*maxx;


netSVM = fitrsvm(SVMP160',SVMY160);
outval = (predict(netSVM,SVMP160'));
outvalmax=outval*maxx;
SVMOutf160train=outvalmax';
outvaltest=(predict(netSVM,SVMPtest160'))';


outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget160;
SVMOutf160=outvaltestmax;
SVMmsetest160=mse(SVMOutf160,testingtarget160);
SVMmapetest160=mape(SVMOutf160,testingtarget160);
SVMmbetest160=mbe(SVMOutf160,testingtarget160);
SVMr2test160=rsquare(SVMOutf160,testingtarget160);
SVMperf160=[SVMmsetest160 SVMmapetest160 SVMmbetest160 SVMr2test160];
SVMPtestMax160=SVMPtest160'*maxx;

meantarget160=[meantarget150 mean(testingtarget160)];
meanSVM160=[meanSVM150 mean(SVMOutf160)];

figure
plot(meantarget160,height160,'k');
hold on
plot(meanSVM160,height160,'-.g');

hold off
title('average')
legend('measured','SVM est','Location','northwest')

display ('160 m Testing:     MSE     MAPE        MBE        R2')
[SVMperf160]
SVMperfall=[SVMperfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 170

nex=13;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y170 = trainingtarget';
Ytest170 = testingtarget';
testingtarget170=Ytest170'*maxx;

testingtargetmax=testingtarget*maxx;
target170=testingtarget170;

%
height170=[height160 170];
mxr=12.5+nex*0.5;
rang170=[0 mxr];
rl170=[1:mxr];
% SVM WSE
%
SVMP170 = [SVMP160; SVMOutf160train/maxx];
SVMY170 = trainingtarget';
SVMPtest170 = [SVMPtest160; SVMOutf160'/maxx];
SVMYtest170 = testingtarget';
SVMtestingtarget170=SVMYtest170'*maxx;


netSVM = fitrsvm(SVMP170',SVMY170);
outval = (predict(netSVM,SVMP170'));
outvalmax=outval*maxx;
SVMOutf170train=outvalmax';
outvaltest=(predict(netSVM,SVMPtest170'))';


outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget170;
SVMOutf170=outvaltestmax;
SVMmsetest170=mse(SVMOutf170,testingtarget170);
SVMmapetest170=mape(SVMOutf170,testingtarget170);
SVMmbetest170=mbe(SVMOutf170,testingtarget170);
SVMr2test170=rsquare(SVMOutf170,testingtarget170);
SVMperf170=[SVMmsetest170 SVMmapetest170 SVMmbetest170 SVMr2test170];
SVMPtestMax170=SVMPtest170'*maxx;

meantarget170=[meantarget160 mean(testingtarget170)];
meanSVM170=[meanSVM160 mean(SVMOutf170)];

figure
plot(meantarget170,height170,'k');
hold on
plot(meanSVM170,height170,'-.g');

hold off
title('average')
legend('measured','SVM est','Location','northwest')

display ('170 m Testing:     MSE     MAPE        MBE        R2')
[SVMperf170]
SVMperfall=[SVMperfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];

%% 180

nex=14;

traininginput=series(1:trainingnum,1:inputsize);
trainingtarget=series(1:trainingnum,inputsize+nex);
%testing

testinginput=seriesT(:,1:inputsize);
testingtarget=seriesT(:,inputsize+nex);

Y180 = trainingtarget';
Ytest180 = testingtarget';
testingtarget180=Ytest180'*maxx;

testingtargetmax=testingtarget*maxx;
target180=testingtarget180;

%
height180=[height170 180];
mxr=12.5+nex*0.5;
rang180=[0 mxr];
rl180=[1:mxr];
% SVM WSE
%
SVMP180 = [SVMP170; SVMOutf170train/maxx];
SVMY180 = trainingtarget';
SVMPtest180 = [SVMPtest170; SVMOutf170'/maxx];
SVMYtest180 = testingtarget';
SVMtestingtarget180=SVMYtest180'*maxx;

netSVM = fitrsvm(SVMP180',SVMY180);
outval = (predict(netSVM,SVMP180'));
outvalmax=outval*maxx;
SVMOutf180train=outvalmax';
outvaltest=(predict(netSVM,SVMPtest180'))';


outvaltestmax=outvaltest'*maxx;
testingtargetmax=testingtarget180;
SVMOutf180=outvaltestmax;
SVMmsetest180=mse(SVMOutf180,testingtarget180);
SVMmapetest180=mape(SVMOutf180,testingtarget180);
SVMmbetest180=mbe(SVMOutf180,testingtarget180);
SVMr2test180=rsquare(SVMOutf180,testingtarget180);
SVMperf180=[SVMmsetest180 SVMmapetest180 SVMmbetest180 SVMr2test180];
SVMPtestMax180=SVMPtest180'*maxx;

meantarget180=[meantarget170 mean(testingtarget180)];
meanSVM180=[meanSVM170 mean(SVMOutf180)];

figure
plot(meantarget180,height180,'k');
hold on
plot(meanSVM180,height180,'-.g');

hold off
title('average')
legend('measured','SVM est','Location','northwest')

display ('180 m Testing:     MSE     MAPE        MBE        R2')
[SVMperf180]
SVMperfall=[SVMperfall; mse(outvaltestmax,testingtarget*maxx) mape(outvaltestmax,testingtarget*maxx) mbe(outvaltestmax,testingtarget*maxx) rsquare(outvaltestmax,testingtarget*maxx)];
