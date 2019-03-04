function allnets=Farcast(nn,mode)
%% Settings
includediff=1;
includecyc=0;
excludeX=1;
includeMA=0;
transX=1;
numhistpt=10;
numofstocks=30;
numofcryptos=50;
numofnets=10;
D1 = LagOp({1,-1},'Lags',[0,1]);
period=20;
farvmod=1;
newrangemin=0;
newrangemax=1;
%gad=0.955:0.005:1;
maxdyc=0;
if maxdyc==1; dyclbl=1; else dyclbl=0; end
wts = [1/period*2;repmat(1/period,period-1,1);1/period*2];
D20=LagOp({1,-1},'Lags',[0,period]);
Dseasonal=D1;
maxMA=0;
maxMAs=[3];
tmpf=[2000];
minlag=3;lags=[7];
%% Load data
options=weboptions('timeout',10);
S=webread('https://min-api.cryptocompare.com/data/histoday?fsym=XLM&tsym=USD&limit=550',options);
S.close=[S.Data.close];
dates=[S.Data.time];
companies{1}={'Financials','Technology','Real%20Estate','Energy','Communication%20Services'};
companies{2}={'Computer%20Hardware'};
urls={};names={};
for i=1:size(companies,2)
    switch i
        case 1
            for sector_i=1:size(companies{1},2)
                urls{1,end+1}=['https://api.iextrading.com/1.0/stock/market/collection/sector?collectionName=' companies{1}{sector_i}];
                names{1,end+1}=companies{1}{sector_i};
            end
        case 2
            for tag_i=1:size(companies{2},2)
                urls{1,end+1}=['https://api.iextrading.com/1.0/stock/market/collection/tag?collectionName=' companies{2}{tag_i}];
                names{1,end+1}=companies{2}{tag_i};
            end
    end
end
parfor url_i=1:size(urls,2)
    stocksyms{url_i}=webread(urls{url_i},options);
end
syms={};
parfor getsym_i=1:size(urls,2)
    [~,idx]=sort([stocksyms{getsym_i}.avgTotalVolume],'descend');
    sm={stocksyms{getsym_i}.symbol};
    syms=horzcat(syms,{sm{idx(1:numofstocks)}});
end
parfor url_i=1:size(syms,2)
    urls2{url_i}=['https://api.iextrading.com/1.0/stock/' syms{url_i} '/chart/2y'];
    stock{url_i}=webread(urls2{url_i},options);
end
top100url='https://min-api.cryptocompare.com/data/top/mktcapfull?limit=100&tsym=USD';
top100=webread(top100url);
parfor i=1:numofcryptos
    crysyms{1,i}=top100.Data(i).CoinInfo.Name;
end
%crysyms={'ETH','LTC','XLM','NEM','XRP','EOS','NEO','ADA','QTUM','BCH','DASH','ZEC','BNB'};
urls3=cell(1,size(crysyms,2)+1);
parfor url_i3=1:size(crysyms,2)
    urls3{url_i3}=['https://min-api.cryptocompare.com/data/histoday?fsym=' crysyms{url_i3} '&tsym=USD&limit=550']
    crypto{url_i3}=webread(urls3{url_i3},options);
end
% urls3{size(crysyms,2)+1}=['https://min-api.cryptocompare.com/data/histoday?fsym=BTC&tsym=USD&limit=550'];
% crypto{size(crysyms,2)+1}=webread(urls3{size(crysyms,2)+1});
%% get relevant data
clear closes stdates cp cryp
for rel_i=1:size(stock,2)
    if isa(stock{rel_i},'struct')
        closes{:,rel_i}=[stock{rel_i}.close]';
        cp{:,rel_i}=[stock{rel_i}.volume]';
        stdates{:,rel_i}={stock{rel_i}.date}';
    end
end
for rel_i2=1:size(crypto,2)
    if isa(crypto{rel_i2},'struct') && ~isempty(fieldnames(crypto{rel_i2}.Data)) && isfield((crypto{rel_i2}.Data),'time')
        cryp{:,rel_i2}=[crypto{rel_i2}.Data.close]';
        crypvol{:,rel_i2}=[crypto{rel_i2}.Data.volumeto]'-[crypto{rel_i2}.Data.volumefrom]';
    end
end
crypempti=find(~cellfun(@isempty,cryp));
cryp={cryp{crypempti}};
crypvolempti=find(~cellfun(@isempty,crypvol));
crypvol={crypvol{crypvolempti}};
crypto=horzcat({crypto{crypempti}},{crypto{crypvolempti}});
cryp=horzcat(cryp,crypvol);
crysyms={crysyms{crypempti}};
nwi=find(~cellfun(@isempty,closes));
closes={closes{1,[nwi]}};
cp={cp{1,[nwi]}};
syms={syms{1,[nwi]}};
stdates={stdates{1,[nwi]}};
src={closes,cp};
varsz=size(closes,2)+size(cp,2);
for src_i=1:size(src,2)-1
    stdates=horzcat(stdates,stdates);
end
stockclose=zeros(size(dates,2),varsz);
tgtdatesser=dates;

parfor dates_i=1:varsz
    datesser{dates_i}=cellfun(@(x)posixtime((datetime(x))),stdates{1,dates_i});
end
parfor dates_i2=1:size(cryp,2)
    crypdatesser{dates_i2}=[crypto{1,dates_i2}.Data.time];
end
clear stockclose Xsrc
nobs=size(dates,2);
clear dates2
stockclose=[];
for src_i=1:size(src,2)-1
    for api_i2=1:size(src{src_i},2)
        for sym_i=1:size(dates,2)
            
            day=tgtdatesser(sym_i);
            
            if isa(stock{api_i2},'struct')
                if ~isempty(closes{1,api_i2}(find(datesser{1,api_i2}==day),:))
                    Xsrc{src_i}(sym_i,api_i2)=[src{src_i}{api_i2}(find(datesser{1,api_i2}==day),:)];
                    dates2(sym_i,api_i2)=[dates(find(datesser{1,api_i2}==day))];
                end
            end
        end
    end
    stockclose=horzcat(stockclose,Xsrc{src_i});
end

cryptoclose=[];
for cryp_i=1:size(cryp,2)
    for sym_i=1:size(dates2,1)
        day=dates2(sym_i,1);
        if ~isempty(cryp{cryp_i}(find(crypdatesser{cryp_i}==day),:))
            cryptoclose(sym_i,cryp_i)=[cryp{cryp_i}(find(crypdatesser{1,cryp_i}==day),:)];
        end
    end
end
stockclose=[stockclose cryptoclose];
stockclose=stockclose(:,sum(stockclose,1)~=0);
parfor dat_i=1:size(dates2,1)
    day=dates(1,dat_i);
    chosenclose(dat_i,1)=S.close(1,find([S.Data.time]==day));
end
stockclose(isnan(stockclose))=0;
vi=1:nobs;

%% treat missing values
parfor seq_i=1:size(stockclose,2)
    nz=stockclose(:,seq_i)~=0;
    b=num2str(nz');
    b=strrep(b,' ','');
    zb=strfind(b,'0');
    ze=strfind(b,'01');
    stz_i{seq_i}=(zb(1,1):ze(1,1));
    stz_l(seq_i)=length(zb(1,1):ze(1,1));
end
valid_i=find(stz_l<nobs/5);
stz_l=stz_l(valid_i);
stockclose=stockclose(:,valid_i);
lon_stz_l=max(stz_l);
stockclose=stockclose(lon_stz_l+1:end,:);
%get longest sequence of valid data
%remove seasonality from the sequences
seqall_i=cell(1,size(stockclose,2));
nobs2=size(stockclose,1);
unsesall=zeros(nobs2,1);
for seq_i=1:size(stockclose,2)
    nz=stockclose(:,seq_i)~=0;
    nz=[ 0 nz' 0 ]; %remember to subtracr 1 from indices
    b=num2str(nz);
    b=strrep(b,' ','');
    ib=strfind(b,'01');
    ie=strfind(b,'10');
    
    for getseq=1:size(ie,2)
        seqall_i{seq_i,getseq}=ib(getseq):ie(getseq)-1;
        refseq{seq_i,getseq}=stockclose(ib(getseq):ie(getseq)-1,seq_i);
        if length(refseq{seq_i,getseq})>2
            [seqtrend{seq_i,getseq},seqses{seq_i,getseq}]=hpfilter(refseq{seq_i,getseq},14400);
        else
            seqtrend{seq_i,getseq}=1;
            seqses{seq_i,getseq}=0;
        end
        unsesrefseq{seq_i,getseq}=refseq{seq_i,getseq}-seqses{seq_i,getseq};
        unsesall(seqall_i{seq_i,getseq},seq_i)=unsesrefseq{seq_i,getseq};
        sesall(seqall_i{seq_i,getseq},seq_i)=seqses{seq_i,getseq};
    end
    for unsesseq=1:size(ie,2)-1
        bridgeseq{seq_i,unsesseq}=seqall_i{seq_i,unsesseq}(1,1):seqall_i{seq_i,unsesseq+1}(end);
        misseq{seq_i,unsesseq}=seqall_i{seq_i,unsesseq}(end)+1:seqall_i{seq_i,unsesseq+1}(1,1)-1;
        mislocbridge=find(unsesall(bridgeseq{seq_i,unsesseq},seq_i)==0);
        unmislocbridge=find(unsesall(bridgeseq{seq_i,unsesseq},seq_i)~=0);
        unsesall(misseq{seq_i,unsesseq},seq_i)=mean(unsesall(bridgeseq{seq_i,unsesseq}(unmislocbridge),seq_i));
        unsesall(misseq{seq_i,unsesseq},seq_i)=spline(1:size(bridgeseq{seq_i,unsesseq},2),unsesall(bridgeseq{seq_i,unsesseq},seq_i),mislocbridge);
        sesall=interpft(sesall,nobs2);
    end
    mis_i{seq_i}=vi(~ismember(1:nobs2,seqall_i{seq_i}));
end
clear X
parfor repack_i=1:size(stockclose,2)
    X(:,repack_i)=unsesall(:,repack_i)+sesall(:,repack_i);
end

for crct_tail=1:size(stockclose,2)
    bseq=[X(1:5,crct_tail)];
    bseq(bseq==0)=mean(bseq(bseq~=0));
    X(1:5,crct_tail)=bseq;
    eseq=[X(end-5:end,crct_tail)];
    eseq(eseq==0)=mean(eseq(eseq~=0));
    X(end-5:end,crct_tail)=eseq;
    locs=find(X(:,crct_tail)==0);
    X(locs,crct_tail)=pchip(1:nobs2,X(:,crct_tail),locs);
end


chosenclose2=chosenclose(lon_stz_l+1:end,1);
%exclude='GPRO';
%X=X(:,find(cellfun(@isempty,strfind(syms,exclude))));
y=chosenclose2;
X=X(find(sum(X,2)~=0),:);
nobs2=size(X,1);
%
% for i5=1:size(X,2)
%     X(:,i5)=zscore([X(:,i5)])';
% end

%% Preprocessing
%plot seasonal trend and get the number of fluctuations
%get log of the data.
%estimate and calculate best difference from the log.
%estimate and calculate best lag from the differenced term.
%add best difference, best MA and best Lag to the predictor variables.
%add hpfilter to predictor variables
%Do that for all the predictor variables which are time series themselves.
for nets_i=1:numofnets
mulagi={};medlagi={};
clear mulagi medlagi ppmu ppmed allp muallmu medallmu
for maxma_i=1:size(maxMAs,2)
    for lag_i=1:size(lags,2)
        maxlag=lags(lag_i);
        gad=ones(1,maxlag);
        gadrng=(gad(end)-gad(1));
        for tmp_i=1:size(tmpf,2)
            terms={};
            %% Transformations to external data and the log of the response
            clear diffuntrend
         
                sestrend= X;
                % figure;
                if transX
                    % seasonal plus nonseasonal trends
                    for isym=1:size(sestrend,2)
                        
                        diffuntrend(:,isym)=diff_Farcast(sestrend(:,isym),Dseasonal);
                    end
                else
                    diffuntrend=diff_Farcast(sestrend(:,end),Dseasonal);
                end
                %         diffuntrend{sym_i}=diff2sd_Farcast(sestrend);
                
                if includediff
                    terms{1,end+1}=diffuntrend;
                    %  subplot(4,1,1);plot(diffuntrend{sym_i})
                end
                % %seasonality adapted moving average
                
                if transX
                    for isym=1:size(sestrend,2)
                        sesadapted_MA(:,isym) = conv(sestrend(:,isym),wts,'valid');
                    end
                else
                    sesadapted_MA = conv(sestrend(:,end),wts,'valid');
                end
                if includeMA
                    terms{1,end+1}=sesadapted_MA;
                end
                mai=size(terms,2);

            %% Exclude non varying predictors
            clear fncyc fntrend trendvrr cycvrr
            if transX
                for i=1:size(sestrend,2)
                    [trendvrr(:,i),cycvrr(:,i)]=hpfilter(sestrend(:,i),tmpf(tmp_i));
                end
                %figure;plot(cycvrr);
                clear fnvr
                for i=1:size(sestrend,2)
                    fnvr(i)=var(cycvrr(:,i));
                end
                varthresh=0;
                %figure;plot(fnvr);
                dyc_i=fnvr>varthresh;
                dyc_X=sestrend(:,dyc_i);
                dyc_cycvrr=cycvrr(:,dyc_i);
                %dyc_cycvrr=[cycvrr tgtcycvrr];
                %     figure;plot(fnvr(fnvr>2))
                %     figure;plot(fntrend);
                % else
                % [trendvrr,cycvrr]=hpfilter(sestrend(:,end),tmpf(tmp_i));
                % dyc_i=size(sestrend,2);
                % dyc_X=sestrend(:,dyc_i);
                % dyc_cycvrr=cycvrr(:,dyc_i);
            end
            dyc_diff=[];
            if includediff
                if transX
                    dyc_diff=terms{1,1}(:,dyc_i);
                else
                    dyc_diff=terms{1,1};
                end
            end
            if includeMA
                if transX
                    dyc_MA=terms{1,mai}(:,dyc_i);
                else
                    dyc_MA=terms{1,mai};
                end
                dyc_terms={dyc_diff,dyc_MA};
            else
                dyc_terms={dyc_diff};
            end
            if includecyc;
                terms{1,end+1}=cycvrr;
                dyc_terms{1,end+1}=dyc_cycvrr;
            end
            if maxdyc==1; terms=dyc_terms;end
            if maxdyc==1;
                terms{1,end+1}=dyc_X;
            else
                terms{1,end+1}=X;
            end
            terms={terms{1,find(~cellfun(@isempty,terms))}};
            %% Create predictors mat
            for size_i=1:size(terms,2)
                sizes(size_i)=size(terms{size_i},1);
            end
            minisize=min(sizes);
            tgt=y(nobs2-minisize+1:end);
            predictors=[];
            for unpack_i=1:size(terms,2)
                delay=size(terms{1,unpack_i},1)-minisize;
                predictors=horzcat(predictors, terms{1,unpack_i}(delay+1:end,:));
            end
            
            warning off
            
            
            allp{lag_i}{tmp_i}=zeros(maxlag);
            for numpredpts=minlag:maxlag
                switch mode
                    case 'test'
                        retractpred_i=numpredpts;
                        testlags=maxlag;
                    case 'forecast'
                        retractpred_i=0;
                        testlags=0;
                end
                
                if nn
                    modtgt=(((newrangemax-newrangemin)./(max(tgt)-min(tgt))).*(tgt-(max(tgt))))+newrangemax;
                    curtgt=modtgt(1:end-retractpred_i);
                      clear movavg
                    movavg=[];
                    if maxMA
                        curMA=maxMAs(maxma_i);
                        for i=1:curMA
                            movavg(:,end+1) = movmean(curtgt,i);
                        end
                    end
                    %% remove stationary predictors
                    predmin = min(predictors,[],1);
                    predmax = max(predictors,[],1);
                    idxConstant = predmin.*1.3 < predmax;
                    predictors=predictors(:,idxConstant);

                   
                    %% Prepare network
                    if excludeX
                        predictors=[];
                    end
                    normtgt=tgt/max(tgt);
                    neurpredall=predictors';
                    neurtgtall=tgt';
                    neurXall=[neurtgtall; neurpredall];
                    %% Standardize predictors
                    predmu = mean(neurXall(:),1);
                    predsig = std(neurXall(:),0,1);
                    for norm_i = 1:size(neurXall,1)
                       neurXallnew(norm_i,:)=(((newrangemax-newrangemin)./(max(neurXall(norm_i,:))-min(neurXall(norm_i,:)))).*(neurXall(norm_i,:)-(max(neurXall(norm_i,:)))))+newrangemax;
                    end
%                     for norm_i = 1:numel(neurXall)
%                         neurXall(norm_i) = (neurXall(norm_i) - predmu) ./ predsig;
%                     end
                    neurXtrain=neurXallnew(:,1:end-retractpred_i);
                    neurYtrain=neurXallnew(:,retractpred_i+1:end);
                    
                    nfeatures = size(neurXtrain,1);
                    numHiddenUnits = 200;
                    layers = [ ...
                        sequenceInputLayer(nfeatures)
                        bilstmLayer(numHiddenUnits,'Output','sequence')
                        fullyConnectedLayer(nfeatures)
                        regressionLayer];
                    
                    maxEpochs = 600;
                    options = trainingOptions('adam', ...
                        'MaxEpochs',maxEpochs, ...
                        'GradientThreshold',1, ...
                        'shuffle','never',...
                        'InitialLearnRate',0.001, ...
                        'LearnRateSchedule','piecewise', ...
                        'LearnRateDropPeriod',125, ...
                        'ExecutionEnvironment','gpu',...
                        'LearnRateDropFactor',0.2, ...
                        'Verbose',0, ...
                        'Plots','none');
                    %% Train network
                    net = trainNetwork(neurXtrain,neurYtrain,layers,options);
                    %% test regression
                    switch mode
                        case 'test'
                            net = predictAndUpdateState(net,neurXtrain);
                            [net,fcast] = predictAndUpdateState(net,neurYtrain(:,end));
                            for i = 2:numpredpts
                                [net,fcast(:,i)] = predictAndUpdateState(net,fcast(:,i-1),'ExecutionEnvironment','gpu');
                            end
                            farcast=fcast(1,:)';
                        case 'forecast'
                            retractpred_i=0;
                            testlags=0;
                            farcast=fcasty(1,1:numpredpts)';
                    end
                    %% Forecast
                    figure;
                    plot([modtgt(end-retractpred_i-(numhistpt-1):end-retractpred_i);farcast(:,1)])
                    hold on;
                    plot(modtgt(end-retractpred_i-(numhistpt-1):end),'r','LineWidth',1)
                    hold on;
                else
                    farcast=varf([curtgt/div movavg predictors(1:end-numpredpts,:)],maxlag,numpredpts,size(tgt,1)-numpredpts+1,[],farvmod);
                    figure;
                    plot([tgt(end-maxlag-numhistpt-1:end-numpredpts);(farcast(:,1).*div)])
                    hold on;
                    plot(tgt(end-maxlag-2:end),'r','LineWidth',1)
                    hold on;
                    tl=['init hp: ' num2str(tmpf(tmp_i)) ' & farvmod: ' num2str(farvmod) ' & lags: ' num2str(minlag) ':' num2str(maxlag)...
                        ' & timedecay: ' num2str(gadrng) ' & Low variation included: ' num2str(dyclbl)];
                    title(tl);
                end
                %  ppmu=cell(1,maxlag); ppmed=cell(1,maxlag);
                allp{lag_i}{tmp_i}(maxlag-numpredpts+1:maxlag,numpredpts)=(farcast(:,1));
                for i3=1:size(allp{lag_i}{tmp_i},2)
                    allp{lag_i}{tmp_i}(i3,:)=(allp{lag_i}{tmp_i}(i3,:)).*gad;
                    in_i=[allp{lag_i}{tmp_i}(i3,:)]~=0;
                    ppmu{lag_i}(tmp_i,i3)=mean([allp{lag_i}{tmp_i}(i3,in_i)]);
                    ppmed{lag_i}(tmp_i,i3)=median(allp{lag_i}{tmp_i}(i3,in_i));
                end
            end
        end
        mulagima{lag_i,:}=mean(ppmu{lag_i},1);
        medlagima{lag_i,:}=median(ppmed{lag_i},1);
        
        mulagi{maxma_i,lag_i}=mulagima{lag_i,:};
        medlagi{maxma_i,lag_i}=medlagima{lag_i,:};
    end
end

for i2=1:size(mulagi,2)
    for i=1:size(mulagi,1)
        clear tmp tmp2
        tmp(i,:)=mulagi{i,i2};
        mulagiavg2{i2}=mean(tmp,1);
        tmp2(i,:)=medlagi{i,i2};
        medlagiavg2{i2}=mean(tmp2,1);
    end
end
muall=zeros(max(lags));
medall=zeros(max(lags));
for mn=1:size(mulagiavg2,2)
    muall(mn,max(lags)-lags(mn)+1:end)= (mulagiavg2{1,mn})';
    medall(mn,max(lags)-lags(mn)+1:end)= (medlagiavg2{1,mn})';
    %     muall(mn,max(lags)-lags(mn)+1:end)=mulagiavg2{mn};
    %      medall(mn,max(lags)-lags(mn)+1:end)=medlagiavg2{mn};
end
for mumu_i=1:max(lags)
    muallmu(mumu_i)=mean(muall(muall(:,mumu_i)~=0,mumu_i),1);
    medallmu(mumu_i)=mean(medall(medall(:,mumu_i)~=0,mumu_i),1);
end
invalid=max(find(muallmu==0 & medallmu==0));
muallmuvalid=muallmu(invalid+1:end);
medallmuvalid=medallmu(invalid+1:end);

if ~isempty(invalid)
lastknownpt=neurXallnew(1,end-testlags:end-testlags+invalid);
else
 lastknownpt=neurXallnew(1,end-testlags);
 muallmuvalid=muallmu;
medallmuvalid=medallmu;
end
figure;
subplot(2,1,1)
plot(modtgt(end-(numhistpt-1)-retractpred_i:end),'r','LineWidth',1)
hold on;
plot(numhistpt:numhistpt+retractpred_i,[lastknownpt muallmuvalid],'g');hold on; plot(numhistpt:numhistpt+retractpred_i,[lastknownpt medallmuvalid],'k');
title('Actual and Predicted price')
lseofmu=(abs(modtgt(end-retractpred_i+1:end)'- muallmuvalid)./modtgt(end-retractpred_i+1:end)').*100;
lseofmed=(abs(modtgt(end-retractpred_i+1:end)'- medallmuvalid)./modtgt(end-retractpred_i+1:end)').*100;
MSEmed=mean(lseofmed);
MSEmu=mean(lseofmu);
subplot(2,1,2);plot(lseofmu,'g');hold on;plot(lseofmed,'k');
title('Absolute difference in percentage at each point')
allnets(nets_i).net=net;
allnets(nets_i).mu=muallmuvalid;
allnets(nets_i).med=medallmuvalid;
allnets(nets_i).muabserr=lseofmu;
allnets(nets_i).medabserr=lseofmed;
allnets(nets_i).medMSE=MSEmed;
allnets(nets_i).muMSE=MSEmu;
clear MSEmu MSEmed muallmuvalid medallmuvalid lseofmu lseofmed net
end
for i=1:size(allnets,2)
allnetsmuall(i,:)=allnets(i).mu;
allnetsmedall(i,:)=allnets(i).med;
end
allnetsmu=mean(allnetsmuall,1);
allnetsmed=mean(allnetsmedall,1);
figure;
subplot(2,1,1)
plot(modtgt(end-(numhistpt-1)-retractpred_i:end),'r','LineWidth',1)
hold on;
plot(numhistpt:numhistpt+retractpred_i,[lastknownpt allnetsmu],'g');hold on; plot(numhistpt:numhistpt+retractpred_i,[lastknownpt allnetsmed],'k');
title(['Actual and Predicted price over ' num2str(numofnets) ' networks'])
lseofallnetsmu=(abs(modtgt(end-retractpred_i+1:end)'- allnetsmu)./modtgt(end-retractpred_i+1:end)').*100;
lseofallnetsmed=(abs(modtgt(end-retractpred_i+1:end)'- allnetsmed)./modtgt(end-retractpred_i+1:end)').*100;
MSEallnetsmed=mean(lseofallnetsmed);
MSEallnetsmu=mean(lseofallnetsmu);
subplot(2,1,2);plot(lseofallnetsmu,'g');hold on;plot(lseofallnetsmed,'k');
title('Absolute difference in percentage at each point')
allnets(1).allnetsmu=allnetsmu;
allnets(1).allnetsmed=allnetsmed;
allnets(1).MSEallnetsmu=MSEallnetsmu;
allnets(1).MSEallnetsmed=MSEallnetsmed;
allnets(1).LSEallnetsmed=lseofallnetsmed;
allnets(1).LSEallnetsmu=lseofallnetsmu;
end
function [m_diff]=diff_Farcast(M,ft)
msize=size(M,2);
for x_i=1:msize
    m_diff(:,x_i) = filter(ft,M(:,x_i));
end
end
function [Xdiff,validdiff]=diff2sd_Farcast(X)
D1 = LagOp({1,-1},'Lags',[0,1]);
xsize=size(X,2);
validdiff=zeros(1,xsize);n=0;
while ~isempty(find(validdiff==0))
    n=n+1;
    Xdiff(:,validdiff==0)=diff_Farcast(X(:,validdiff==0),D1);
    validdiff=abs(mean(X(:,validdiff==0),1)-mean(Xdiff(:,validdiff==0),1))> abs(2*std(X(:,validdiff==0)));
end
end