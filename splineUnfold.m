
%% berstein testing area snipet from unfold spline basis creator
cfg.timeexpandparam = 400
cfg.windowlength = 300

if cfg.timeexpandparam > cfg.windowlength
    warning('Spline: Your timeexpandparam is larger than the maximum (%i, max %i). Lower it (or in crease epoch length) to get rid of this warning',cfg.timeexpandparam,cfg.windowlength)
    cfg.timeexpandparam = cfg.windowlength;
end
if cfg.timeexpandparam < 3
    error('You need at least three splines (timeexpandparam & windowlength have to be >2)')
end
knots = linspace(1,cfg.windowlength,cfg.timeexpandparam-2);
knots = [repmat(knots(1),1,3) knots repmat(knots(end),1,3)];
basis = Bernstein(1:cfg.windowlength,knots,[],4)'; % 4 is the order
basis(end,end) = 1; % there s

% knots = linspace(1,300,4-2);
% 
% basis = Bernstein(1:300,knots,[],4)';
%
n = round(nEvents/2)+1;
values1 = normrnd(2, 1, [n, 1]);
values2 = normrnd(5, 14, [5*n, 1]);
sac = [values1;values2];
sac((sac<0.1) | (sac>25)) = [];
hist(sac,40)


%%
amp1 = 7;
amp2 = 6.8;
amp3 = 2;
sig1 = 25;
sig2 = 25;
sig3 = 70;
mu1= 100;
mu2 =145;
mu3 = 200;

response =  @(x,mu,sig1,amp1,sig2,amp2,sig3,amp3) amp1.*exp(-(((x-mu-100).^2)/(2.*sig1.^2)))...
                                                 -amp2.*exp(-(((x-mu-145).^2)/(2*sig2.^2)))...
                                                 +amp3.*exp(-(((x-mu-200).^2)/(2.*sig3.^2)));

%load any subject to use an EEG struct as template

tmax = 700%sec
EEG.xmax = tmax
nsamples = round(EEG.srate*tmax);
EEG.data = zeros(128,nsamples);
EEG.times =linspace(0,tmax*1000,nsamples)
EEG.pnts = nsamples ;


%plot(sac_amp,sac_resp)
%create responses and events
events = {'saccade','fixation','saccade','fixation','saccade','fixation','saccade','fixation','saccade'};
onsets = {500      ,1500      ,2000    ,  2400     , 2700    , 3200      ,   3600  ,  4000   ,4500 }
saccades_count = sum(strcmp(events, 'saccade'));
sac_amp  = linspace(1,25,saccades_count);%define the range of saccade amplitudes
sac_resp = nthroot(sac_amp,6);%.009*sac_amp.^2;
ev=1
sac=1
EEG.event = [];
event =[];
words = {'fixation', 'saccade'};

nextOnset  = 100;
nEvents  = 1500;
contador = 0;
n = round(nEvents/12)+2;
values1 = normrnd(2, 1, [n, 1]);
values2 = normrnd(5, 14, [5*n, 1]);
sac = [values1;values2];
sac((sac<0.1) | (sac>25)) = [];
for res = 1:nEvents
    %random latency
    randomValue = randi([400, 600]);
    %random event type
    randomIndex = randi([1, numel(words)]);
    randomEvent = words{randomIndex};
    %random sac amp
    randomSacamp= randi([2, 25]);

    if (nextOnset+300)> nsamples
        break
    end

    if mod(contador,2)
        EEG = generate_square_resp(nextOnset,EEG,200, sac(res));
        event.type = 'saccade';
        event.latency = nextOnset;%onsets{res};
        event.sac_amplitude = sac(res); %sac_amp(sac);
        %sac = sac +1;
    else 
        %EEG.data(1,:) =  EEG.data(1,:) + response(EEG.times,onsets{res},sig1,amp1,sig2,amp2,sig3,amp3);
        EEG = generate_gaussian_resp(nextOnset,EEG,300);
        event.type = 'fixation';
        event.latency = nextOnset;%onsets{res};
        event.sac_amplitude = [];
    end
    contador = contador +1;
    nextOnset = nextOnset + randomValue;
    EEG.event = [ EEG.event ; event];
end

t = [2,2,2,2,16.5,25,25,25,25];
plot(EEG.data(1,:))
%%
model     = {'y ~ 1','y ~ 1+spl(sac_amplitude,5)'}; 

cfgDesign = [];
cfgDesign.eventtypes = {'fixation','saccade'}; % we model the fixation onset
cfgDesign.formula = model ; %saccade_amp + cat(hard_easy)'}%+stimulusDur+cat(hard_easy)'}
EEG = uf_designmat(EEG,cfgDesign);



%TIME EXPAND
cfgTimeexpand = [];
cfgTimeexpand.timelimits = [0,.6];
EEG = uf_timeexpandDesignmat(EEG,cfgTimeexpand);
%keyboard


%FITTING MODEL 
EEG= uf_glmfit(EEG);

%%
figure;
n_plots = size(EEG.unfold.beta_dc,3)

% Divide the window into two subplots, arranged side by side
for p = 1:numel(EEG.unfold.colnames)

subplot(1,n_plots,p);
plot(EEG.unfold.times, EEG.unfold.beta_dc(1,:,p));
title(EEG.unfold.colnames{p});
end

% subplot(1,2,2);
% plot(EEG.unfold.times, EEG.unfold.beta_dc(1,:,2));
% title('Plot 2');

%
zero_sample = find(EEG.unfold.times == 0)

%%
ufresult= uf_condense(EEG);

sac_amp_ix = cellfun(@(x)~isempty(x),(strfind({ufresult.param.name},'sac_amplitude'))); % get all betas related to sac_amplitude
timeix = get_min(0.115,ufresult.times); % somewhere around the p100
splinevalue = [ufresult.param(sac_amp_ix).value];
figure;
plot(splinevalue,squeeze(ufresult.beta(1,timeix,sac_amp_ix)),'-o'),xlabel('saccade amplitude')
%%
ufresult= uf_condense(EEG);


ufresult= uf_predictContinuous(ufresult,'predictAt',{{'sac_amplitude',linspace(0.3,25,100)}});
sac_amp_ix = cellfun(@(x)~isempty(x),(strfind({ufresult.param.name},'sac_amplitude'))); % get all betas related to sac_amplitude
y = ufresult.beta(1,timeix,sac_amp_ix) + 1.5331; 
% we could add the intercept; it is a constant offset
%y = y + ufresult.beta(1,timeix,1);
plot([ufresult.param(sac_amp_ix).value],squeeze(y)),xlabel('saccade amplitude'),title(sprintf('Non-linear relationship between ERP and saccadic amplitude @ %.2f s',ufresult.times(timeix)))
%%
spl = EEG.unfold.splines{1};
ufresult= uf_condense(EEG);
sacX =linspace(0.3,25,500); % at which points to evaluate?
splineSet = spl.splinefunction(sacX,spl.knots);
% remove the spline that was moved into the intercept
splineSet(:,ufresult.unfold.splines{1}.removedSplineIdx) = [];
beta = squeeze(ufresult.beta(1,timeix,2:end-1)); % 2: to remove the intercept, end-1 because of the stimulus event predictor
splineEvaluated = bsxfun(@times,splineSet,beta'); % weight the basis functions by the betas, but don't add them
plot(sacX,splineEvaluated); hold all;
plot(sacX,sum(splineEvaluated,2),'k','LineWidth',1.5), % add the basis functions at each saccade-amplitude to get the modelfit
xlabel('saccade amplitude'),title(sprintf('Non-linear relationship between ERP and saccadic amplitude @ %.2f s',ufresult.times(timeix)))
%%
ufpredict = uf_predictContinuous(ufresult); % evaluate the splines at the quantiles
uf_plotParam(ufpredict,'channel',1,'add_average',1);
%%
subplot(2,1,1),uf_plotDesignmat(EEG,'sort',0,'figure',0)
subplot(2,1,2),uf_plotDesignmat(EEG,'sort',1,'figure',0)
%%
spl = EEG.unfold.splines{1};
subplot(2,1,1),histogram(spl.paramValues,100),title('Histogram of saccadic amplitudes')
splineSet = spl.splinefunction(linspace(0,25,500),spl.knots);
subplot(2,1,2),plot(linspace(0,25,500),splineSet),xlabel('saccadic amplitude')

%% Functions
function EEG = generate_square_resp(sample_onset,EEG,sample_dur,sacamp)
    times_onset = 1000*sample_onset/EEG.srate;
    times_dur = 1000*sample_dur/EEG.srate;
    
    EEG.data(1,:) =  EEG.data(1,:) + (heaviside(EEG.times -times_onset)-heaviside(EEG.times -(times_onset+times_dur)))*nthroot(sacamp,6);
end
function EEG = generate_gaussian_resp(sample_onset,EEG,sample_dur)
    times_onset = 1000*sample_onset/EEG.srate;
    times_dur = 1000*sample_dur/EEG.srate;
    x= EEG.times;
    mu = times_onset;
    amp1 = 7;
    amp2 = 6.8;
    amp3 = 2;
    sig1 = 25;
    sig2 = 25;
    sig3 = 70;
    mu1= 100;
    mu2 =145;
    mu3 = 200;
    EEG.data(1,:) =  EEG.data(1,:) + amp1.*exp(-(((x-mu-100).^2)/(2.*sig1.^2)))...
                                                 -amp2.*exp(-(((x-mu-145).^2)/(2*sig2.^2)))...
                                                 +amp3.*exp(-(((x-mu-200).^2)/(2.*sig3.^2)));

end
