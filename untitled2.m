
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

tmax = 10%sec
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
for res = 1:length(events)
    if strcmp(events{res},'saccade')
        EEG = generate_square_resp(onsets{res},EEG,200,sac_amp(sac))
        event.type = 'saccade';
        event.latency = onsets{res};
        event.sac_amplitude = sac_amp(sac);
        sac = sac +1;
    elseif strcmp(events{res},'fixation')
        %EEG.data(1,:) =  EEG.data(1,:) + response(EEG.times,onsets{res},sig1,amp1,sig2,amp2,sig3,amp3);
        EEG = generate_gaussian_resp(onsets{res},EEG,300)
        event.type = 'fixation';
        event.latency = onsets{res};
    end
    EEG.event = [ EEG.event ; event]
end


plot(EEG.data(1,:))
%%
model     = {'y ~ 1','y ~ 1+spl(sac_amplitude,5)'}; 

cfgDesign = [];
cfgDesign.eventtypes = {'fixation','saccade'}; % we model the fixation onset
cfgDesign.formula = model ; %saccade_amp + cat(hard_easy)'}%+stimulusDur+cat(hard_easy)'}
EEG = uf_designmat(EEG,cfgDesign);



%TIME EXPAND
cfgTimeexpand = [];
cfgTimeexpand.timelimits = [-.2,.6];
EEG = uf_timeexpandDesignmat(EEG,cfgTimeexpand);
%keyboard


%FITTING MODEL 
EEG= uf_glmfit(EEG);

%%
figure;

% Divide the window into two subplots, arranged side by side
subplot(1,2,1);
plot(EEG.unfold.times, EEG.unfold.beta_dc(1,:,1));
title('Plot 1');

subplot(1,2,2);
plot(EEG.unfold.times, EEG.unfold.beta_dc(1,:,2));
title('Plot 2');




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
