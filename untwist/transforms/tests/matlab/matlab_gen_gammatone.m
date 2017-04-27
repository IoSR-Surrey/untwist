% NOTE: these functions have been edited for untwist validation purposes
flo = 100;                     % Lowest center frequency in Hz;
fs = 44100;                    % Sampling rate in Hz;
numChannels = 24;
[fcoefs, cf, erb] = MakeERBFilters(fs,numChannels,flo);

N = 8192;                     % Number of samples;
insig = [1, zeros(1, N)];     % Impulse signal;

outsig = ERBFilterBank(insig,fcoefs);    

save('../../../../data/test_data/gammatone.mat', 'outsig', 'cf', 'erb');