% load_wav.m - Load and filter audio file to remove harmonics
% Specify the path to your .wav file
filename = 'salamander.wav';

% Load the audio file
[data, samplingFrequency] = audioread(filename);

% Display some information about the audio file
disp(['Sample frequency: ', num2str(samplingFrequency), ' Hz']);
disp(['Number of Samples: ', num2str(length(data))]);

% Apply bandpass filter to focus on fundamental frequencies and remove harmonics
lowFreq = 51;  % Hz
highFreq = 900; % Hz

% Design a bandpass filter to isolate fundamental frequencies
bandpassFilt = designfilt('bandpassfir', ...
                     'StopbandFrequency1', lowFreq-50, ...
                     'PassbandFrequency1', lowFreq, ...
                     'PassbandFrequency2', highFreq, ...
                     'StopbandFrequency2', highFreq+50, ...
                     'StopbandAttenuation1', 60, ...
                     'PassbandRipple', 1, ...
                     'StopbandAttenuation2', 60, ...
                     'SampleRate', samplingFrequency);

% Apply filter to remove harmonics and noise outside our frequency range of interest
data = filter(bandpassFilt, data);
%sound(data, samplingFrequency);