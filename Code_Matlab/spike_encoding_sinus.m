%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%       Encodage d'un signal sinusoidal                                   %
%                                                                         %
%       Etudiant : FAVREAU Francois                                       %
%       Directeur : ROUAT Jean                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all,
clc;

% Parameters
simul_duration = 1; % amount of time given to the simulation (seconds)

nbchan = 1;     % EEG.nbchan
nb_samples = 1000*simul_duration;  % EEG.pnts
stimuli = 1;    % EEG.trials

% Create the matrix corresponding to spikes 
final_output_spike = zeros(nbchan, nb_samples, stimuli);

threshold_ampl_freq = 1; % threshold for ampl_freq algorithm
threshold_BSA = 0.86;    % threshold for BSA algorithm

% Time characteristics
Fs = nb_samples;    % sampling frequency (sample per second)
dt = simul_duration/nb_samples;  % second per sample
stopTime = simul_duration;       % second
t = (0:dt:stopTime-dt)';     % seconds    

% Sine wave 
Fc = 10;           % frequency (Hertz)
y = 0.5*(1+sin(2*pi*Fc*t)); % continuous function
   
% Type of spike encoding choice
mode = 0; % amplitude/frequency = 0 ; BSA = 1
switch mode
    case 0
        encoding_type = 'Encoding type : Amplitude/frequency';
        final_output_spike = ampl_freq( y, threshold_ampl_freq, nb_samples, final_output_spike );
    case 1
        encoding_type = 'Encoding type : BSA algorithm';
        final_output_spike = BSA_algo( y, threshold_BSA, nb_samples, final_output_spike );
    otherwise
        disp('Choose type of spike encoding : amplitude/frequency = 0 ; BSA = 1')
end

% Blank in signal
blank = 1;
start_blank = 300;
length_blank = 50;
if (blank==0)
    for i=start_blank:start_blank+length_blank
        final_output_spike(:,i,:) = 0;
    end
end
        
% Display spike trains
display_input_and_spikeTrains( y, final_output_spike, t, encoding_type )

% Save matrix
if mode==0
    file_name = sprintf('dataset_%s_%d_sec','ampl_freq', simul_duration);
elseif mode==1
    file_name = sprintf('dataset_%s_%d_sec','BSA', simul_duration);    
end
save(file_name,'final_output_spike')

