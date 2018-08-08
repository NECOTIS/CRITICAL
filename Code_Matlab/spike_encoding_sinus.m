%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%             Encodage d'un signal sinusoidal                             %
%                                                                         %
%       Étudiant : FAVREAU Francois                                       %
%       Directeur : ROUAT Jean                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all,
clc;

nbchan = 1;     % EEG.nbchan
nb_samples = 1000;  % EEG.pnts
stimuli = 1;    % EEG.trials

% Create the matrix corresponding to spikes 
final_output_spike = zeros(nbchan, nb_samples, stimuli);

threshold_ampl_freq = 5; % threshold for ampl_freq algorithm
threshold_BSA = 0.86; % threshold for ampl_freq algorithm

% Sampling sinusoidal function
Ts = 100;       % sinus period
fs = 1/Ts;      % sinus frequency
t = 1:nb_samples;     % time sampling
y = sin(2*pi*fs*t); % sinus generation

% Type of spike encoding choice
mode = 1; % amplitude/frequency = 0 ; BSA = 1
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

% Display spike trains
display_input_and_spikeTrains( y, final_output_spike, t, encoding_type )

% Save matrix
save('C:\Users\ffavreau\PycharmProjects\Essai\6_Reservoir_RNN\Code_Matlab\dataset_BSA','final_output_spike')

