function [] = display_input_and_spikeTrains( y, final_output_spike, t, encoding_type )
% Display input signal and the spike trains corresponding
    
    figure

    subplot(2,1,1)
    plot(t,y);
    title('Input signal');
    xlabel('Time (seconds)');
    ylabel('Amplitude');


    subplot(2,1,2)
    stem(final_output_spike);
    xlabel('Time (samples)');
    title('Spike trains');
    
    suptitle(encoding_type)

end

