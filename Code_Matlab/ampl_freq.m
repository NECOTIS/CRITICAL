function [ final_output_spike ] = ampl_freq( y, threshold, nb_samples, final_output_spike )
% Fill the spike matrix corresponding to the signal

    tamp = 0;
    for i=1:nb_samples
        tamp = tamp + y(i);
        if tamp > threshold
            final_output_spike(:,i,:) = 1;
            tamp = 0;
        else
            final_output_spike(:,i,:) = 0;
        end
    end
end

