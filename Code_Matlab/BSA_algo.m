function [ final_output_spike ] = BSA_algo( y, BSA_threshold, nb_samples, final_output_spike )
% BSA algorithm

    % FIR filter
    FIR_filter = fir1(10,0.05);

    % BSA
    for i=1:nb_samples
        error_matrix_1 = 0;
        error_matrix_2 = 0;
        for j = 1:length(FIR_filter)
            if (i+j-1) < nb_samples
                error_matrix_1 = error_matrix_1 + abs(y(i+j-1) - FIR_filter(j));
                error_matrix_2 = error_matrix_2 + abs(y(i+j-1));
            end
        end
        
        if error_matrix_1 <= (error_matrix_2 - BSA_threshold)
            final_output_spike(:,i,:) = 1;         

            for j = 1:length(FIR_filter)
                if (i+j-1) <= length(y)
                    y(i+j-1) = y(i+j-1) - FIR_filter(j);
                end
            end
            
        else
            final_output_spike(:,i,:) = 0;      
        end
    end

        
end

