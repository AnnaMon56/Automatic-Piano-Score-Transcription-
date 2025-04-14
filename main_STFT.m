% Audio Note Detection with Spectrogram and Component Reconstruction
clear;close all;clc

% Load audio data (this now includes filtering)
load_wav;

% Sampling frequency
Fs = samplingFrequency;

% Use the ENTIRE audio file
x = data;

% Use a shorter window length for better time resolution with sequential notes
win_length = ceil(0.025 * Fs); 
hop_size = ceil(win_length/4); 
num_seg = floor((numel(x) - win_length)/hop_size) + 1;

% Apply window function
window_func = hann(win_length);
window_func = window_func(:); 

% Segment the audio
x_seg = zeros(win_length, num_seg);
for i = 1:num_seg
    start_id = (i-1) * hop_size + 1;
    end_id = start_id + win_length - 1;
    if end_id <= length(x)
        segment = x(start_id:end_id);
        segment = segment(:); % Force column vector
        x_seg(:,i) = segment .* window_func;
    end
end

% Compute STFT
nfft = 2^nextpow2(win_length); 
X = zeros(nfft/2+1, num_seg);

% Check for GPU support
try
    gpuDevice();
    useGPU = true;
catch
    useGPU = false;
end

for i = 1:num_seg
    % Compute FFT with potential GPU acceleration
    if useGPU
        try
            temp = fft(gpuArray(x_seg(:,i)), nfft);
            temp = gather(temp(1:nfft/2+1));
        catch
            temp = fft(x_seg(:,i), nfft);
            temp = temp(1:nfft/2+1);
        end
    else
        temp = fft(x_seg(:,i), nfft);
        temp = temp(1:nfft/2+1);
    end
    
    X(:,i) = temp;
end

% Calculate magnitude spectrogram
V = abs(X);

% Calculate frequency array
f = (0:nfft/2) * Fs/nfft;

% Time points
t = (0:num_seg-1) * hop_size / Fs;

% Display the spectrogram
figure;
imagesc(t, f, 20*log10(V + eps)); 
axis xy; 
title('Spectrogram');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
ylim([40 1500]); % Modified to match our adjusted frequency range
colorbar;

% Use a fixed small number of components
r = 5; 
fprintf('Optimal NMF component count: %d\n', r);

rng(42);

% Apply NMF with optimal components
[W, H] = NMF(V, r);


Y = cell(r, 1);
y = cell(r, 1);
reconstructed_x = zeros(size(x));

for comp = 1:r
    % Create rank-1 approximation by outer product of W and H components
    Y{comp} = W(:, comp) * H(comp, :);

    Y_complex = Y{comp} .* exp(1i * angle(X));
    
    % Convert to full FFT representation (including negative frequencies)
    Y_full = zeros(nfft, num_seg);
    Y_full(1:nfft/2+1, :) = Y_complex;
    for i = 2:nfft/2
        Y_full(nfft-i+2, :) = conj(Y_complex(i, :));
    end
    
    % Inverse FFT to get time domain segments
    y_seg = zeros(nfft, num_seg);
    for i = 1:num_seg
        y_seg(:, i) = real(ifft(Y_full(:, i)));
    end
    
    % Overlap-add to reconstruct the time signal
    y{comp} = zeros(length(x), 1);
    
    for i = 1:num_seg
        start_id = (i-1) * hop_size + 1;
        end_id = start_id + nfft - 1;
        if end_id <= length(y{comp})
            y{comp}(start_id:end_id) = y{comp}(start_id:end_id) + y_seg(:, i);
        end
    end
    
    % Normalize to avoid clipping
    y{comp} = y{comp} / max(abs(y{comp}));
    
    % Add to the reconstructed signal
    reconstructed_x = reconstructed_x + y{comp};
end

% Normalize reconstructed signal to avoid clipping
reconstructed_x = reconstructed_x / max(abs(reconstructed_x));

% Pre-allocate for all_notes with reasonable size
max_expected_notes = 500;
all_notes = zeros(max_expected_notes, 7);
note_count = 0;

% Minimum duration for note detection
min_note_duration = 0.1; 

% Process each component
for comp = 1:r
    % Create a weighted spectrum that emphasizes lower frequencies
    weighted_W = W(:, comp) .* (1 ./ (1 + 0.1 * f'));
    
    % Find prominent peaks in weighted frequency spectrum
    [peaks, locs] = findpeaks(weighted_W, 'SortStr', 'descend', 'MinPeakHeight', max(weighted_W)*0.3);
    
    % Focus on peaks in the expected frequency range for musical notes (40-1000 Hz)
    valid_peaks = find(f(locs) >= 40 & f(locs) <= 1000);
    
    if ~isempty(valid_peaks)
        peak_idx = locs(valid_peaks(1));
        
        % Improve frequency estimation with enhanced parabolic interpolation
        peak_freq = refine_frequency(W(:, comp), peak_idx, f);
        
        fprintf('Component %d detected frequency: %.2f Hz\n', comp, peak_freq);
        fprintf('MIDI value before rounding: %.2f\n', 12 * log2(peak_freq/440) + 69);
        
        % Get the activation for this component
        activation = H(comp, :);
        
        % Normalize activation to 0-1 range
        activation = activation / max(activation + eps);
        activation_mean = mean(activation);
        activation_std = std(activation);
        threshold = activation_mean + 0.4 * activation_std;

        % Apply threshold
        note_mask = activation > threshold;
        
        % Find note segmentation (start/end points)
        note_changes = diff([0, note_mask, 0]);
        note_starts = find(note_changes == 1);
        note_ends = find(note_changes == -1) - 1;
        
        % Process each note segment
        for n = 1:length(note_starts)
            start_idx = note_starts(n);
            end_idx = note_ends(n);
            
            % Convert to time
            start_time = t(start_idx);
            end_time = t(end_idx);
            duration = end_time - start_time;
            
            % Only keep notes longer than the minimum duration
            if duration >= min_note_duration
                % Calculate average intensity
                avg_intensity = mean(activation(start_idx:end_idx));
                velocity = round(avg_intensity * 127);
                
                % Convert frequency to MIDI note
                [midi_note, note_name, cents_off] = freq_to_note(peak_freq);
                
                % Add to notes list
                note_count = note_count + 1;
                if note_count <= max_expected_notes
                    all_notes(note_count, :) = [start_time, end_time, duration, peak_freq, midi_note, velocity, comp];
                else
                    % Expand if needed
                    all_notes = [all_notes; [start_time, end_time, duration, peak_freq, midi_note, velocity, comp]];
                    max_expected_notes = max_expected_notes * 2;
                end
            end
        end
    end
end

% Trim unused rows from all_notes
if note_count < size(all_notes, 1)
    all_notes = all_notes(1:note_count, :);
end

% Sort notes by start time
if ~isempty(all_notes)
    all_notes = sortrows(all_notes, 1);
end

% Add octave correction based on harmonics analysis
for i = 1:size(all_notes, 1)
    % Check if this note could be an octave higher than the true fundamental
    freq = all_notes(i, 4);
    half_freq = freq / 2;
    
    % Check if a half-frequency note would be in our expected range
    if half_freq >= 40 % Reasonable lower bound for musical notes
        % Convert to MIDI for comparison
        [midi_half, ~, ~] = freq_to_note(half_freq);
        
        % Check the energy at half frequency to determine if it's the true fundamental
        half_freq_idx = round(half_freq/(Fs/nfft)) + 1;
        if half_freq_idx <= size(W, 1)
            energy_ratio = W(half_freq_idx, all_notes(i, 7)) / W(round(freq/(Fs/nfft)) + 1, all_notes(i, 7));
            
            % If there's significant energy at half the frequency, it's likely the true note
            if energy_ratio > 0.3 % Threshold can be tuned
                % Update the note to the octave below
                all_notes(i, 4) = half_freq;
                all_notes(i, 5) = midi_half;
                fprintf('Corrected octave for note %d: %.2f Hz -> %.2f Hz\n', i, freq, half_freq);
            end
        end
    end
end

i = 1;
while i < size(all_notes, 1)
    % Check if current note overlaps with next note
    if i < size(all_notes, 1) && all_notes(i, 2) > all_notes(i+1, 1)
        % If same pitch and same component, keep stronger one
        if all_notes(i, 5) == all_notes(i+1, 5) && all_notes(i, 7) == all_notes(i+1, 7)
            if all_notes(i, 6) < all_notes(i+1, 6)
                all_notes(i, :) = [];
            else
                all_notes(i+1, :) = [];
            end
        else
            i = i + 1;
        end
    else
        i = i + 1;
    end
end

i = 1;
while i < size(all_notes, 1)
    j = i + 1;
    while j <= size(all_notes, 1)
        % Check if notes overlap in time
        time_overlap = (all_notes(i,1) < all_notes(j,2)) && (all_notes(j,1) < all_notes(i,2));
        
        % Check if higher note is approximately a harmonic of lower note
        freq_ratio = all_notes(j,4) / all_notes(i,4);
        is_harmonic = abs(freq_ratio - round(freq_ratio)) < 0.1 && round(freq_ratio) > 1;
        
        if time_overlap && is_harmonic
            % Remove the higher frequency note (harmonic)
            all_notes(j,:) = [];
        else
            j = j + 1;
        end
    end
    i = i + 1;
end

i = 1;
while i < size(all_notes, 1)
    j = i + 1;
    if j <= size(all_notes, 1) && all_notes(i,5) == all_notes(j,5) % Same MIDI note
        % If notes are very close but not overlapping
        if all_notes(j,1) - all_notes(i,2) < 0.1 && all_notes(j,1) > all_notes(i,2)
            % Check if they might be separate notes
            midpoint = (all_notes(i,2) + all_notes(j,1))/2;
            % Split them more distinctly
            all_notes(i,2) = midpoint - 0.02;
            all_notes(j,1) = midpoint + 0.02;
        end
    end
    i = i + 1;
end

% Add post-processing for musical consistency
if size(all_notes, 1) > 3
    % Calculate median note range
    median_midi = median(all_notes(:, 5));
    
    % Check for notes that are far above the median
    for i = 1:size(all_notes, 1)
        if all_notes(i, 5) > median_midi + 10 % If more than 10 semitones above median
            % Check if an octave adjustment would bring it closer to the median
            potential_octave_down = all_notes(i, 5) - 12;
            if abs(potential_octave_down - median_midi) < abs(all_notes(i, 5) - median_midi)
                % Adjust the note down an octave
                all_notes(i, 5) = potential_octave_down;
                all_notes(i, 4) = all_notes(i, 4) / 2; % Adjust frequency
                fprintf('Adjusted note %d down an octave for musical consistency\n', i);
            end
        end
    end
end

% Perform more conservative tuning calibration
fprintf('Analyzing pitch deviations for tuning calibration...\n');
if ~isempty(all_notes)
    % Analyze pitch deviations
    deviations = zeros(size(all_notes, 1), 1);
    for i = 1:size(all_notes, 1)
        freq = all_notes(i, 4);
        midi_fractional = 12 * log2(freq/440) + 69;
        deviations(i) = midi_fractional - round(midi_fractional);
    end
    
    % Calculate median deviation in cents
    median_dev_cents = median(deviations) * 100;
    deviation_std = std(deviations) * 100;
    fprintf('Median pitch deviation: %.1f cents (std: %.1f)\n', median_dev_cents, deviation_std);
    
    % Only adjust if very consistent deviation is found
    if abs(median_dev_cents) > 25 && abs(median_dev_cents) < 45 && deviation_std < 15
        % Adjust all note frequencies and MIDI values
        tuning_factor = 2^(median_dev_cents/(1200));
        original_freqs = all_notes(:, 4);
        all_notes(:, 4) = all_notes(:, 4) / tuning_factor;
        
        % Recalculate MIDI values
        for i = 1:size(all_notes, 1)
            [all_notes(i, 5), ~, ~] = freq_to_note(all_notes(i, 4));
        end
        
        fprintf('Adjusted tuning by %.1f cents\n', median_dev_cents);
        
        % Print before/after for a few notes
        for i = 1:min(5, size(all_notes, 1))
            [~, old_note, ~] = freq_to_note(original_freqs(i));
            [~, new_note, ~] = freq_to_note(all_notes(i, 4));
            fprintf('Note %d: %s (%.1f Hz) -> %s (%.1f Hz)\n', i, old_note, original_freqs(i), new_note, all_notes(i, 4));
        end
    else
        fprintf('No tuning adjustment applied - not consistent enough.\n');
    end
end

% Display detected notes
fprintf('\nDetected Notes:\n');
fprintf('----------------\n');
fprintf('Start Time | End Time  | Duration | Frequency | MIDI | Note    | Velocity | Component\n');
fprintf('----------|-----------|----------|-----------|------|---------|----------|----------\n');

if ~isempty(all_notes)
    for i = 1:size(all_notes, 1)
        start_time = all_notes(i, 1);
        end_time = all_notes(i, 2);
        duration = end_time - start_time;
        freq = all_notes(i, 4);
        midi = all_notes(i, 5);
        velocity = all_notes(i, 6);
        comp = all_notes(i, 7);
        
        % Get note name
        [~, note_name, ~] = freq_to_note(freq);
        
        fprintf('%9.3f | %9.3f | %8.3f | %9.1f | %4d | %-7s | %8d | %9d\n', ...
                start_time, end_time, duration, freq, midi, note_name, velocity, comp);
    end
else
    fprintf('No notes detected.\n');
end

% Create piano roll visualization
figure;
hold on;

% Create a distinct color for each component
colors = hsv(r);

% Plot each note as a rectangle
for i = 1:size(all_notes, 1)
    start_time = all_notes(i, 1);
    end_time = all_notes(i, 2);
    midi_note = all_notes(i, 5);
    velocity = all_notes(i, 6) / 127; % Normalize velocity to 0-1
    comp = all_notes(i, 7);
    
    % Plot note as rectangle with component-specific color
    rectangle('Position', [start_time, midi_note-0.4, end_time-start_time, 0.8], ...
              'FaceColor', [colors(comp,:), velocity], ...
              'EdgeColor', 'k');
    
    % Add note name text
    [~, note_name, ~] = freq_to_note(all_notes(i, 4));
    text(start_time + (end_time-start_time)/2, midi_note, note_name, ...
         'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
         'FontSize', 8);
end

title('Detected Notes Piano Roll');
xlabel('Time (s)');
ylabel('MIDI Note Number');
grid on;

% Add a colorbar legend for components
colormap(hsv(r));
cb = colorbar;
cb.Ticks = linspace(0, 1, r);
cb.TickLabels = arrayfun(@num2str, 1:r, 'UniformOutput', false);
ylabel(cb, 'Component');

% Display summary
fprintf('\nTotal notes detected: %d\n', size(all_notes, 1));

% Function to convert frequency to note name with improved semitone handling
function [midi_note, note_name, cents_off] = freq_to_note(freq)
    % Reference frequency for A4
    A4_freq = 440; % Hz
    A4_midi = 69;  % MIDI note number
    
    % Calculate MIDI note number (fractional)
    midi_fractional = 12 * log2(freq/A4_freq) + A4_midi;
    
    % Calculate cents off from nearest semitone
    cents_off = 100 * (midi_fractional - round(midi_fractional));
    
    % Use more intelligent rounding - apply a small bias to make sure
    % frequencies very close to the boundary round to the correct note
    if abs(cents_off) > 40
        % If we're more than 40 cents off, be more careful with rounding
        if cents_off > 0
            midi_note = floor(midi_fractional + 0.5);
        else
            midi_note = ceil(midi_fractional - 0.5);
        end
    else
        % Standard rounding for most cases
        midi_note = round(midi_fractional);
    end
    
    % Recalculate cents off based on final MIDI note
    cents_off = 100 * (midi_fractional - midi_note);
    
    % Get note name
    note_idx = mod(midi_note, 12) + 1;
    octave = floor(midi_note / 12) - 1;
    
    % Note names
    note_names = {'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'};
    
    % Return note name with octave
    note_name = [note_names{note_idx}, num2str(octave)];
end

% Improved frequency refinement function with better handling of edge cases
function refined_freq = refine_frequency(V, peak_idx, f)
    % Get magnitudes around the peak
    if peak_idx > 1 && peak_idx < length(V)
        y0 = V(peak_idx-1);
        y1 = V(peak_idx);
        y2 = V(peak_idx+1);
        
        % Avoid division by zero
        if (y0 - 2*y1 + y2) ~= 0
            % Parabolic interpolation
            delta = 0.5 * (y0 - y2) / (y0 - 2*y1 + y2);
            
            % Constrain delta to avoid extreme values
            delta = max(-1, min(1, delta));
            
            % Calculate refined frequency
            df = f(2) - f(1);
            refined_freq = f(peak_idx) + delta * df;
        else
            refined_freq = f(peak_idx);
        end
    else
        refined_freq = f(peak_idx);
    end
end