import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

from main import read_csv_file, sinusoidal_function

file_paths = [rf"Data\Norm_map_zwobble_{i}_detectorSum_ROI_Cr.csv" for i in range(1, 7, 1)]
colors = ["b", "r", "g", "y", "m", "k"]

data = read_csv_file(file_paths[0])
# Convert the data to a NumPy array
data_array = np.array(data, dtype=float)

dane = np.zeros(data_array.shape[0])
for file_path, color in zip(file_paths, colors):

    data = read_csv_file(file_path)
    # Convert the data to a NumPy array
    data_array = np.array(data, dtype=float)

    # for i in range(data_array.shape[0]):
    #    plt.plot(i, np.argmax(data_array[i, :]), marker='.', color=color)
    # plt.plot([np.argmax(data_array[i, :]) for i in range(data_array.shape[0])], color=color)
    for i, (d, r) in enumerate(zip(dane, [np.argmax(data_array[i, :]) for i in range(data_array.shape[0])])):
        dane[i] = d + r

dane = np.asarray(dane)
np.multiply(dane, 1.0 / 6.0)
plt.plot(dane)

plt.savefig("3.png")

time_values = range(len(dane))  # Time values
amplitude_values = dane  # Example signal

# Perform FFT
fft_result = np.fft.fft(amplitude_values)

# Compute the frequencies
sampling_frequency = 1 / (time_values[1] - time_values[0])  # Sampling frequency
frequencies = np.fft.fftfreq(len(amplitude_values), d=1 / sampling_frequency)

# Find index of maximum amplitude in FFT result (excluding DC component at index 0)
max_index = np.argmax(np.abs(fft_result[1:])) + 1

# Initial guesses for fitting parameters
initial_amplitude_guess = np.abs(fft_result[max_index]) / len(amplitude_values)
initial_frequency_guess = frequencies[max_index]
initial_phase_guess = np.angle(fft_result[max_index])

# Perform curve fitting
popt, pcov = curve_fit(sinusoidal_function, time_values, amplitude_values,
                       p0=[initial_amplitude_guess, initial_frequency_guess, initial_phase_guess])

# Extract fitted parameters
fitted_amplitude, fitted_frequency, fitted_phase = popt

print(
    f"{file_path}:\nfitted_amplitude:{fitted_amplitude} \nfitted_frequency:{fitted_frequency} \nfitted_phase:{fitted_phase}")

# Generate fitted sinusoidal function
fitted_function = sinusoidal_function(time_values, fitted_amplitude, fitted_frequency, fitted_phase)

lam = lambda x, a, b, c, f=fitted_function: fitted_function[x] + a * x + b + c * x * x

popt, pcov = curve_fit(lam, time_values, amplitude_values, p0=[0, 0, 0])

# Generate fitted sinusoidal function
fitted_function = lam(time_values, *popt)
print(
    f"wspulczynikKierunkowy:{popt[0]} \noffset:{popt[1]} \nwspulczynikKwadratowy:{popt[2]}")

# Plot original data and fitted function
plt.figure(figsize=(10, 6))
plt.plot(time_values, amplitude_values, label='Original Data')
plt.plot(time_values, fitted_function, label='Fitted Sinusoidal Function', linestyle='--')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Original Data and Fitted Sinusoidal Function')
plt.legend()
plt.grid()
# plt.show()
plt.savefig(r"new.png")
plt.close()
plt.cla()
plt.clf()
# plt.show()


#    fiting(data_array)
