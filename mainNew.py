import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from analize import analizer
from furier import resolveFurier
from main import resolvData, read_csv_file

file_paths = [rf"Data\Norm_map_zwobble_{i}_detectorSum_ROI_Cr.csv" for i in range(1, 7, 1)]


# Define the sinusoidal function
def cosuidal_function(t, amplitude, frequency, phase):
    return amplitude * np.cos(2 * np.pi * frequency * t + phase)


def sinusoidal_function(t, amplitude, frequency, phase):
    return amplitude * np.sin(2 * np.pi * frequency * t + phase)


for file_path in file_paths:
    print(file_path)

    amplitude_values = resolvData(np.array(read_csv_file(file_path), dtype=float))

    time_values = range(len(amplitude_values))  # Time values

    a,b,c = resolveFurier(amplitude_values).fff()



    fitted_function, popt = analizer(amplitude_values, cosuidal_function, [10,b, c]).fit()

    fft_result = resolveFurier(fitted_function-amplitude_values).fff()

    fitted_function, popt = analizer(amplitude_values, lambda x, amplitude, frequency, phase, f=fitted_function: f[x] + cosuidal_function(x, amplitude, frequency, phase), [*fft_result]).fit()

    # fitted_function, popt = analizer(amplitude_values,
    #                                 lambda t, amplitude, frequency, phase, f=fitted_function : f[t] + sinusoidal_function(t, amplitude, frequency, phase),
    #                                 [np.abs(fft_result[max_index]) / len(amplitude_values), frequencies[max_index],
    #                                  np.angle(fft_result[max_index])]).fit()

    fitted_amplitude, fitted_frequency, fitted_phase = popt

    fitted_function, popt = analizer(amplitude_values,
                                     lambda x, A, B, C, f=fitted_function: f[x] * (A * x * x + B * x + C),
                                     [0, 0, fitted_amplitude]).fit()

    fitted_function, _ = analizer(amplitude_values,
                                  lambda x, a, b, c, f=fitted_function: f[x] + a * x + b + c * x * x,
                                  [0, 0, 0]).fit()

    #fft_result = resolveFurier(fitted_function-amplitude_values).fff()

    #fitted_function, popt = analizer(amplitude_values, lambda x, amplitude, frequency, phase, f=fitted_function: f[x] + cosuidal_function(x, amplitude, frequency, phase), [*fft_result]).fit()

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
    plt.savefig(file_path + r".png")
    plt.close()
    plt.cla()
    plt.clf()

    plt.figure(figsize=(10, 6))
    plt.plot(time_values, amplitude_values-fitted_function, label='Original Data')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title('Residum')
    plt.legend()
    plt.grid()
    # plt.show()
    plt.savefig(file_path + r"residum.png")
    plt.close()
    plt.cla()
    plt.clf()

