from matplotlib import pyplot as plt


class ploter:

    def __init__(self, data: tuple):
        self.data = data
        self.time_values = range(len(data[0][0]))

    def plot(self, file=None, title=""):
        plt.figure(figsize=(10, 6))
        for d in self.data[0]:
            plt.plot(self.time_values, d[0])

        plt.title(title)
        plt.legend()
        plt.grid()

        if file:
            plt.savefig(file)

        plt.close()
        plt.cla()
        plt.clf()


if __name__ == "__main__":
    forPloting = ((amplitude_values, {"label": 'Original Data'}),
                  (fitted_function, {"label": 'Fitted Sinusoidal Function', "linestyle": '--'}))

    ploter(forPloting).plot(file_path + r".png")

    ploter(((amplitude_values - fitted_function,), {"label": "residum"})).plot(file_path + r"residum.png")