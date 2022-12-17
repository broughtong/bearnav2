#!/usr/bin/env python
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


filenames = ["recorded_alignment_2d.txt", "recorded_alignment.txt"]


if __name__ == "__main__":
    import pandas as pd
    plt.figure()
    for f in filenames:
        df = pd.read_csv(f)
        vals = df.values
        trimmed = np.trim_zeros(vals, trim="f")[:1480]
        plt.plot(trimmed)
        print(np.mean(abs(trimmed)))
    plt.legend(filenames)
    plt.show()