import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

df1 = pd.read_csv("data_lista_2_f1.csv")
df2 = pd.read_csv("data_lista_2_f2.csv")

def plot(df, f_name, exercise):
    plt.figure(figsize=(5, 2.7), layout='constrained')
    for degree in range(1,20 + 1):
        df = df1[df1["degree"] == degree]
        if degree == 1:
            plt.plot(df["x"], df["exact"], label = f_name)
        plt.plot(df["x"], df["approx"], label = f"degree = {degree}")
    plt.xlabel('x label')
    plt.ylabel('y label')
    plt.title(f"exercise {exercise}")
    plt.legend()
    plt.show()


plot(df1, "|x|", 3)
plot(df2, "(1/x^2 + 1)", 4)