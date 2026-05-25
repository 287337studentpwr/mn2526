import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

df = pd.read_csv("data_lista_1.csv")

def plot(x, f_1, f_2):
    plt.figure(figsize=(5, 2.7), layout='constrained')
    plt.plot(x, f_1, label ="f_1")
    plt.plot(x, f_2, label = "f_2")
    plt.xlabel('x label')
    plt.ylabel('y label')
    plt.title("Simple Plot")
    plt.legend()
    plt.show()
    
def plot_diff(x, f_1, f_2):
    plt.figure(figsize=(5, 2.7), layout='constrained')
    plt.plot(x, abs(f_1 - f_2), label= "diff")
    plt.xlabel('x label')
    plt.ylabel('y label')
    plt.title("Simple Plot")
    plt.legend()
    plt.show()

plot(df["x"], df["g1"], df["g2"])
plot_diff(df["x"], df["g1"], df["g2"])