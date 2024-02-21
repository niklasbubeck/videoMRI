import os 
import sys 
import glob 
from matplotlib import pyplot as plt 
import pandas as pd 
import seaborn as sns



def vis_boxplot(dataframe, x='slice_nr', y='ssim'): 
    sns.set_theme(style="ticks")

    # Initialize the figure with a logarithmic x axis
    f, ax = plt.subplots(figsize=(7, 6))
    # ax.set_xscale("log")

    print(dataframe)

    # Plot the orbital period with horizontal boxes
    sns.boxplot(
        dataframe, x=x, y=y
    )

    # Add in points to show each observation
    sns.stripplot(dataframe, x=x, y=y, size=4, color=".3")

    # Tweak the visual presentation
    ax.xaxis.grid(True)
    ax.set(ylabel="")
    sns.despine(trim=True, left=True)
    plt.savefig("/vol/aimspace/users/bubeckn/diffusion-autoencoders/outputs/out.png")


if __name__ == "__main__":
    path = "/vol/aimspace/users/bubeckn/diffusion-autoencoders/outputs/diffusion/1diffae3D/test/videos/interpolation/data.csv"
    df = pd.read_csv(path).sort_values( by="slice_nr",ascending=True)
    vis_boxplot(df, y="ssim")