import matplotlib.pyplot as plt 

def visualize(rows, cols):
    """
    Send one subplot at a time 
    """
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 8))
    for plots in axes[:,:]:
        for subplot in plots:
            yield subplot  