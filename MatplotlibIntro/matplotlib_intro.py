# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
<Name> Dallin Seyfried
<Class> Math 321 - Vol 2
<Date> 9/15/22
"""
import matplotlib
import numpy as np
from matplotlib import pyplot as plt


# Problem 1
def var_of_means(n):
    """ Create an (n x n) array of values randomly sampled from the standard
    normal distribution. Compute the mean of each row of the array. Return the
    variance of these means.

    Parameters:
        n (int): The number of rows and columns in the matrix.

    Returns:
        (float) The variance of the means of each row.
    """
    # Use numpy to generate the matrix, then calculate the means and variance of it
    rand_mat = np.random.normal(size=(n, n))
    mean_mat = np.mean(rand_mat, axis=1)
    var_mat = np.var(mean_mat)
    return var_mat

def prob1():
    """ Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """
    # Initialize x and y axis arrays
    inputs = [x for x in range(100, 1001, 100)]
    outputs = []

    # For each input append the output for var_of_means to outputs
    for input in inputs:
        outputs.append(var_of_means(input))

    # Use matplotlib to construct axes, plot data, then show the plot
    plt.plot(inputs, outputs)
    plt.title("Problem 1 - Variance of Means")
    plt.xlabel("Input for var_of_means")
    plt.ylabel("Output for var_of_means")
    plt.show()

# Problem 2
def prob2():
    """ Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    x = np.linspace(-2 * np.pi, 2 * np.pi, 100)

    # Create, label, and plot sin, cos, arctan graphs
    f = np.sin(x)
    g = np.cos(x)
    h = np.arctan(x)
    plt.plot(x, f, label="sin")
    plt.plot(x, g, label="cos")
    plt.plot(x, h, label="arctan")
    plt.title("Problem 2 - functions: sin, cos, arctan")
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.legend(loc="upper left")
    plt.show()



# Problem 3
def prob3():
    """ Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Set the range of the x-axis to [-2,6] and the range of the
           y-axis to [-6,6].
    """
    # Split and generate domains
    x_left = np.linspace(-2, 1, 100, endpoint=False)
    x_right = np.linspace(1, 6, 100)[1:]

    # Build left and right sides of function
    f_left = 1 / (x_left - 1)
    f_right = 1 / (x_right - 1)

    # Plot label and show the two functions together
    plt.plot(x_left, f_left, 'm--', lw=4, label="1 / (x - 1)")
    plt.plot(x_right, f_right, 'm--', lw=4)
    plt.title("Problem 3")
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.xlim(-2, 6)
    plt.ylim(-6, 6)
    plt.legend(loc="upper left")
    plt.show()


# Problem 4
def prob4():
    """ Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi], each in a separate subplot of a single figure.
        1. Arrange the plots in a 2 x 2 grid of subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """
    x = np.linspace(0, 2 * np.pi, 100)

    # Create and subplot the sin(x) graph
    ax1 = plt.subplot(221)
    ax1.plot(x, np.sin(x), 'g-')
    plt.xlim(0, 2 * np.pi)
    plt.ylim(-2, 2)
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.title("sin(x)")

    # Create and subplot the sin(2x) graph
    ax2 = plt.subplot(222)
    ax2.plot(x, np.sin(2 * x), 'r--')
    plt.xlim(0, 2 * np.pi)
    plt.ylim(-2, 2)
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.title("sin(2x)")

    # Create and subplot the 2sin(x) graph
    ax3 = plt.subplot(223)
    ax3.plot(x, 2 * np.sin(x), 'b--')
    plt.xlim(0, 2 * np.pi)
    plt.ylim(-2, 2)
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.title("2sin(x)")

    # Create and subplot the 2sin(2x) graph
    ax4 = plt.subplot(224)
    ax4.plot(x, 2 * np.sin(2 * x), 'm:')
    plt.xlim(0, 2 * np.pi)
    plt.ylim(-2, 2)
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.title("2sin(2x)")

    # Show all four subplots
    plt.suptitle("Problem 4 - Variations on sin(x)")
    plt.tight_layout()
    plt.show()



# Problem 5
def prob5():
    """ Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes against latitudes. Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """
    # Get and parse the data from FARS.npy
    data = np.load("FARS.npy")
    times = data[:, 0]
    longitudes = data[:, 1]
    latitudes = data[:, 2]

    # Create and subplot the scatter plot for coordinates
    ax1 = plt.subplot(121)
    ax1.plot(longitudes, latitudes, 'k,')
    plt.xlabel("longitudes")
    plt.ylabel("latitudes")
    plt.title("Longitude and Latitudes Scatterplot")
    plt.axis("equal")

    # Create and subplot the histogram for time
    ax2 = plt.subplot(122)
    ax2.hist(times, bins=24, range=[0, 24])
    plt.xlabel("Hours")
    plt.ylabel("Amount")
    plt.title("Times Histogram")
    plt.xlim(0, 24)

    plt.tight_layout()
    plt.show()

# Problem 6
def prob6():
    """ Plot the function g(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of g, and one with a contour
            map of g. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Include a color scale bar for each subplot.
    """

    # Setup Domain and Create the Function
    x = np.linspace(-2 * np.pi, 2 * np.pi)
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    g = np.sin(X) * np.sin(Y) / (X * Y)

    # Plot the heat map of g
    plt.subplot(121)
    plt.pcolormesh(X, Y, g, cmap="magma", shading="auto")
    plt.colorbar()
    plt.xlim(-2 * np.pi, 2 * np.pi)
    plt.ylim(-2 * np.pi, 2 * np.pi)
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.title("Heat Map of g")

    # Plot the contour map of g with 20 level curves
    plt.subplot(122)
    plt.contour(X, Y, g, 20, cmap="coolwarm")
    plt.colorbar()
    plt.xlabel("x axis")
    plt.ylabel("y axis")
    plt.title("Contour Map of g")

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    print("We don't make mistakes, just happy little accidents.")
    # print(var_of_means(4))
    # prob1()
    # prob2()
    # prob3()
    # prob4()
    # prob5()
    # prob6()
