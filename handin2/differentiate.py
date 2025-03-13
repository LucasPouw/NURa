import numpy as np

def central_difference(func, x, h):
    return 0.5 * (func(x + h) - func(x - h)) / h


def ridder(func, x, h=0.1, d=float(2), m=5, target_error=1e-10):
    
    x = np.atleast_1d(x)
    
    df = np.zeros((m, len(x))).astype(np.float64)
    for i in range(m):
        df[i,:] = central_difference(func, x, h)
        h /= d

    lowest_error = np.inf
    for i in range(1, m):
        trial_df = df.copy()

        # print('Checking lowest error:', lowest_error)

        d = d**(2 * i)
        for j in range(m - i):
            trial_df[j,:] = (d * trial_df[j+1,:] - trial_df[j,:]) / (d - 1)
        
        # trial_error = np.sum( np.abs(trial_df[1,:] - lowest_error) ) / len(x)  # Mean error per point
        trial_error = 100  # TODO
        
        if trial_error > lowest_error:
            print('Terminated at m=', i)
            return df[0,:], df[1,:]
        else:
            df = trial_df
            lowest_error = trial_error

            if lowest_error < target_error:
                print('Target error reached at m=', i)
                return df[0,:], df[1,:]

    return df[0,:], df[1,:]


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    func = lambda x: x**2 * np.sin(x)
    df = lambda x: 2 * x * np.sin(x) + x**2 * np.cos(x)

    xx = np.linspace(0, 2*np.pi, 200)
    print(ridder(func, xx))

    # plt.figure()
    # plt.plot(xx, df(xx))
    # colors=['red', 'blue', 'black']
    # for i, h in enumerate([0.1, 0.01, 0.001]):
    #     plt.plot(xx, central_difference(func, xx, h), color=colors[i])
    # plt.plot(xx, ridder(func, xx)[0], linewidth=5, zorder=-1, color='magenta')
    # plt.show()

    plt.figure()
    colors=['red', 'blue', 'black']
    for i, h in enumerate([0.1, 0.01, 0.001]):
        plt.scatter(xx, np.log10( abs(df(xx) - central_difference(func, xx, h)) ) - np.log10(abs(df(xx))), color=colors[i], marker='.')
    plt.scatter(xx, np.log10( abs(df(xx) - ridder(func, xx)[0]) ) - np.log10(abs(df(xx))), zorder=-1, color='magenta', marker='.')
    plt.show()
