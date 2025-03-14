import numpy as np


def central_difference(func, x, h):
    return 0.5 * (func(x + h) - func(x - h)) / h


def ridder(func, x, h=0.1, d=float(2), m=5, target_error=1e-10):
    
    x = np.atleast_1d(x)
    
    df = np.zeros((m, len(x))).astype(np.float64)
    for i in range(m):
        df[i,:] = central_difference(func, x, h)
        h /= d

    d2 = d**2
    extrap = 1
    best_error = np.inf
    for i in range(1, m):
        trial_df = df.copy()

        extrap *= d2
        for j in range(m - i):
            trial_df[j,:] = (extrap * trial_df[j+1,:] - trial_df[j,:]) / (extrap - 1)
        
        trial_error = np.max( np.abs(trial_df[0,:] - trial_df[1,:]) )
        
        if trial_error > best_error:  # New estimate is worse, return old estimate
            print('Best estimate reached at m =', i + 1)
            return df[0,:], best_error
        else:
            df = trial_df
            best_error = trial_error

            if best_error < target_error:  # New estimate is accurate enough
                print('Target error reached at m =', i + 1)
                return df[0,:], best_error
    
    print('No stopping condition reached. Finished loop at m =', i + 1)

    return df[0,:], best_error


if __name__ == '__main__':
    pass
