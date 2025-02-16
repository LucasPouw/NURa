#%%

import numpy as np

class Interpolator():

    def __init__(self,
                 points):
        
        """Input points must be (2, N) array"""

        self.points = points
        self.points_x = points[0, :]  # TODO: Assert that the points are strictly increasing by checking and making decreasing into increasing by * -1
        self.points_y = points[1, :]

        self.min_x = min(self.points_x)
        self.max_x = max(self.points_x)
        

    def bisection(self, x, M):
        """Find the closest point on either side of x. This implementation only works for an INCREASING set of data points."""

        max_idx = len(self.points_x) - 1
        left, right = 0, max_idx  # Starting indeces are the first and last point

        while right - left > 1:

            mid = (left + right) // 2
            
            if x < self.points_x[mid]:  # x is in left half-interval
                right = mid
            else:
                left = mid

        # Catch cases if not enough points on the left of x by setting 0 as the lowest possible index
        lowest_idx = max(0, left - M // 2 + 1)
        # Also catch the case if not enough points on the right of x by making sure the highest idx does not exceed the max idx
        lowest_idx = min(lowest_idx, max_idx - M + 1)  # highest_idx = lowest_idx + M - 1, so max_idx - M + 1 gives the criterion on lowest_idx

        return lowest_idx, left, right
    

    def linear(self, array):

        assert len(self.points_x) >= 2, f'Linear interpolation requires at least 2 points. Got {len(self.points_x)}'

        interpolated = np.zeros_like(array)
        for i, x in enumerate(array):

            assert self.min_x <= x <= self.max_x, 'Query outside of data range. We do not allow extrapolation.'

            lowest_idx, _, _ = self.bisection(x, M=2)
            highest_idx = lowest_idx + 1

            x_low, y_low = self.points[:, lowest_idx]
            x_high, y_high = self.points[:, highest_idx]

            interpolated[i] = (y_high - y_low) * (x - x_low) / (x_high - x_low) + y_low

        return interpolated
    
    
    def polynomial(self, array, M):

        """ Polynomial interpolation using Neville's algorithm """

        assert len(self.points_x) >= M, f'Interpolation of order {M - 1} requires at least {M} points. Found {len(self.points_x)}'

        interpolated = np.zeros_like(array)
        errors = np.zeros_like(array)
        for n, x in enumerate(array):

            lowest_idx, left, right = self.bisection(x, M)

            distance_right = self.points_x[right] - x
            distance_left = x - self.points_x[left]
            assert distance_right >= 0, 'Query outside of data range. We do not allow extrapolation.'
            assert distance_left >= 0, 'Query outside of data range. We do not allow extrapolation.'

            polynomial = self.points_y[lowest_idx:lowest_idx + M].copy()

            if M == 1:
                if distance_left > distance_right:
                    closest_idx = right
                else:
                    closest_idx = left
                interpolated[n] = polynomial[closest_idx - lowest_idx]
                continue

            valid_x = self.points_x[lowest_idx:lowest_idx + M]
            # print('interval', valid_x)
            for k in range(1, M):
                for i in range(M - k):
                    j = i + k
                    # print(f'Interval from {i} to {j}')

                    xi = valid_x[i]
                    xj = valid_x[j]

                    F = polynomial[i]
                    G = polynomial[i + 1]

                    polynomial[i] = ((xj - x) * F + (x - xi) * G) / (xj - xi)

            interpolated[n] = polynomial[0]
            errors[n] = polynomial[1]

        return interpolated, errors
    
if __name__ == '__main__':

    import matplotlib.pyplot as plt
    from matplotlib.image import imread
    from scipy.signal import square

    image = imread("M42_128.jpg")
    # # first_row_pixvals = image[0,:]
    # first_row_pixvals = np.arange(image.shape[0])
    # first_row_pixidx = np.arange(image.shape[0])
    # data = np.array([first_row_pixidx, first_row_pixvals])

    xsin = np.linspace(0, 4 * np.pi, 10)
    ysin = np.sin(xsin)
    data = np.array([xsin, ysin])

    Interp = Interpolator(data)

    # xinterp = np.linspace(0, image.shape[0] - 1, 201)
    xinterp = np.linspace(0, 4 * np.pi, 100)
    # xinterp = np.array([3.4])
    yinterp = Interp.linear(xinterp)

    # print(yinterp)

    yinterp_poly, errors = Interp.polynomial(xinterp, M=5)
    print(yinterp_poly)

    plt.figure(figsize=(8,6))
    plt.scatter(data[0,:], data[1,:], marker='x', color='black', s=75, zorder=10, label='Data')
    plt.plot(xinterp, np.sin(xinterp), color='red', label='Analytical', linewidth=5, zorder=-1)
    plt.plot(xinterp, yinterp, color='blue', label='Linear', linewidth=2)
    plt.plot(xinterp, yinterp_poly, color='green', zorder=3, label='Poly', linewidth=2)
    plt.legend()
    plt.show()

    


# %%
