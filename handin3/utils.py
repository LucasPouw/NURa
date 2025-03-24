import numpy as np


def readfile(filename):
        f = open(filename, 'r')
        data = f.readlines()[3:]  # Skip first 3 lines 
        nhalo = int(data[0])  # Number of halos
        radius = []
        
        for line in data[1:]:
            if line[:-1]!='#':
                radius.append(float(line.split()[0]))
        
        radius = np.array(radius, dtype=float)    
        f.close()
        return radius, nhalo  # Return the virial radius for all the satellites in the file, and the number of halos


if __name__ == '__main__':
     pass