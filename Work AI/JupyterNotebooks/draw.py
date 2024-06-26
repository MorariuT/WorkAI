import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import random
from mpl_toolkits.mplot3d import Axes3D
traing_points = 15;
traing_points_x = [2, 1, 2, 1, 2.1, 2.1, 1.2, 2.1, 9, 8, 10, 8, 9, 8, 1];
traing_points_y = [1, 2, 2, 1, 1.3, 1.1, 1.0, 1, 10, 7, 6, 8, 9, 9, 10];

'''
for i in range(traing_points):
    x = random.randint(0, 100);
    y = random.randint(0, 100);
    traing_points_x.append(x);
    traing_points_y.append(y);
'''
plt.figure()
plt.plot(traing_points_x, traing_points_y, 'ro');
B = 0;
T = 0;

def f(x, B, T):
    return T * x + B;

def Error(b, tetha):
    E = 0;

    for i in range(traing_points):
        x = traing_points_x[i];
        y = traing_points_y[i];
        #print("--> f(x): ", f(x), " y: ", y);
        E += (f(x, b, tetha) - y)**2
        
    E /= 2 * traing_points;
    return E;
val_pos_b = np.arange(-10, 21, 0.1);
val_pos_t = np.arange(-6, 6, 0.1);
points_b = val_pos_b;
points_th = val_pos_t;
points_err = np.array([]);
'''


plotting_points = []

mn = 10000000;
Tmn = 0;
Bmn = 0;

for b in val_pos:
    for th in val_pos:
        #print("b:", b, "th:", th, "z:", Error(b, th));
        E = Error(b, th);
        points_b = np.append(points_b, b);
        points_th = np.append(points_th, th);
        points_err = np.append(points_err, E);

        plotting_points.append((b, th, E));

        if(E < mn):
            mn = E;
            Tmn = th;
            Bmn = b;

T = Tmn;
B = Bmn;

pints_function_f = np.arange(0, 10, 1);
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(traing_points_x, traing_points_y, 'ro');
ax.plot(f(pints_function_f, B, T), 'b--');
plt.show()

print("th: ", Tmn, "b: ", Bmn);
        

print(points_b);
print(points_th);
print(points_err)

'''


from matplotlib.ticker import LinearLocator 
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 


X, Y = np.meshgrid(points_b, points_th)
print(len(X), X);
print(len(Y), Y)
zs = np.array(Error(np.ravel(X), np.ravel(Y)))
Z = zs.reshape(X.shape)
print(Z);
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap=cm.jet)
ax.view_init(35, 145) 

#ax.plot_surface(X, Y, Z)
plt.xlabel("b");
plt.ylabel("th");
plt.show();
