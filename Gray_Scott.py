import threading as th
import random
import time
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.use("Qt4agg")  # or "Qt5agg" depending on you version of Qt


plt.rcParams['figure.dpi'] = 100

'''Definition der Laufvariablen'''
t_tot = 25000
dt = 1
l = 200
t = np.arange(0, t_tot+0.01, dt)
n = len(t)

'''Arrays fuer die beiden Konzentrationen: (Zeit, y-Laenge, x-Laenge)'''
A = np.empty(shape=(l, l), dtype=float)
B = np.empty(shape=(l, l), dtype=float)
An = np.empty(shape=(l, l), dtype=float)
Bn = np.empty(shape=(l, l), dtype=float)

'''Verschobene Konzentrationsarrays fuer die Diffusion'''
A_oben = np.empty(shape=(l, l), dtype=float)
A_unten = np.empty(shape=(l, l), dtype=float)
A_links = np.empty(shape=(l, l), dtype=float)
A_rechts = np.empty(shape=(l, l), dtype=float)
B_oben = np.empty(shape=(l, l), dtype=float)
B_unten = np.empty(shape=(l, l), dtype=float)
B_links = np.empty(shape=(l, l), dtype=float)
B_rechts = np.empty(shape=(l, l), dtype=float)

A_diff_gesamt = np.empty(shape=(l, l), dtype=float)
B_diff_gesamt = np.empty(shape=(l, l), dtype=float)

f = 0.062
k = 0.061

Da = 0.2
Db = 0.1

'''
Oszillationen:
f = 0.014
k = 0.045

Punktoszillationen:
f = 0.02
k = 0.055 bis 0.058

Bobbel:
f = 0.048
k = 0.066

Bakterien-like:
f = 0.02
k = 0.06

Turing-like:
f = 0.03
k = 0.055

Würschen:
f = 0.03
k = 0.055

f = 0.037
k = 0.06

Würmer:
f = 0.078
k = 0.061

Löcher:
f = 0.039
k = 0.058

f = 0.062
k = 0.061

Chaos:
f = 0.026
k = 0.051

f = 0.034
k = 0.056
'''

h = 0

betrachtungsintervall = 10

keep_going = True

'''Currently not working:'''


def key_capture_thread():
    global keep_going
    input()
    keep_going = False


plt.ion()
# , cmap='plasma', interpolation="nearest", cmap='hsv', interpolation="lanczos"
figure = plt.imshow(A[:, :], cmap='inferno', interpolation="bilinear")
plt.figsize = (10, 10)
# plt.title("Bereit..")
plt.axis('off')
plt.clim(0, 0.3)
plt.tight_layout()
plt.show(block=False)


def draw_figure(i):
    figure.set_data(B[:, :])
    # plt.title("t=" + str(np.round((dt * i), decimals=1)) + ", Iteration " + str(i))
    plt.pause(0.001)
    return


start = time.time()

A[:, :] = 1
B[:, :] = 0

print("Start der Berechnung...")
th.Thread(target=key_capture_thread, args=(),
          name='key_capture_thread', daemon=True).start()

keep_going = True

while keep_going == True:
    for i in range(1, n):

        if i % int(random.random() * 100 + 1) == 0 and i < 200:
            # f1 = 1
            f2 = 1
            # x1 = int(random.random() * lx)
            # y1 = int(random.random() * ly)
            x2 = int(random.random() * l)
            y2 = int(random.random() * l)
            # A[x1, y1] += f1
            # A[x1, y1] -= f1
            B[x2, y2] += f2
            # B[x2, y2] -= f2

        A_oben[:, :] = np.roll(A[:, :], -1, axis=0)
        A_unten[:, :] = np.roll(A[:, :], 1, axis=0)
        A_links[:, :] = np.roll(A[:, :], -1, axis=1)
        A_rechts[:, :] = np.roll(A[:, :], 1, axis=1)

        B_oben[:, :] = np.roll(B[:, :], -1, axis=0)
        B_unten[:, :] = np.roll(B[:, :], 1, axis=0)
        B_links[:, :] = np.roll(B[:, :], -1, axis=1)
        B_rechts[:, :] = np.roll(B[:, :], 1, axis=1)

        A_diff_gesamt[:, :] = Da * (- 4 * A[:, :] + A_oben[:, :]
                                    + A_unten[:, :] + A_links[:, :] + A_rechts[:, :])
        B_diff_gesamt[:, :] = Db * (- 4 * B[:, :] + B_oben[:, :]
                                    + B_unten[:, :] + B_links[:, :] + B_rechts[:, :])

        An[:, :] = A[:, :] + (- (A[:, :] * B[:, :] ** 2)
                              + f * (1 - A[:, :])) * dt + A_diff_gesamt[:, :] * dt
        Bn[:, :] = B[:, :] + (+ (A[:, :] * B[:, :] ** 2)
                              - ((k + f) * B[:, :])) * dt + B_diff_gesamt[:, :] * dt

        A[:, :] = An[:, :]
        B[:, :] = Bn[:, :]

        end = time.time()
        if i % ((t_tot/dt)/1000) == 0:
            end = time.time()
            print("Iteration " + str(i) + " " + str(np.round((i/n)*100, decimals=1)) + "% erledigt (" + str(np.round(i / (end - start), decimals=1)) + " I/s, "
                  + str(np.round(i / (end - start)/betrachtungsintervall, decimals=1)) + " f/s) Restdauer: " + str(np.round((((t_tot/dt)-i)/(i / (end - start)))/60, decimals=1)) + " min")

        if i % betrachtungsintervall == 0:
            draw_figure(i)
            h += 1

    keep_going = False
