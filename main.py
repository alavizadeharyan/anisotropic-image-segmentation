from PIL import Image, ImageOps
import numpy as np
from scipy.sparse.linalg import spsolve
from scipy.sparse import csr_matrix, lil_matrix, identity
import matplotlib.pyplot as plt
from LocalFunctions import cost, initialization, update_c, calculate_g, draw_tail, data_to_image
from StiffnessMatrix import stiffness_matrix


is_supervised = (input('Do you want to supervise the algorithm? (yes/no): ') == 'yes')
is_bw = (input('Is the image black-and-white? (yes/no): ') == 'yes')
img = Image.open(input('Image: '))

if is_bw:
    data = 1/255 * np.asarray(ImageOps.grayscale(img))
else:
    data = 1/255 * np.asarray(img)
    
m = len(data)
n = len(data[0])
    
nlevel, C, A, alpha, epsilon_max = initialization(is_supervised, is_bw, None, None, None, None, None)

while True:
    if is_supervised:
        K = []
        for l in range(nlevel):
            K.append(stiffness_matrix(m,n,A[l]))
        print('Stiffness matrices are set.')
    else:
        K = stiffness_matrix(m,n,A[0])
        print('Stiffness matrix is set.')

    G = calculate_g(C[0], data, is_bw)
    for l in range(1, nlevel):
        G = np.vstack((G, calculate_g(C[l], data, is_bw)))

    U = np.zeros((nlevel,m*n))
    for k in range(m*n):
        U[np.argmin(G.T[k]),k] = 1

    V = np.zeros((nlevel,m*n))

    plt.ion()
    Y = []
    fig_cost, ax_cost = plt.subplots()
    ax_cost.set_ylabel('cost')
    fig_image, ax_image = plt.subplots()
    
    dist = 1
    epsilon = epsilon_max

    while epsilon >= 1:
        while dist > .00001:
            if is_supervised:
                for l in range(nlevel):
                    V[l] = spsolve((-1)*epsilon*epsilon*K[l] + identity(m*n) , U[l])
            else:
                V = spsolve((-1)*epsilon*epsilon*K + identity(m*n) , U.T).T

            Zeta = G + np.matmul(1/epsilon * np.diag(alpha), np.ones((nlevel,m*n)) - 2*V)

            U_old = U.copy()
            U = np.zeros((nlevel,m*n))
            for k in range(m*n):
                U[np.argmin(Zeta.T[k]),k] = 1
            dist = np.linalg.norm(U - U_old)/np.linalg.norm(U)

            if not is_supervised:
                for l in range(nlevel):
                    C[l] = update_c(data, U[l], is_bw)
                for l in range(0, nlevel):
                    G[l] = calculate_g(C[l], data, is_bw)

            Y = np.append(Y, [cost(m,n,nlevel,U,V,G,A,epsilon,alpha)])
            draw_tail(epsilon, Y, ax_cost, .5)

            if is_bw:
                new_data = np.reshape(np.matmul(C,U), (m,n))
            else:
                new_data = np.reshape(np.matmul(C.T,U).T, (m,n,3))

            data_to_image(new_data, ax_image, epsilon, .3, is_bw)

        ax_image.set_title("Final image for $\epsilon = $%.1f" % epsilon)
        plt.pause(2)
        Y = np.append(Y, [None])
        epsilon /= 2
        dist = 1

    ax_image.set_title("Finished")

    if input('continue? (yes/no): ') == 'yes':
        nlevel, C, A, alpha, epsilon_max = initialization(is_supervised, is_bw, nlevel, C, A, alpha, epsilon_max)
    else:
        break
