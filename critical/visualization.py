
import logging
import numpy as np
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import axes3d, art3d

logger = logging.getLogger(__name__)


def quadraticBezierPath(posA, posB, nstep=8, controlDist=0.1, controlDim=[0, 1, 2]):
    segments = []

    vecP1 = posB - posA
    if np.linalg.norm(vecP1) == 0.0:
        raise Exception('Same start and end vertices!')

    vecP2 = np.array([0, 0, 0])
    vecP2[controlDim[2]] = 1.0
    vecC = np.cross(vecP1, vecP2)
    if np.linalg.norm(vecC) == 0.0:
        # The vectors are colinear so change the reference plane
        vecP2 = np.array([0, 0, 0])
        vecP2[controlDim[1]] = 1.0
    vecC = np.cross(vecP1, vecP2)
    vecC = vecC / np.linalg.norm(vecC) * controlDist

    posD = posA + (posB - posA) / 2.0
    posC = posD + vecC

    for t in np.linspace(0, 1.0, nstep):
        pos = posA * (1 - t) ** 2 + posB * t ** 2 + posC * 2 * t * (1 - t)
        segments.append(pos)
    return segments


def draw3D(G, S, showAxisLabels=True):
    logger.debug('Drawing 3D microcircuit overview...')

    fig = plt.figure(facecolor='white', figsize=(6, 5))
    ax = axes3d.Axes3D(fig)

    # Calculate mean minimum distance between neurons
    positions = np.stack([G[:].x, G[:].y, G[:].z], axis=-1)
    D = np.linalg.norm(positions - positions[:, np.newaxis], ord=2, axis=-1)
    D[D == 0.0] = np.inf
    neuronSpacing = np.mean(np.min(D, axis=-1))
    logger.debug('Estimated mean neural spacing is %f m' % (neuronSpacing))

    # Draw each neuron
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'o')

    # Draw each synaptic connections
    for i in range(len(G)):
        for j in range(len(G)):
            posA = np.array([G.x[i], G.y[i], G.z[i]])
            posB = np.array([G.x[j], G.y[j], G.z[j]])
            w = S.w[i, j]

            if len(w) > 0:
                # Generate a synaptic link between the neurons with a Bezier curve
                curvePoints = quadraticBezierPath(posA, posB, nstep=8, controlDist=float(0.5 * neuronSpacing))
                segments = [(curvePoints[s], curvePoints[s + 1]) for s in range(len(curvePoints) - 1)]

                # Use different colors for intracolumnar, intercolumnar and inhibitory connections
                if np.array_equal(G.mmidx[i], G.mmidx[j]):
                    # Same minicolumn
                    if G.ntype[i] > 0:
                        # Excitatory
                        color = 'gray'
                    else:
                        # Inhibitory
                        color = 'dodgerblue'
                else:
                    # Different minicolumn
                    if G.ntype[i] > 0:
                        # Excitatory
                        color = 'firebrick'
                    else:
                        # Inhibitory
                        color = 'yellowgreen'

                lines = art3d.Line3DCollection(segments, color=color)
                ax.add_collection3d(lines)

    # Disable any grid/pane/axis
    ax.grid(False)
    for a in (ax.w_xaxis, ax.w_yaxis, ax.w_zaxis):
        for t in a.get_ticklines() + a.get_ticklabels():
            t.set_visible(False)
        a.line.set_visible(False)
        a.pane.set_visible(False)

    if showAxisLabels:
        ax.set_xlabel('Length')
        ax.set_ylabel('Width')
        ax.set_zlabel('Height')

    return fig
