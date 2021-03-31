import numpy as np
import cv2


def mutual_information(Image1, Image2):
    """ Mutual information for joint histogram
...     """

    hgram, x_edges, y_edges = np.histogram2d(Image1.ravel(),
                                             Image2.ravel(), bins=20)
    pxy = hgram / float(np.sum(hgram))

    px = np.sum(pxy, axis=1)  # marginal for x over y

    py = np.sum(pxy, axis=0)  # marginal for y over x

    px_py = px[:, None] * py[None, :]  # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays

    nzs = pxy > 0  # Only non-zero pxy values contribute to the sum

    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))


if __name__ == '__main__':
    pass
