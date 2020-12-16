import numpy as np


def get_random_eraser(p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=1 / 0.3, v_l=0, v_h=255):

    def eraser(img):

        if img.ndim == 3:
            img_h, img_w, img_c = img.shape

        elif img.ndim == 2:
            img_h, img_w = img.shape

        if np.random.rand() > p:
            return img

        while True:
            s = np.random.uniform(sl, sh) * img_h * img_w
            r = np.random.uniform(r1, r2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))

            xe = np.random.randint(0, img_w)
            ye = np.random.randint(0, img_h)

            if xe + w <= img_w and ye + h <= img_h:
                break

        img[ye:ye + h, xe:xe + w] = np.random.uniform(v_l, v_h)

        return img

    return eraser
