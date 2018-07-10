import numpy as np
import matplotlib.pyplot as plt


def main():
    fig, (ax1, ax2) = plt.subplots(n_cols=2)

    # Test 1
    c = np.array(
        [
            [  # erste Zeile
                [0, 0, 255],  # erstes Feld = blau
                [0, 255, 0],  # zweites Feld = gruen
                [255, 0, 0],  # drittes Feld = rot
            ],
            [  # zweite Zeile
                [0, 0, 255],  # blau
                [255, 0, 0],  # rot
                [0, 255, 0],  # gruen
            ],
            [  # dritte Zeile
                [255, 0, 0],  # rot
                [0, 255, 0],  # gruen
                [0, 0, 255],  # blau
            ],
        ]
    )
    ax1.imshow(c)

    # Test 2
    c = np.array(
        [
            [[255, 255, 000], [255, 127, 000], [255, 000, 000]],
            [[127, 255, 000], [127, 127, 000], [127, 000, 000]],
            [[000, 255, 000], [000, 127, 000], [000, 000, 000]],
        ]
    )
    ax2.imshow(c)

    fig.tight_layout()
    fig.savefig("test_imshow.png")


if __name__ == "__main__":
    main()
