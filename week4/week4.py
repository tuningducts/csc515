import cv2
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


IMAGE_PATH = "Mod4CT1.jpg"
SIGMAS =  [0.8, 1.6]
KERNELS = [3, 5, 7]
FILTERS = ["gaussian", "mean", "median"]



def load_image(path) -> np.ndarray:
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"image not found at path: {path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def image_to_gray(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def apply_filter(image, kernels, sigmas=None, blur_type="gaussian") -> dict:
    filtered_images = {}
    if blur_type == "gaussian":
        filtered_images = {k:{s:cv2.GaussianBlur(image, (k,k), s) for s in sigmas} for k in kernels}
    elif blur_type == "mean":
        filtered_images = {k:cv2.blur(image, (k,k)) for k in kernels}
    elif blur_type == "median":
        filtered_images = {k:cv2.medianBlur(image, k) for k in kernels if k % 2 == 1}
    else:
        raise ValueError("Invalid blur type. Choose from 'gaussian', 'mean', 'median'.")
    return filtered_images

def show_images(filtered_images: dict, sigmas: list[float], original: np.ndarray) -> None:
    """
    filtered_images structure (per kernel):
      filtered_images[k] = {
         "mean": <img>,
         "median": <img>,
         "gaussian": {sigma: <img>}
      }
    Shows:
      Row 0: Original (spans all 4 image columns)
      Rows 1..3: kernels k in ascending order
      Cols: Mean | Median | Gaussian(σ1) | Gaussian(σ2)
      Leftmost col: row labels "k = ..."
    """
    kernels = sorted(filtered_images.keys())
    s1, s2 = sigmas

    # 4 rows total: 1 original + len(kernels) rows
    # 5 cols total: 1 label column + 4 image columns
    fig = plt.figure(figsize=(14, 7), constrained_layout=True)
    gs = gridspec.GridSpec(
        nrows=1 + len(kernels),
        ncols=5,
        figure=fig,
        width_ratios=[0.18, 1, 1, 1, 1],  # narrow label col
        height_ratios=[1, 1, 1, 1]
    )

    ax_label_top = fig.add_subplot(gs[0, 0])  
    ax_label_top.axis("off")

    ax_orig = fig.add_subplot(gs[0, 1:])  
    ax_orig.imshow(original)
    ax_orig.set_title("Original Image", fontsize=13, pad=1)
    ax_orig.set_xticks([]); ax_orig.set_yticks([])

   
    col_titles = ["Mean", "Median", f"Gaussian (σ={s1})", f"Gaussian (σ={s2})"]

   
    for r, k in enumerate(kernels, start=1):
        ax_label = fig.add_subplot(gs[r, 0])
        ax_label.axis("off")
        ax_label.text(
            0.5, 0.5, f"k = {k}",
            ha="center", va="center",
            fontsize=11, fontweight="bold"
        )
        row_imgs = [
            filtered_images[k]["mean"],
            filtered_images[k]["median"],
            filtered_images[k]["gaussian"][s1],
            filtered_images[k]["gaussian"][s2],
        ]

        for c in range(4):
            ax = fig.add_subplot(gs[r, c + 1])
            img = row_imgs[c]
            if img.ndim == 2:
                ax.imshow(img, cmap="gray")
            else:
                ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.5)
            if r == 1:
                ax.set_title(col_titles[c], fontsize=11, pad=8)

    fig.suptitle(
        "          Filter Comparison",
        fontsize=13
    )
    plt.show()

def main() -> None:
    image = load_image(IMAGE_PATH)
    gray_image = image_to_gray(image)
    mean_all_kernels = apply_filter(gray_image, KERNELS, None, "mean")
    median_all_kernels = apply_filter(gray_image, KERNELS, None, "median")
    guassian_all_kernels = apply_filter(gray_image, KERNELS, SIGMAS, "gaussian")

    filter_images = {
        k: {
            "gaussian": guassian_all_kernels[k],
            "mean":     mean_all_kernels[k],
            "median":   median_all_kernels[k]
        }
        for k in KERNELS
    }
     
    show_images(
        filtered_images=filter_images,
        sigmas=SIGMAS,
        original=image
    )
             

if __name__ == "__main__":
   main()
