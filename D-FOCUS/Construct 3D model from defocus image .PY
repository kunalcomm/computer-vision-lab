import cv2
import matplotlib.pyplot as plt
import numpy as np


def compute_focus_measure(img, kernel_size=15):
    """Computes focus measure using Modified Laplacian (SML) for each pixel."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Compute second derivatives (Laplacian along X and Y)
    lx = cv2.Sobel(gray, cv2.CV_64F, 2, 0, ksize=3)
    ly = cv2.Sobel(gray, cv2.CV_64F, 0, 2, ksize=3)

    # Absolute sum of Laplacians gives focus measure
    sml = np.abs(lx) + np.abs(ly)

    # Smooth the focus measure across a neighborhood
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float64) / (
        kernel_size**2
    )
    focus_map = cv2.filter2D(sml, -1, kernel)
    return focus_map


def build_3d_depth_map(image_stack, focal_distances):
    """Computes a dense depth map from a stack of images taken at different focal distances."""
    num_images, h, w, _ = image_stack.shape

    # Store focus measures for each image in the stack
    focus_stack = np.zeros((num_images, h, w))

    for i in range(num_images):
        focus_stack[i] = compute_focus_measure(image_stack[i])

    # Find the index of the image with maximum focus measure for every pixel
    best_focus_index = np.argmax(focus_stack, axis=0)

    # Map indices to corresponding focal distances to construct depth map
    depth_map = focal_distances[best_focus_index]

    # Smooth depth map using a bilateral filter to remove noise while preserving edges
    depth_map_smoothed = cv2.bilateralFilter(
        depth_map.astype(np.float32), d=9, sigmaColor=0.1, sigmaSpace=5
    )

    return depth_map_smoothed


def plot_3d_surface(depth_map, texture_img):
    """Visualizes the reconstructed depth map as a 3D surface model."""
    h, w = depth_map.shape
    x = np.arange(0, w, 1)
    y = np.arange(0, h, 1)
    X, Y = np.meshgrid(x, y)

    fig = plt.figure(figsize=(12, 6))

    # Plot 2D Depth Map
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(depth_map, cmap='jet')
    ax1.set_title("Reconstructed Depth Map")
    fig.colorbar(im, ax=ax1, shrink=0.6)

    # Plot 3D Mesh
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    # Subsample for smooth, fast 3D rendering
    step = 4
    ax2.plot_surface(
        X[::step, ::step],
        Y[::step, ::step],
        depth_map[::step, ::step],
        cmap='viridis',
        edgecolor='none',
    )
    ax2.set_title("3D Model / Surface Reconstruction")
    ax2.set_zlabel("Depth (Focal Distance)")
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.show()


# --- Demonstration with Synthetic Focal Stack ---
if __name__ == "__main__":
    h, w = 200, 200
    num_frames = 5
    focal_distances = np.array([10.0, 20.0, 30.0, 40.0, 50.0])

    # Create synthetic focal stack (a sphere at different blurs)
    image_stack = np.zeros((num_frames, h, w, 3), dtype=np.uint8)
    x, y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    r = np.sqrt(x**2 + y**2)
    synthetic_depth = np.clip(1.0 - r, 0, 1) * 40 + 10  # Range: 10 to 50

    for i, f in enumerate(focal_distances):
        # Blur increases as object distance deviates from focal distance `f`
        blur_amount = np.abs(synthetic_depth - f)
        base_img = (synthetic_depth * 5).astype(np.uint8)

        # Apply spatially varying blur to simulate defocus
        frame = cv2.GaussianBlur(
            cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR), (15, 15), 0
        )
        image_stack[i] = frame

    # Run Depth from Focus
    reconstructed_depth = build_3d_depth_map(image_stack, focal_distances)

    # Display 3D Surface Model
    plot_3d_surface(reconstructed_depth, image_stack[0])