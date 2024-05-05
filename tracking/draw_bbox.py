import cv2
import matplotlib.pyplot as plt


def get_width(x1, x2, **kwargs):
    """
    Compute the width from bounding box corners.

    Args:
    x1 (float): x-coordinate of the left upper corner.
    x2 (float): x-coordinate of the right bottom corner.

    Returns:
    int: Returns an int containing the width.
    """
    width = x2 - x1
    return width


def get_height(y1, y2, **kwargs):
    """
    Compute the width and height from bounding box corners.

    Args:
    y1 (float): y-coordinate of the left upper corner.
    y2 (float): y-coordinate of the right bottom corner.

    Returns:
    int: Returns an int containing the height.
    """
    height = y2 - y1
    return height


def draw_bounding_boxes(tracklet, image_path, original=None):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )  # Convert from BGR to RGB for matplotlib

    if image is None:
        print(f"Image not found: {image_path}")
        return

    fig, ax = plt.subplots()
    ax.imshow(image)
    estimate_rect = plt.Rectangle(
        (tracklet.x1 * image.shape[1], tracklet.y1 * image.shape[0]),
        tracklet.get_width() * image.shape[1],
        tracklet.get_height() * image.shape[0],
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    if original:
        original_rect = plt.Rectangle(
            (original["x1"] * image.shape[1], original["y1"] * image.shape[0]),
            get_width(**original) * image.shape[1],
            get_height(**original) * image.shape[0],
            linewidth=1,
            edgecolor="g",
            facecolor="none",
        )
    ax.add_patch(estimate_rect)
    ax.add_patch(original_rect)

    # Draw trajectory
    if len(tracklet.positions) > 1:
        x_vals = [pos[0] * image.shape[1] for pos in tracklet.positions]
        y_vals = [pos[1] * image.shape[0] for pos in tracklet.positions]
        ax.plot(x_vals, y_vals, marker="o", markersize=2, linestyle="-", color="yellow")

    ax.set_title(f"Track ID: {tracklet.track_id}")
    ax.axis("off")
    plt.show()


if __name__ == "__main__":
    # sample usage
    draw_bounding_boxes(tracklet, f"{base_path}/images/{frame['image']}", frame['bbox_corners'])


