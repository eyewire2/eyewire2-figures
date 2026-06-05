import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import euclidean
from matplotlib.colors import LinearSegmentedColormap, to_hex

colors = {
    'softred': '#DA3B3C',  # EXAMPLES / RGCs
    'darkblue': '#1D74B9',  # ACs
    'lightblue': '#32A9DE',  # dACs
    'orange': '#F39237',
    'darkgreen': '#0C7C59',  # BCs
    'gold': '#E9D985',  # Glia
    'pink': '#FFCAD4',  # Glia/other
    'violet': '#B68CB8',  # HCs
    'cyan': '#17CFB9',  # OFF SAC line
    'salmon': '#FFC09F',  # ON SAC line (UPDATED)
}

cellclass2color = {
    'AC': colors['darkblue'],
    'dAC': colors['lightblue'],
    'RGC': colors['softred'],
    'BC': colors['darkgreen'],
    'Glia': colors['gold'],
    'Glia/other': colors['pink'],
    'HC': colors['violet'],
    'OFF SAC': colors['cyan'],
    'Off-SAC': colors['cyan'],
    'ON SAC': colors['salmon'],
    'On-SAC': colors['salmon'],
}


def hex_to_rgb(hex_color):
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb_color):
    """Convert RGB tuple to hex color."""
    return '#{:02x}{:02x}{:02x}'.format(int(rgb_color[0]), int(rgb_color[1]), int(rgb_color[2]))


def reorder_colors_for_distinction(colors, n_neighbors=2):
    """
    Reorder colors so that neighboring colors are maximally distinguishable.

    Parameters:
    -----------
    colors : list of tuples or list of strings
        List of RGB color tuples (e.g., [(255, 0, 0), (0, 255, 0)])
        or hex color strings (e.g., ['#FF0000', '#00FF00'])
    n_neighbors : int
        Number of recent neighbors to consider when placing next color
        (default=2 means consider the 2 most recently placed colors)

    Returns:
    --------
    list
        Reordered colors in the same format as input
    """
    # Detect input format and convert to RGB if needed
    if isinstance(colors[0], str):
        input_is_hex = True
        rgb_colors = [hex_to_rgb(c) for c in colors]
    else:
        input_is_hex = False
        rgb_colors = colors

    colors_array = np.array(rgb_colors)
    n = len(colors_array)

    # Start with the first color
    ordered_indices = [0]
    remaining = set(range(1, n))

    while remaining:
        # Get the last n_neighbors colors (or fewer if we don't have that many yet)
        recent_indices = ordered_indices[-n_neighbors:]
        recent_colors = colors_array[recent_indices]

        # Find the color that maximizes minimum distance to recent neighbors
        max_min_dist = -1
        best_idx = None

        for idx in remaining:
            candidate_color = colors_array[idx]

            # Calculate distances to all recent neighbors
            distances = [euclidean(candidate_color, recent_color)
                         for recent_color in recent_colors]

            # Use minimum distance (worst case) as the score
            min_dist = min(distances)

            if min_dist > max_min_dist:
                max_min_dist = min_dist
                best_idx = idx

        ordered_indices.append(best_idx)
        remaining.remove(best_idx)

    # Return in original format
    if input_is_hex:
        return [colors[i] for i in ordered_indices]
    else:
        return [rgb_colors[i] for i in ordered_indices]


def interpolate_colors(
        n,
        colors=(
                "#da3b3c",
                "#f39237",
                "#e9d985",
                "#0c7c59",
                "#1d74b9",
                "#b68cb8",
                "#ffcad4"
        ),
        plot=True):
    # Create a colormap
    cmap = LinearSegmentedColormap.from_list("custom_rainbow", colors)

    # Sample 40 colors
    interpolated_colors = [to_hex(cmap(i / (n - 1))) for i in range(n)]

    # --- Plotting ---
    if plot:
        fig, ax = plt.subplots(figsize=(12, 2))

        # Plot the 40-color gradient
        for i, col in enumerate(interpolated_colors):
            ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=col))

        # Plot the 7 anchor colors above
        for i, col in enumerate(colors):
            ax.add_patch(plt.Rectangle((i * (n / len(colors)), 1.1), n / len(colors), 0.6, color=col))

        # Formatting
        ax.set_xlim(0, n)
        ax.set_ylim(0, 2)
        ax.axis("off")
        plt.tight_layout()
        plt.show()

    return interpolated_colors
