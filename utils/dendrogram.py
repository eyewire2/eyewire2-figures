import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import variation


# ============================================================================
# Publication-Quality Dendrogram from Leiden Clusters
# Following the method from single-cell papers (e.g., Shekhar et al.)
# ============================================================================

class ClusterDendrogram:
    """
    Build hierarchical dendrogram from pre-computed cluster assignments.

    This method:
    1. Computes average feature vectors per cluster
    2. Filters features by expression level and variability
    3. Hierarchically clusters the cluster centroids
    4. Provides bootstrap confidence (if using pvclust in R) or dendrogram
    """

    def __init__(self, data, cluster_labels, feature_names=None):
        """
        Parameters:
        -----------
        data : array-like, shape (n_samples, n_features)
            Data matrix (samples × features)
        cluster_labels : array-like, shape (n_samples,)
            Leiden (or any) cluster assignments for each sample
        feature_names : list, optional
            Names of features
        """
        self.data = np.array(data)
        self.cluster_labels = np.array(cluster_labels)
        self.n_samples, self.n_features = self.data.shape

        if feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(self.n_features)]
        else:
            self.feature_names = feature_names

        self.clusters = np.unique(cluster_labels)
        self.n_clusters = len(self.clusters)

    def compute_cluster_averages(self, log_transform=False, pseudocount=1):
        """
        Compute average feature vector for each cluster.

        Parameters:
        -----------
        log_transform : bool
            Whether to log-transform averages (log(x + pseudocount))
        pseudocount : float
            Pseudocount to add before log transformation

        Returns:
        --------
        avg_features : array, shape (n_clusters, n_features)
            Average feature matrix (clusters × features)
        """
        avg_features = np.zeros((self.n_clusters, self.n_features))

        for i, cluster in enumerate(self.clusters):
            cluster_mask = self.cluster_labels == cluster
            cluster_data = self.data[cluster_mask, :]
            avg_features[i, :] = cluster_data.mean(axis=0)

        # Log-transform if requested
        if log_transform:
            avg_features = np.log(avg_features + pseudocount)

        self.avg_features = avg_features
        return avg_features

    def filter_features(self, min_value=-np.inf, min_cv=-np.inf, verbose=True):
        """
        Filter features by average value and variability.

        Parameters:
        -----------
        min_value : float
            Minimum average value across clusters
        min_cv : float
            Minimum coefficient of variation (CV = std/mean)

        Returns:
        --------
        filtered_features : array
            Filtered feature matrix
        selected_features : list
            Names of selected features
        """
        if not hasattr(self, 'avg_features'):
            raise ValueError("Run compute_cluster_averages() first")

        # Filter by minimum value
        mean_values = self.avg_features.mean(axis=0)
        value_mask = mean_values >= min_value

        # Filter by coefficient of variation
        # CV = std / mean (for each feature across clusters)
        cv = variation(self.avg_features, axis=0)
        cv_mask = cv >= min_cv

        # Combine filters
        feature_mask = value_mask & cv_mask

        self.filtered_features = self.avg_features[:, feature_mask]
        self.selected_features = [self.feature_names[i]
                                  for i in range(self.n_features) if feature_mask[i]]

        if verbose:
            print(f"Feature filtering results:")
            print(f"  Total features: {self.n_features}")
            print(f"  Value filter (≥{min_value}): {value_mask.sum()} passed")
            print(f"  CV filter (≥{min_cv}): {cv_mask.sum()} passed")
            print(f"  Both filters: {feature_mask.sum()} passed")
            print(f"  Percentage retained: {100 * feature_mask.sum() / self.n_features:.1f}%")

        return self.filtered_features, self.selected_features

    def compute_distance_matrix(self, metric='euclidean'):
        """
        Compute pairwise distances between cluster centroids.

        Parameters:
        -----------
        metric : str
            Distance metric: 'euclidean', 'correlation', 'cosine', etc.

        Returns:
        --------
        dist_matrix : array, shape (n_clusters, n_clusters)
            Pairwise distance matrix
        """
        if not hasattr(self, 'filtered_features'):
            raise ValueError("Run filter_features() first")

        # Compute pairwise distances
        distances = pdist(self.filtered_features, metric=metric)
        self.dist_matrix = squareform(distances)
        self.metric = metric

        return self.dist_matrix

    def hierarchical_cluster(self, method='average'):
        """
        Perform hierarchical clustering on cluster centroids.

        Parameters:
        -----------
        method : str
            Linkage method: 'average', 'complete', 'ward', 'single'
            Paper uses 'average' (UPGMA)

        Returns:
        --------
        Z : array
            Linkage matrix from scipy
        """
        if not hasattr(self, 'dist_matrix'):
            raise ValueError("Run compute_distance_matrix() first")

        # Perform hierarchical clustering
        distances = pdist(self.filtered_features, metric=self.metric)
        self.Z = linkage(distances, method=method)
        self.method = method

        return self.Z

    def plot_dendrogram(self, figsize=(12, 6), show_distances=True,
                        color_threshold=None, title=None):
        """
        Plot the dendrogram with cluster labels.

        Parameters:
        -----------
        figsize : tuple
            Figure size
        show_distances : bool
            Whether to show y-axis (distances)
        color_threshold : float
            Threshold for coloring clusters in dendrogram
        title : str
            Custom title
        """
        if not hasattr(self, 'Z'):
            raise ValueError("Run hierarchical_cluster() first")

        fig, ax = plt.subplots(figsize=figsize)

        # Plot dendrogram
        dendro = dendrogram(
            self.Z,
            labels=self.clusters,
            ax=ax,
            leaf_font_size=11,
            color_threshold=color_threshold if color_threshold else 0,
            above_threshold_color='gray',
        )

        # Formatting
        if title is None:
            title = (f'Hierarchical Clustering of {self.n_clusters} Clusters\n'
                     f'({self.method.capitalize()} linkage, '
                     f'{self.metric.capitalize()} distance, '
                     f'{len(self.selected_features)} features)')

        ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
        ax.set_xlabel('Cluster', fontsize=12, fontweight='bold')

        if show_distances:
            ax.set_ylabel(f'{self.metric.capitalize()} Distance',
                          fontsize=12, fontweight='bold')
        else:
            ax.set_yticks([])

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.tick_params("x", rotation=90)

        plt.tight_layout()
        return fig, dendro

    def plot_heatmap_with_dendrogram(self, top_n_features=50, figsize=(14, 10)):
        """
        Plot heatmap of top variable features with dendrogram.

        Parameters:
        -----------
        top_n_features : int
            Number of most variable features to display
        figsize : tuple
            Figure size
        """
        if not hasattr(self, 'Z'):
            raise ValueError("Run hierarchical_cluster() first")

        # Select top variable features
        feature_vars = np.var(self.filtered_features, axis=0)
        top_idx = np.argsort(feature_vars)[-top_n_features:]

        plot_data = self.filtered_features[:, top_idx]
        plot_features = [self.selected_features[i] for i in top_idx]

        # Plot
        g = sns.clustermap(
            plot_data.T,  # Transpose: features × clusters
            row_cluster=False,  # Don't cluster features
            col_linkage=self.Z,  # Use our dendrogram
            cmap='RdBu_r',
            center=0,
            figsize=figsize,
            xticklabels=self.clusters,
            yticklabels=plot_features if top_n_features <= 100 else False,
            cbar_kws={'label': 'Feature Value'},
            dendrogram_ratio=0.15,
            colors_ratio=0.03,
            cbar_pos=(.02, .8, .05, .12)
        )

        g.ax_heatmap.set_xlabel('Cluster')
        g.ax_heatmap.set_ylabel('Feature')
        return g

