"""
Alternative normalization strategies for continuous columns in CTGAN.

This module provides different normalizers for handling multimodal distributions:
- VGMNormalizer: Variational Gaussian Mixture Model (original CTGAN approach)
- KDENormalizer: Kernel Density Estimation based normalization
- DPMNormalizer: Dirichlet Process Mixture Model normalization
"""

import numpy as np
from scipy.stats import norm
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
from sklearn.mixture import BayesianGaussianMixture
from abc import ABC, abstractmethod


class BaseNormalizer(ABC):
    """Base class for all normalizers."""
    
    @abstractmethod
    def fit(self, data):
        """Fit the normalizer to data.
        
        Args:
            data: 1D numpy array of continuous values
            
        Returns:
            dict: Information about fitted normalization
        """
        pass
    
    @abstractmethod
    def transform(self, data, fit_info):
        """Transform data using fitted normalizer.
        
        Args:
            data: 1D numpy array of values to transform
            fit_info: dict returned by fit()
            
        Returns:
            tuple: (normalized_values, mode_indicators)
                normalized_values: 1D array of normalized scalars
                mode_indicators: 2D array of one-hot encoded modes
        """
        pass
    
    @abstractmethod
    def inverse_transform(self, normalized_values, mode_indicators, fit_info):
        """Inverse transform to get original scale.
        
        Args:
            normalized_values: Normalized scalar values
            mode_indicators: One-hot encoded mode indicators
            fit_info: dict from fit()
            
        Returns:
            numpy array: Original scale values
        """
        pass


class VGMNormalizer(BaseNormalizer):
    """Original VGM normalizer from CTGAN paper.
    
    Uses Variational Gaussian Mixture Model to detect modes in continuous data
    and normalize each value based on its assigned mode.
    """
    
    def __init__(self, max_clusters=10, weight_threshold=0.005):
        """Initialize VGM normalizer.
        
        Args:
            max_clusters: Maximum number of mixture components
            weight_threshold: Minimum weight for a component to be valid
        """
        self.max_clusters = max_clusters
        self.weight_threshold = weight_threshold
    
    def fit(self, data):
        """Fit Bayesian Gaussian Mixture Model.
        
        Args:
            data: 1D numpy array of continuous values
            
        Returns:
            dict: Fitted model information
        """
        gm = BayesianGaussianMixture(
            n_components=self.max_clusters,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=0.001,
            max_iter=100,
            n_init=1,
            random_state=0
        )
        
        gm.fit(data.reshape(-1, 1))
        
        # Filter components by weight threshold
        valid_component_indicator = gm.weights_ > self.weight_threshold
        num_components = valid_component_indicator.sum()
        
        if num_components == 0:
            # Fallback: use single component
            valid_component_indicator[0] = True
            num_components = 1
        
        return {
            'model': gm,
            'num_components': int(num_components),
            'means': gm.means_.reshape(-1)[valid_component_indicator],
            'stds': np.sqrt(gm.covariances_).reshape(-1)[valid_component_indicator],
            'weights': gm.weights_[valid_component_indicator]
        }
    
    def transform(self, data, fit_info):
        """Transform using VGM.
        
        Args:
            data: 1D numpy array of values to transform
            fit_info: dict from fit()
            
        Returns:
            tuple: (normalized_values, mode_indicators)
        """
        num_components = fit_info['num_components']
        means = fit_info['means']
        stds = fit_info['stds']
        
        normalized_values = np.zeros(len(data))
        mode_indicators = np.zeros((len(data), num_components))
        
        for i, value in enumerate(data):
            # Compute probabilities for each mode
            probs = []
            for mean, std in zip(means, stds):
                prob = norm.pdf(value, loc=mean, scale=std)
                probs.append(prob)
            
            probs = np.array(probs)
            if probs.sum() > 0:
                probs = probs / probs.sum()
            else:
                probs = np.ones(len(probs)) / len(probs)
            
            # Sample mode
            selected_mode = np.random.choice(num_components, p=probs)
            
            # Normalize within mode
            mean = means[selected_mode]
            std = stds[selected_mode]
            normalized_values[i] = (value - mean) / (4 * std)
            
            # One-hot encode mode
            mode_indicators[i, selected_mode] = 1.0
        
        return normalized_values, mode_indicators
    
    def inverse_transform(self, normalized_values, mode_indicators, fit_info):
        """Inverse transform from normalized space.
        
        Args:
            normalized_values: 1D array of normalized values
            mode_indicators: 2D array of one-hot mode indicators
            fit_info: dict from fit()
            
        Returns:
            numpy array: Original scale values
        """
        means = fit_info['means']
        stds = fit_info['stds']
        
        # Get selected mode for each sample
        selected_modes = np.argmax(mode_indicators, axis=1)
        
        # Inverse normalize
        original_values = np.zeros(len(normalized_values))
        for i, (norm_val, mode_idx) in enumerate(zip(normalized_values, selected_modes)):
            mean = means[mode_idx]
            std = stds[mode_idx]
            original_values[i] = norm_val * (4 * std) + mean
        
        return original_values


class KDENormalizer(BaseNormalizer):
    """Kernel Density Estimation based normalizer.
    
    Uses KDE to estimate the data distribution and detect modes via peak finding.
    More flexible than VGM for non-Gaussian mode shapes.
    """
    
    def __init__(self, bandwidth='scott', min_modes=1, max_modes=10):
        """Initialize KDE normalizer.
        
        Args:
            bandwidth: KDE bandwidth ('scott', 'silverman', or float)
            min_modes: Minimum number of modes to detect
            max_modes: Maximum number of modes to detect
        """
        self.bandwidth = bandwidth
        self.min_modes = min_modes
        self.max_modes = max_modes
    
    def fit(self, data):
        """Fit KDE and detect modes.
        
        Args:
            data: 1D numpy array of continuous values
            
        Returns:
            dict: Fitted model information
        """
        # Fit KDE
        kde = KernelDensity(bandwidth=self.bandwidth, kernel='gaussian')
        kde.fit(data.reshape(-1, 1))
        
        # Create grid for mode detection
        data_min, data_max = data.min(), data.max()
        data_range = data_max - data_min
        x_grid = np.linspace(
            data_min - 0.1 * data_range,
            data_max + 0.1 * data_range,
            1000
        )
        
        # Compute log density
        log_density = kde.score_samples(x_grid.reshape(-1, 1))
        density = np.exp(log_density)
        
        # Find peaks (modes)
        peaks, properties = find_peaks(
            density,
            prominence=np.max(density) * 0.1,  # At least 10% of max density
            distance=len(x_grid) // (self.max_modes * 2)  # Minimum distance between peaks
        )
        
        # Handle case with no peaks or too many peaks
        if len(peaks) == 0:
            # Use mean as single mode
            mode_locations = np.array([data.mean()])
        elif len(peaks) > self.max_modes:
            # Take top max_modes by prominence
            top_peaks_idx = np.argsort(properties['prominences'])[-self.max_modes:]
            mode_locations = x_grid[peaks[top_peaks_idx]]
        else:
            mode_locations = x_grid[peaks]
        
        mode_locations = np.sort(mode_locations)
        num_components = len(mode_locations)
        
        # Estimate local standard deviation around each mode
        mode_stds = []
        for mode_loc in mode_locations:
            # Find data points near this mode
            distances = np.abs(data - mode_loc)
            nearest_points = data[distances < np.percentile(distances, 30)]
            
            if len(nearest_points) > 1:
                local_std = np.std(nearest_points)
            else:
                local_std = data.std() / num_components
            
            mode_stds.append(max(local_std, 1e-6))  # Avoid zero std
        
        mode_stds = np.array(mode_stds)
        
        return {
            'kde': kde,
            'num_components': int(num_components),
            'means': mode_locations,
            'stds': mode_stds,
            'x_grid': x_grid,
            'density': density
        }
    
    def transform(self, data, fit_info):
        """Transform using KDE-detected modes.
        
        Args:
            data: 1D numpy array of values to transform
            fit_info: dict from fit()
            
        Returns:
            tuple: (normalized_values, mode_indicators)
        """
        num_components = fit_info['num_components']
        means = fit_info['means']
        stds = fit_info['stds']
        
        normalized_values = np.zeros(len(data))
        mode_indicators = np.zeros((len(data), num_components))
        
        for i, value in enumerate(data):
            # Assign to nearest mode (could also use density-based assignment)
            distances = np.abs(means - value)
            selected_mode = np.argmin(distances)
            
            # Normalize within mode
            mean = means[selected_mode]
            std = stds[selected_mode]
            normalized_values[i] = (value - mean) / (4 * std)
            
            # One-hot encode mode
            mode_indicators[i, selected_mode] = 1.0
        
        return normalized_values, mode_indicators
    
    def inverse_transform(self, normalized_values, mode_indicators, fit_info):
        """Inverse transform from normalized space.
        
        Args:
            normalized_values: 1D array of normalized values
            mode_indicators: 2D array of one-hot mode indicators
            fit_info: dict from fit()
            
        Returns:
            numpy array: Original scale values
        """
        means = fit_info['means']
        stds = fit_info['stds']
        
        selected_modes = np.argmax(mode_indicators, axis=1)
        
        original_values = np.zeros(len(normalized_values))
        for i, (norm_val, mode_idx) in enumerate(zip(normalized_values, selected_modes)):
            mean = means[mode_idx]
            std = stds[mode_idx]
            original_values[i] = norm_val * (4 * std) + mean
        
        return original_values


class DPMNormalizer(BaseNormalizer):
    """Dirichlet Process Mixture Model normalizer.
    
    Uses DPMM to automatically determine the optimal number of modes.
    More principled Bayesian approach compared to VGM.
    """
    
    def __init__(self, max_components=20, weight_concentration_prior=0.01, 
                 weight_threshold=0.005):
        """Initialize DPM normalizer.
        
        Args:
            max_components: Maximum number of mixture components
            weight_concentration_prior: DP concentration parameter (lower = fewer components)
            weight_threshold: Minimum weight for a component to be valid
        """
        self.max_components = max_components
        self.weight_concentration_prior = weight_concentration_prior
        self.weight_threshold = weight_threshold
    
    def fit(self, data):
        """Fit Dirichlet Process Mixture Model.
        
        Args:
            data: 1D numpy array of continuous values
            
        Returns:
            dict: Fitted model information
        """
        # Use Bayesian GMM with Dirichlet Process prior
        dpmm = BayesianGaussianMixture(
            n_components=self.max_components,
            weight_concentration_prior_type='dirichlet_process',
            weight_concentration_prior=self.weight_concentration_prior,
            covariance_type='full',
            max_iter=200,
            n_init=3,  # More initializations for stability
            random_state=0,
            verbose=0
        )
        
        dpmm.fit(data.reshape(-1, 1))
        
        # Filter components by weight
        active_components = dpmm.weights_ > self.weight_threshold
        num_components = active_components.sum()
        
        if num_components == 0:
            # Fallback: use component with highest weight
            active_components[np.argmax(dpmm.weights_)] = True
            num_components = 1
        
        # Extract active components
        means = dpmm.means_.reshape(-1)[active_components]
        stds = np.sqrt(dpmm.covariances_.reshape(-1))[active_components]
        weights = dpmm.weights_[active_components]
        
        # Renormalize weights
        weights = weights / weights.sum()
        
        # Sort by mean for consistency
        sort_idx = np.argsort(means)
        means = means[sort_idx]
        stds = stds[sort_idx]
        weights = weights[sort_idx]
        
        return {
            'model': dpmm,
            'num_components': int(num_components),
            'means': means,
            'stds': stds,
            'weights': weights,
            'active_components': active_components
        }
    
    def transform(self, data, fit_info):
        """Transform using DPMM.
        
        Args:
            data: 1D numpy array of values to transform
            fit_info: dict from fit()
            
        Returns:
            tuple: (normalized_values, mode_indicators)
        """
        num_components = fit_info['num_components']
        means = fit_info['means']
        stds = fit_info['stds']
        weights = fit_info['weights']
        
        normalized_values = np.zeros(len(data))
        mode_indicators = np.zeros((len(data), num_components))
        
        for i, value in enumerate(data):
            # Compute responsibility (posterior probability) for each component
            responsibilities = []
            for mean, std, weight in zip(means, stds, weights):
                likelihood = norm.pdf(value, loc=mean, scale=std)
                responsibility = weight * likelihood
                responsibilities.append(responsibility)
            
            responsibilities = np.array(responsibilities)
            if responsibilities.sum() > 0:
                responsibilities = responsibilities / responsibilities.sum()
            else:
                responsibilities = np.ones(len(responsibilities)) / len(responsibilities)
            
            # Sample mode based on responsibilities
            selected_mode = np.random.choice(num_components, p=responsibilities)
            
            # Normalize within mode
            mean = means[selected_mode]
            std = stds[selected_mode]
            normalized_values[i] = (value - mean) / (4 * std)
            
            # One-hot encode mode
            mode_indicators[i, selected_mode] = 1.0
        
        return normalized_values, mode_indicators
    
    def inverse_transform(self, normalized_values, mode_indicators, fit_info):
        """Inverse transform from normalized space.
        
        Args:
            normalized_values: 1D array of normalized values
            mode_indicators: 2D array of one-hot mode indicators
            fit_info: dict from fit()
            
        Returns:
            numpy array: Original scale values
        """
        means = fit_info['means']
        stds = fit_info['stds']
        
        selected_modes = np.argmax(mode_indicators, axis=1)
        
        original_values = np.zeros(len(normalized_values))
        for i, (norm_val, mode_idx) in enumerate(zip(normalized_values, selected_modes)):
            mean = means[mode_idx]
            std = stds[mode_idx]
            original_values[i] = norm_val * (4 * std) + mean
        
        return original_values