import numpy as np
import scipy.io
import scipy.signal
import scipy.stats
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

class SpatioTemporalAnalyzer:
    """
    Comprehensive analysis of spatiotemporal patterns in spiking neural network data.
    Designed to work with data from varying sigma_e values for comparison with eigenspectrum analysis.
    """
    
    def __init__(self, N=300000, q_e=0.8, q_p=0.1, q_s=0.05, q_v=0.05):
        self.N = N
        self.q_e = q_e
        self.q_p = q_p
        self.q_s = q_s
        self.q_v = q_v
        
        # Analysis parameters
        self.spatial_bin_size = 0.02  # Spatial bin size for correlation analysis
        self.temporal_bin_size = 1.0  # Temporal bin size (ms)
        self.max_lag = 50  # Maximum lag for correlation analysis (ms)
        self.spatial_bins = np.arange(0, 1 + self.spatial_bin_size, self.spatial_bin_size)
        
    def load_spike_data(self, data_path, sigma_e_idx):
        """Load and preprocess spike data for a given sigma_e value"""
        try:
            data = scipy.io.loadmat(f'{data_path}/spikes_{self.N}.mat')
            spikes = data['s'][:, :]
            del data
            
            # Separate populations
            populations = self._separate_populations(spikes)
            return populations
        except Exception as e:
            print(f"Error loading data for sigma_e index {sigma_e_idx}: {e}")
            return None
    
    def _separate_populations(self, spikes):
        """Separate spike data into different populations"""
        populations = {}
        
        # Excitatory population
        e_mask = spikes[1, :] < self.N * self.q_e
        if np.any(e_mask):
            populations['E'] = spikes[:, e_mask]
            # Normalize position to [0,1]
            populations['E'][1, :] = populations['E'][1, :] / (self.N * self.q_e)
        
        # PV population
        pv_mask = (spikes[1, :] >= self.N * self.q_e) & (spikes[1, :] < self.N * (self.q_e + self.q_p))
        if np.any(pv_mask):
            populations['PV'] = spikes[:, pv_mask]
            populations['PV'][1, :] = (populations['PV'][1, :] - self.N * self.q_e) / (self.N * self.q_p)
        
        # SST population
        if self.q_s > 0:
            sst_mask = (spikes[1, :] >= self.N * (self.q_e + self.q_p)) & (spikes[1, :] < self.N * (self.q_e + self.q_p + self.q_s))
            if np.any(sst_mask):
                populations['SST'] = spikes[:, sst_mask]
                populations['SST'][1, :] = (populations['SST'][1, :] - self.N * (self.q_e + self.q_p)) / (self.N * self.q_s)
        
        # VIP population
        if self.q_v > 0:
            vip_mask = spikes[1, :] >= self.N * (self.q_e + self.q_p + self.q_s)
            if np.any(vip_mask):
                populations['VIP'] = spikes[:, vip_mask]
                populations['VIP'][1, :] = (populations['VIP'][1, :] - self.N * (self.q_e + self.q_p + self.q_s)) / (self.N * self.q_v)
        
        return populations
    
    def compute_spatial_firing_rate(self, population_spikes, t_start=0, t_end=None):
        """Compute spatial firing rate profile"""
        if t_end is None:
            t_end = np.max(population_spikes[0, :])
        
        # Filter spikes in time window
        time_mask = (population_spikes[0, :] >= t_start) & (population_spikes[0, :] <= t_end)
        filtered_spikes = population_spikes[:, time_mask]
        
        if filtered_spikes.shape[1] == 0:
            return self.spatial_bins[:-1], np.zeros(len(self.spatial_bins)-1)
        
        # Compute spatial histogram
        counts, _ = np.histogram(filtered_spikes[1, :], bins=self.spatial_bins)
        rate = counts / (t_end - t_start) * 1000  # Convert to Hz
        
        return self.spatial_bins[:-1], rate
    
    def compute_temporal_firing_rate(self, population_spikes, t_start=0, t_end=None):
        """Compute temporal firing rate profile"""
        if t_end is None:
            t_end = np.max(population_spikes[0, :])
        
        time_bins = np.arange(t_start, t_end + self.temporal_bin_size, self.temporal_bin_size)
        counts, _ = np.histogram(population_spikes[0, :], bins=time_bins)
        rate = counts / (self.temporal_bin_size / 1000.0)  # Convert to Hz
        
        return time_bins[:-1], rate
    
    def compute_spatial_correlation_function(self, population_spikes, t_start=0, t_end=None):
        """Compute spatial correlation function"""
        spatial_pos, spatial_rate = self.compute_spatial_firing_rate(population_spikes, t_start, t_end)
        
        if len(spatial_rate) == 0 or np.all(spatial_rate == 0):
            return np.arange(len(spatial_pos)), np.zeros(len(spatial_pos))
        
        # Compute autocorrelation
        correlation = np.correlate(spatial_rate - np.mean(spatial_rate), 
                                 spatial_rate - np.mean(spatial_rate), mode='full')
        correlation = correlation / np.max(correlation)
        
        # Center the correlation function
        center_idx = len(correlation) // 2
        lags = np.arange(-center_idx, center_idx + 1) * self.spatial_bin_size
        
        return lags, correlation
    
    def compute_temporal_correlation_function(self, population_spikes, t_start=0, t_end=None):
        """Compute temporal autocorrelation function"""
        time_bins, temporal_rate = self.compute_temporal_firing_rate(population_spikes, t_start, t_end)
        
        if len(temporal_rate) == 0 or np.all(temporal_rate == 0):
            return np.arange(min(len(temporal_rate), 2*self.max_lag)), np.zeros(min(len(temporal_rate), 2*self.max_lag))
        
        # Compute autocorrelation up to max_lag
        max_lag_bins = min(len(temporal_rate), self.max_lag)
        correlation = np.correlate(temporal_rate - np.mean(temporal_rate),
                                 temporal_rate - np.mean(temporal_rate), mode='full')
        correlation = correlation / np.max(correlation)
        
        # Extract relevant lags
        center_idx = len(correlation) // 2
        start_idx = max(0, center_idx - max_lag_bins)
        end_idx = min(len(correlation), center_idx + max_lag_bins + 1)
        
        lags = np.arange(start_idx - center_idx, end_idx - center_idx) * self.temporal_bin_size
        
        return lags, correlation[start_idx:end_idx]
    
    def compute_power_spectrum(self, population_spikes, t_start=0, t_end=None):
        """Compute power spectrum of temporal firing rate"""
        time_bins, temporal_rate = self.compute_temporal_firing_rate(population_spikes, t_start, t_end)
        
        if len(temporal_rate) == 0:
            return np.array([]), np.array([])
        
        # Remove DC component
        temporal_rate = temporal_rate - np.mean(temporal_rate)
        
        # Compute FFT
        fft_result = fft(temporal_rate)
        frequencies = fftfreq(len(temporal_rate), self.temporal_bin_size / 1000.0)  # Convert to Hz
        
        # Take only positive frequencies
        positive_mask = frequencies > 0
        frequencies = frequencies[positive_mask]
        power = np.abs(fft_result[positive_mask])**2
        
        return frequencies, power
    
    def compute_spatial_correlation_length(self, population_spikes, t_start=0, t_end=None):
        """Compute spatial correlation length by fitting exponential decay"""
        lags, correlation = self.compute_spatial_correlation_function(population_spikes, t_start, t_end)
        
        if len(correlation) == 0 or np.all(correlation <= 0):
            return 0.0
        
        # Find positive lags only
        positive_mask = lags >= 0
        lags_pos = lags[positive_mask]
        corr_pos = correlation[positive_mask]
        
        # Fit exponential decay: A * exp(-lag/xi)
        try:
            def exp_decay(x, A, xi):
                return A * np.exp(-x / xi)
            
            # Only fit to reasonable range
            max_fit_idx = min(len(lags_pos), int(0.5 / self.spatial_bin_size))
            popt, _ = curve_fit(exp_decay, lags_pos[:max_fit_idx], corr_pos[:max_fit_idx], 
                               p0=[1.0, 0.1], bounds=([0, 0.001], [10, 1.0]))
            return popt[1]  # Return correlation length xi
        except:
            return 0.0
    
    def compute_dominant_frequency(self, population_spikes, t_start=0, t_end=None):
        """Compute dominant frequency from power spectrum"""
        frequencies, power = self.compute_power_spectrum(population_spikes, t_start, t_end)
        
        if len(power) == 0:
            return 0.0
        
        # Find peak frequency (excluding DC)
        peak_idx = np.argmax(power)
        return frequencies[peak_idx]
    
    def compute_pattern_measures(self, populations, t_start=0, t_end=None):
        """Compute comprehensive pattern measures for all populations"""
        measures = {}
        
        for pop_name, pop_spikes in populations.items():
            if pop_spikes.shape[1] == 0:
                continue
                
            measures[pop_name] = {
                'spatial_correlation_length': self.compute_spatial_correlation_length(pop_spikes, t_start, t_end),
                'dominant_frequency': self.compute_dominant_frequency(pop_spikes, t_start, t_end),
                'temporal_variance': self.compute_temporal_variance(pop_spikes, t_start, t_end),
                'spatial_variance': self.compute_spatial_variance(pop_spikes, t_start, t_end),
                'peak_spatial_frequency': self.compute_peak_spatial_frequency(pop_spikes, t_start, t_end)
            }
        
        return measures
    
    def compute_temporal_variance(self, population_spikes, t_start=0, t_end=None):
        """Compute variance of temporal firing rate"""
        time_bins, temporal_rate = self.compute_temporal_firing_rate(population_spikes, t_start, t_end)
        return np.var(temporal_rate) if len(temporal_rate) > 0 else 0.0
    
    def compute_spatial_variance(self, population_spikes, t_start=0, t_end=None):
        """Compute variance of spatial firing rate"""
        spatial_pos, spatial_rate = self.compute_spatial_firing_rate(population_spikes, t_start, t_end)
        return np.var(spatial_rate) if len(spatial_rate) > 0 else 0.0
    
    def compute_peak_spatial_frequency(self, population_spikes, t_start=0, t_end=None):
        """Compute peak spatial frequency from spatial profile"""
        spatial_pos, spatial_rate = self.compute_spatial_firing_rate(population_spikes, t_start, t_end)
        
        if len(spatial_rate) == 0:
            return 0.0
        
        # Remove DC component
        spatial_rate = spatial_rate - np.mean(spatial_rate)
        
        # Compute spatial FFT
        fft_result = fft(spatial_rate)
        frequencies = fftfreq(len(spatial_rate), self.spatial_bin_size)
        
        # Take only positive frequencies
        positive_mask = frequencies > 0
        if not np.any(positive_mask):
            return 0.0
            
        frequencies = frequencies[positive_mask]
        power = np.abs(fft_result[positive_mask])**2
        
        # Find peak frequency
        peak_idx = np.argmax(power)
        return frequencies[peak_idx]

def analyze_sigma_e_sweep(data_base_path, sigma_e_range, param_name='paramR1i', 
                         t_start=0, t_end=200, populations=['E', 'PV', 'SST', 'VIP'],
                         N=300000, q_e=0.8, q_p=0.1, q_s=0.05, q_v=0.05):
    """
    Analyze spatiotemporal patterns across different sigma_e values
    
    Parameters:
    -----------
    data_base_path : str
        Base path to spike data files
    sigma_e_range : range or list
        Range of sigma_e indices to analyze
    param_name : str
        Parameter set name (e.g., 'paramR1i', 'paramR1i2')
    t_start, t_end : float
        Time window for analysis (ms)
    populations : list
        List of population names to analyze
    N : int
        Total number of neurons (default: 300000)
    q_e, q_p, q_s, q_v : float
        Population fractions (default: 0.8, 0.1, 0.05, 0.05)
        Note: q_e + q_p + q_s + q_v should equal 1.0
    """
    
    # Initialize analyzer with correct population proportions
    analyzer = SpatioTemporalAnalyzer(N=N, q_e=q_e, q_p=q_p, q_s=q_s, q_v=q_v)
    
    # Storage for results
    results = {
        'sigma_e_indices': [],
        'sigma_e_values': [],  # Will be filled with actual sigma_e values if available
        'pattern_measures': {pop: {} for pop in populations},
        'correlation_functions': {pop: {} for pop in populations},
        'power_spectra': {pop: {} for pop in populations}
    }
    
    for pop in populations:
        results['pattern_measures'][pop] = {
            'spatial_correlation_length': [],
            'dominant_frequency': [],
            'temporal_variance': [],
            'spatial_variance': [],
            'peak_spatial_frequency': []
        }
        results['correlation_functions'][pop] = {
            'spatial_lags': [],
            'spatial_correlations': [],
            'temporal_lags': [],
            'temporal_correlations': []
        }
        results['power_spectra'][pop] = {
            'frequencies': [],
            'power': []
        }
    
    print("Analyzing spatiotemporal patterns for varying sigma_e...")
    print(f"Parameter set: {param_name}")
    print(f"Time window: {t_start} - {t_end} ms")
    print(f"Populations: {populations}")
    print("-" * 50)
    
    for sigma_e_idx in sigma_e_range:
        print(f"Processing sigma_e index {sigma_e_idx}...", end='')
        
        # Construct data path
        data_path = f'{data_base_path}/batch/{param_name}_sE_0p{sigma_e_idx:02d}'
        
        # Load data
        pop_data = analyzer.load_spike_data(data_path, sigma_e_idx)
        
        if pop_data is None:
            print(" [FAILED]")
            continue
        
        results['sigma_e_indices'].append(sigma_e_idx)
        results['sigma_e_values'].append(sigma_e_idx * 0.01)  # Convert index to actual value
        
        # Analyze each population
        for pop_name in populations:
            if pop_name not in pop_data:
                print(f" [No {pop_name} data]", end='')
                continue
            
            pop_spikes = pop_data[pop_name]
            
            # Compute pattern measures
            measures = analyzer.compute_pattern_measures({pop_name: pop_spikes}, t_start, t_end)
            
            if pop_name in measures:
                for measure_name, value in measures[pop_name].items():
                    results['pattern_measures'][pop_name][measure_name].append(value)
            
            # Compute correlation functions
            spatial_lags, spatial_corr = analyzer.compute_spatial_correlation_function(pop_spikes, t_start, t_end)
            temporal_lags, temporal_corr = analyzer.compute_temporal_correlation_function(pop_spikes, t_start, t_end)
            
            results['correlation_functions'][pop_name]['spatial_lags'].append(spatial_lags)
            results['correlation_functions'][pop_name]['spatial_correlations'].append(spatial_corr)
            results['correlation_functions'][pop_name]['temporal_lags'].append(temporal_lags)
            results['correlation_functions'][pop_name]['temporal_correlations'].append(temporal_corr)
            
            # Compute power spectrum
            frequencies, power = analyzer.compute_power_spectrum(pop_spikes, t_start, t_end)
            results['power_spectra'][pop_name]['frequencies'].append(frequencies)
            results['power_spectra'][pop_name]['power'].append(power)
        
        print(" [DONE]")
    
    print(f"\nCompleted analysis for {len(results['sigma_e_indices'])} sigma_e values")
    return results

def analyze_sigma_e_sweep_multiseed(data_base_path, sigma_e_range, param_name='paramR1i', 
                                   seed_range=range(1, 21), t_start=0, t_end=200, 
                                   populations=['E', 'PV', 'SST', 'VIP'],
                                   N=300000, q_e=0.8, q_p=0.1, q_s=0.05, q_v=0.05):
    """
    Analyze temporal variance and spatial frequency across different sigma_e values and multiple seeds
    
    Parameters:
    -----------
    data_base_path : str
        Base path to spike data files
    sigma_e_range : range or list
        Range of sigma_e indices to analyze
    param_name : str
        Parameter set name (e.g., 'paramR1i', 'paramR1i2_0p2')
    seed_range : range or list
        Range of seed values to analyze (default: 1-20)
    t_start, t_end : float
        Time window for analysis (ms)
    populations : list
        List of population names to analyze
    N : int
        Total number of neurons (default: 300000)
    q_e, q_p, q_s, q_v : float
        Population fractions (default: 0.8, 0.1, 0.05, 0.05)
    
    Returns:
    --------
    dict : Results containing temporal variance and spatial frequency statistics across seeds
        - sigma_e_indices: sigma_e indices analyzed
        - sigma_e_values: actual sigma_e values
        - temporal_variance_stats: dict with mean, std, sem for each population
        - spatial_frequency_stats: dict with mean, std, sem for each population
    """
    
    # Initialize analyzer with correct population proportions
    analyzer = SpatioTemporalAnalyzer(N=N, q_e=q_e, q_p=q_p, q_s=q_s, q_v=q_v)
    
    # Storage for results
    results = {
        'sigma_e_indices': [],
        'sigma_e_values': [],
        'temporal_variance_stats': {pop: {'mean': [], 'std': [], 'sem': []} for pop in populations},
        'spatial_frequency_stats': {pop: {'mean': [], 'std': [], 'sem': []} for pop in populations}
    }
    
    print("Analyzing temporal variance and spatial frequency across multiple seeds...")
    print(f"Parameter set: {param_name}")
    print(f"Time window: {t_start} - {t_end} ms")
    print(f"Populations: {populations}")
    print(f"Seeds: {list(seed_range)}")
    print("-" * 50)
    
    for sigma_e_idx in sigma_e_range:
        print(f"Processing sigma_e index {sigma_e_idx}...", end='')
        
        # Storage for measures across seeds
        seed_temporal_variance = {pop: [] for pop in populations}
        seed_spatial_frequency = {pop: [] for pop in populations}
        
        # Analyze each seed
        valid_seeds = 0
        for seed in seed_range:
            # Construct data path
            data_path = f'{data_base_path}/batch/{param_name}_sE_0p{sigma_e_idx:02d}'
            
            # Load spike data for specific seed
            try:
                data = scipy.io.loadmat(f'{data_path}/spikes_{N}_seed_{seed}.mat')
                spikes = data['s'][:, :]
                del data
                
                # Separate populations
                pop_data = analyzer._separate_populations(spikes)
                
                # Compute measures for each population
                for pop_name in populations:
                    if pop_name in pop_data and pop_data[pop_name].shape[1] > 0:
                        # Temporal variance
                        temporal_var = analyzer.compute_temporal_variance(pop_data[pop_name], t_start, t_end)
                        seed_temporal_variance[pop_name].append(temporal_var)
                        
                        # Peak spatial frequency
                        spatial_freq = analyzer.compute_peak_spatial_frequency(pop_data[pop_name], t_start, t_end)
                        seed_spatial_frequency[pop_name].append(spatial_freq)
                
                valid_seeds += 1
                
            except Exception as e:
                print(f" [Seed {seed} failed: {e}]", end='')
                continue
        
        if valid_seeds == 0:
            print(" [NO VALID SEEDS]")
            continue
        
        results['sigma_e_indices'].append(sigma_e_idx)
        results['sigma_e_values'].append(sigma_e_idx * 0.01)
        
        # Calculate statistics across seeds for each population
        for pop_name in populations:
            # Temporal variance statistics
            if len(seed_temporal_variance[pop_name]) > 0:
                variance_array = np.array(seed_temporal_variance[pop_name])
                
                mean_var = np.mean(variance_array)
                std_var = np.std(variance_array, ddof=1) if len(variance_array) > 1 else 0.0
                sem_var = std_var / np.sqrt(len(variance_array)) if len(variance_array) > 1 else 0.0
                
                results['temporal_variance_stats'][pop_name]['mean'].append(mean_var)
                results['temporal_variance_stats'][pop_name]['std'].append(std_var)
                results['temporal_variance_stats'][pop_name]['sem'].append(sem_var)
            else:
                results['temporal_variance_stats'][pop_name]['mean'].append(0.0)
                results['temporal_variance_stats'][pop_name]['std'].append(0.0)
                results['temporal_variance_stats'][pop_name]['sem'].append(0.0)
            
            # Spatial frequency statistics
            if len(seed_spatial_frequency[pop_name]) > 0:
                frequency_array = np.array(seed_spatial_frequency[pop_name])
                
                mean_freq = np.mean(frequency_array)
                std_freq = np.std(frequency_array, ddof=1) if len(frequency_array) > 1 else 0.0
                sem_freq = std_freq / np.sqrt(len(frequency_array)) if len(frequency_array) > 1 else 0.0
                
                results['spatial_frequency_stats'][pop_name]['mean'].append(mean_freq)
                results['spatial_frequency_stats'][pop_name]['std'].append(std_freq)
                results['spatial_frequency_stats'][pop_name]['sem'].append(sem_freq)
            else:
                results['spatial_frequency_stats'][pop_name]['mean'].append(0.0)
                results['spatial_frequency_stats'][pop_name]['std'].append(0.0)
                results['spatial_frequency_stats'][pop_name]['sem'].append(0.0)
        
        print(f" [DONE - {valid_seeds} seeds]")
    
    print(f"\nCompleted multiseed analysis for {len(results['sigma_e_indices'])} sigma_e values")
    return results

def plot_pattern_measures(results, save_path=None, plot_populations=None):
    """Plot pattern measures vs sigma_e"""
    
    sigma_e_vals = results['sigma_e_values']
    populations = list(results['pattern_measures'].keys())
    if plot_populations:
        populations = plot_populations
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    colors = {'E': 'black', 'PV': 'blue', 'SST': 'orange', 'VIP': 'green'}
    
    measures = ['spatial_correlation_length', 'dominant_frequency', 'temporal_variance', 
               'spatial_variance', 'peak_spatial_frequency']
    titles = ['Spatial Correlation Length', 'Dominant Frequency (Hz)', 'Temporal Variance (Hz²)',
             'Spatial Variance (Hz²)', 'Peak Spatial Frequency (cycles/unit)']
    
    for i, (measure, title) in enumerate(zip(measures, titles)):
        ax = axes[i//3, i%3]
        
        for pop in populations:
            if len(results['pattern_measures'][pop][measure]) > 0:
                ax.plot(sigma_e_vals, results['pattern_measures'][pop][measure], 
                       marker='o', label=pop, color=colors.get(pop, 'gray'))
        
        ax.set_xlabel('σₑ')
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    # Remove empty subplot
    if len(measures) < 6:
        fig.delaxes(axes[1, 2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_correlation_functions(results, sigma_e_indices_to_plot, save_path=None):
    """Plot correlation functions for selected sigma_e values"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = {'E': 'black', 'PV': 'blue', 'SST': 'orange', 'VIP': 'green'}
    
    for i, sigma_e_idx in enumerate(sigma_e_indices_to_plot):
        if sigma_e_idx not in results['sigma_e_indices']:
            continue
            
        idx = results['sigma_e_indices'].index(sigma_e_idx)
        
        # Spatial correlation
        ax = axes[i//2, i%2]
        for pop in results['correlation_functions']:
            if len(results['correlation_functions'][pop]['spatial_lags']) > idx:
                lags = results['correlation_functions'][pop]['spatial_lags'][idx]
                corr = results['correlation_functions'][pop]['spatial_correlations'][idx]
                ax.plot(lags, corr, label=pop, color=colors.get(pop, 'gray'))
        
        ax.set_xlabel('Spatial Lag')
        ax.set_ylabel('Correlation')
        ax.set_title(f'Spatial Correlation (σₑ = {sigma_e_idx * 0.01:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_power_spectra(results, sigma_e_indices_to_plot, save_path=None):
    """Plot power spectra for selected sigma_e values"""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    colors = {'E': 'black', 'PV': 'blue', 'SST': 'orange', 'VIP': 'green'}
    
    for i, sigma_e_idx in enumerate(sigma_e_indices_to_plot):
        if sigma_e_idx not in results['sigma_e_indices']:
            continue
            
        idx = results['sigma_e_indices'].index(sigma_e_idx)
        
        # Power spectrum
        ax = axes[i//2, i%2]
        for pop in results['power_spectra']:
            if len(results['power_spectra'][pop]['frequencies']) > idx:
                freqs = results['power_spectra'][pop]['frequencies'][idx]
                power = results['power_spectra'][pop]['power'][idx]
                if len(freqs) > 0:
                    ax.loglog(freqs, power, label=pop, color=colors.get(pop, 'gray'))
        
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Power')
        ax.set_title(f'Power Spectrum (σₑ = {sigma_e_idx * 0.01:.2f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def plot_temporal_variance_comparison_with_error(results_2pop, results_4pop, save_path=None):
    """
    Plot temporal variance comparison between 2-pop and 4-pop models with error bars
    
    Parameters:
    -----------
    results_2pop, results_4pop : dict
        Results from analyze_sigma_e_sweep_multiseed for 2-pop and 4-pop models
    save_path : str, optional
        Path to save the plot
    """
    
    # Extract sigma_e values
    sigma_e_values_2pop = np.array(results_2pop['sigma_e_values'])
    sigma_e_values_4pop = np.array(results_4pop['sigma_e_values'])
    
    # Extract temporal variance statistics for E population
    mean_2pop = np.array(results_2pop['temporal_variance_stats']['E']['mean'])
    sem_2pop = np.array(results_2pop['temporal_variance_stats']['E']['sem'])
    
    mean_4pop = np.array(results_4pop['temporal_variance_stats']['E']['mean'])
    sem_4pop = np.array(results_4pop['temporal_variance_stats']['E']['sem'])
    
    # Normalize by the last value for comparison
    if len(mean_2pop) > 0 and mean_2pop[-1] > 0:
        norm_mean_2pop = mean_2pop / mean_2pop[-1]
        norm_sem_2pop = sem_2pop / mean_2pop[-1]
    else:
        norm_mean_2pop = mean_2pop
        norm_sem_2pop = sem_2pop
    
    if len(mean_4pop) > 0 and mean_4pop[-1] > 0:
        norm_mean_4pop = mean_4pop / mean_4pop[-1]
        norm_sem_4pop = sem_4pop / mean_4pop[-1]
    else:
        norm_mean_4pop = mean_4pop
        norm_sem_4pop = sem_4pop
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot with error bars
    ax.errorbar(sigma_e_values_2pop, norm_mean_2pop, yerr=norm_sem_2pop, 
                fmt='o-', linewidth=2, markersize=6, capsize=3,
                label='2-pop (E-PV)', color='purple', alpha=0.8)
    ax.errorbar(sigma_e_values_4pop, norm_mean_4pop, yerr=norm_sem_4pop, 
                fmt='s-', linewidth=2, markersize=6, capsize=3,
                label='4-pop (E-PV-SST-VIP)', color='black', alpha=0.8)
    
    ax.set_xlabel('σₑ (Excitatory Spatial Spread)', fontsize=12)
    ax.set_ylabel('Relative Temporal Variance', fontsize=12)
    ax.set_title('Temporal Variance vs σₑ (E Population)\nAveraged across 20 seeds', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.2)
    
    # Add critical sigma_e lines
    critical_sigma_e_4pop = 0.04
    ax.axvline(critical_sigma_e_4pop, linestyle=':', color='black', alpha=0.7, label='Critical σₑ (4-pop)')
    critical_sigma_e_2pop = 0.07
    ax.axvline(critical_sigma_e_2pop, linestyle=':', color='purple', alpha=0.7, label='Critical σₑ (2-pop)')
    
    # Update legend to include critical lines
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"Temporal Variance Analysis Summary (with error bars):")
    print(f"{'='*60}")
    print(f"2-Population Model:")
    print(f"  σₑ range: {sigma_e_values_2pop[0]:.2f} - {sigma_e_values_2pop[-1]:.2f}")
    print(f"  Number of seeds analyzed: {len(results_2pop['temporal_variance_stats']['E']['mean'])}")
    print(f"  Mean temporal variance range: {np.min(mean_2pop):.2e} - {np.max(mean_2pop):.2e}")
    print(f"  Average SEM: {np.mean(sem_2pop):.2e}")
    
    print(f"\n4-Population Model:")
    print(f"  σₑ range: {sigma_e_values_4pop[0]:.2f} - {sigma_e_values_4pop[-1]:.2f}")
    print(f"  Number of seeds analyzed: {len(results_4pop['temporal_variance_stats']['E']['mean'])}")
    print(f"  Mean temporal variance range: {np.min(mean_4pop):.2e} - {np.max(mean_4pop):.2e}")
    print(f"  Average SEM: {np.mean(sem_4pop):.2e}")

def plot_spatial_frequency_comparison_with_error(results_2pop, results_4pop, save_path=None):
    """
    Plot spatial frequency comparison between 2-pop and 4-pop models with error bars
    
    Parameters:
    -----------
    results_2pop, results_4pop : dict
        Results from analyze_sigma_e_sweep_multiseed for 2-pop and 4-pop models
    save_path : str, optional
        Path to save the plot
    """
    
    # Extract sigma_e values
    sigma_e_values_2pop = np.array(results_2pop['sigma_e_values'])
    sigma_e_values_4pop = np.array(results_4pop['sigma_e_values'])
    
    # Extract spatial frequency statistics for E population
    mean_2pop = np.array(results_2pop['spatial_frequency_stats']['E']['mean'])
    sem_2pop = np.array(results_2pop['spatial_frequency_stats']['E']['sem'])
    
    mean_4pop = np.array(results_4pop['spatial_frequency_stats']['E']['mean'])
    sem_4pop = np.array(results_4pop['spatial_frequency_stats']['E']['sem'])
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Plot with error bars
    ax.errorbar(sigma_e_values_2pop, mean_2pop, yerr=sem_2pop, 
                fmt='o-', linewidth=2, markersize=6, capsize=3,
                label='2-pop (E-PV)', color='purple', alpha=0.8)
    ax.errorbar(sigma_e_values_4pop, mean_4pop, yerr=sem_4pop, 
                fmt='s-', linewidth=2, markersize=6, capsize=3,
                label='4-pop (E-PV-SST-VIP)', color='black', alpha=0.8)
    
    ax.set_xlabel('σₑ (Excitatory Spatial Spread)', fontsize=12)
    ax.set_ylabel('Peak Spatial Frequency (cycles/unit)', fontsize=12)
    ax.set_title('Peak Spatial Frequency vs σₑ (E Population)\nAveraged across 20 seeds', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 0.2)
    
    # Add critical sigma_e lines
    critical_sigma_e_4pop = 0.04
    ax.axvline(critical_sigma_e_4pop, linestyle=':', color='black', alpha=0.7, label='Critical σₑ (4-pop)')
    critical_sigma_e_2pop = 0.07
    ax.axvline(critical_sigma_e_2pop, linestyle=':', color='purple', alpha=0.7, label='Critical σₑ (2-pop)')
    
    # Update legend to include critical lines
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print(f"Spatial Frequency Analysis Summary (with error bars):")
    print(f"{'='*60}")
    print(f"2-Population Model:")
    print(f"  σₑ range: {sigma_e_values_2pop[0]:.2f} - {sigma_e_values_2pop[-1]:.2f}")
    print(f"  Mean spatial frequency range: {np.min(mean_2pop):.4f} - {np.max(mean_2pop):.4f} cycles/unit")
    print(f"  Average SEM: {np.mean(sem_2pop):.4f}")
    
    print(f"\n4-Population Model:")
    print(f"  σₑ range: {sigma_e_values_4pop[0]:.2f} - {sigma_e_values_4pop[-1]:.2f}")
    print(f"  Mean spatial frequency range: {np.min(mean_4pop):.4f} - {np.max(mean_4pop):.4f} cycles/unit")
    print(f"  Average SEM: {np.mean(sem_4pop):.4f}")

def plot_combined_measures_comparison(results_2pop, results_4pop, save_path=None):
    """
    Plot both temporal variance and spatial frequency comparisons in subplots
    
    Parameters:
    -----------
    results_2pop, results_4pop : dict
        Results from analyze_sigma_e_sweep_multiseed for 2-pop and 4-pop models
    save_path : str, optional
        Path to save the plot
    """
    
    # Extract data
    sigma_e_values_2pop = np.array(results_2pop['sigma_e_values'])
    sigma_e_values_4pop = np.array(results_4pop['sigma_e_values'])
    
    # Temporal variance data
    temp_mean_2pop = np.array(results_2pop['temporal_variance_stats']['E']['mean'])
    temp_sem_2pop = np.array(results_2pop['temporal_variance_stats']['E']['sem'])
    temp_mean_4pop = np.array(results_4pop['temporal_variance_stats']['E']['mean'])
    temp_sem_4pop = np.array(results_4pop['temporal_variance_stats']['E']['sem'])
    
    # Spatial frequency data
    spat_mean_2pop = np.array(results_2pop['spatial_frequency_stats']['E']['mean'])
    spat_sem_2pop = np.array(results_2pop['spatial_frequency_stats']['E']['sem'])
    spat_mean_4pop = np.array(results_4pop['spatial_frequency_stats']['E']['mean'])
    spat_sem_4pop = np.array(results_4pop['spatial_frequency_stats']['E']['sem'])
    
    # Normalize temporal variance
    norm_temp_mean_2pop = temp_mean_2pop / temp_mean_2pop[-1] if temp_mean_2pop[-1] > 0 else temp_mean_2pop
    norm_temp_sem_2pop = temp_sem_2pop / temp_mean_2pop[-1] if temp_mean_2pop[-1] > 0 else temp_sem_2pop
    norm_temp_mean_4pop = temp_mean_4pop / temp_mean_4pop[-1] if temp_mean_4pop[-1] > 0 else temp_mean_4pop
    norm_temp_sem_4pop = temp_sem_4pop / temp_mean_4pop[-1] if temp_mean_4pop[-1] > 0 else temp_sem_4pop
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Temporal variance
    ax1.errorbar(sigma_e_values_2pop, norm_temp_mean_2pop, yerr=norm_temp_sem_2pop, 
                fmt='o-', linewidth=2, markersize=6, capsize=3,
                label='2-pop (E-PV)', color='purple', alpha=0.8)
    ax1.errorbar(sigma_e_values_4pop, norm_temp_mean_4pop, yerr=norm_temp_sem_4pop, 
                fmt='s-', linewidth=2, markersize=6, capsize=3,
                label='4-pop (E-PV-SST-VIP)', color='black', alpha=0.8)
    
    ax1.set_xlabel('σₑ (Excitatory Spatial Spread)', fontsize=12)
    ax1.set_ylabel('Relative Temporal Variance', fontsize=12)
    ax1.set_title('Temporal Variance vs σₑ', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 0.2)
    
    # Add critical lines to temporal variance plot
    ax1.axvline(0.04, linestyle=':', color='black', alpha=0.7)
    ax1.axvline(0.07, linestyle=':', color='purple', alpha=0.7)
    
    # Plot 2: Spatial frequency
    ax2.errorbar(sigma_e_values_2pop, spat_mean_2pop, yerr=spat_sem_2pop, 
                fmt='o-', linewidth=2, markersize=6, capsize=3,
                label='2-pop (E-PV)', color='purple', alpha=0.8)
    ax2.errorbar(sigma_e_values_4pop, spat_mean_4pop, yerr=spat_sem_4pop, 
                fmt='s-', linewidth=2, markersize=6, capsize=3,
                label='4-pop (E-PV-SST-VIP)', color='black', alpha=0.8)
    
    ax2.set_xlabel('σₑ (Excitatory Spatial Spread)', fontsize=12)
    ax2.set_ylabel('Peak Spatial Frequency (cycles/unit)', fontsize=12)
    ax2.set_title('Peak Spatial Frequency vs σₑ', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 0.2)
    
    # Add critical lines to spatial frequency plot
    ax2.axvline(0.04, linestyle=':', color='black', alpha=0.7)
    ax2.axvline(0.07, linestyle=':', color='purple', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_results(results, filename):
    """Save results to numpy file"""
    np.save(filename, results)
    print(f"Results saved to {filename}")

def load_results(filename):
    """Load results from numpy file"""
    return np.load(filename, allow_pickle=True).item()

