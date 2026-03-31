import numpy as np

class FourierFeatureExtractor:
    def __init__(self, top_k=3):
        self.top_k = top_k

    def get_dominant_features(self, signal):
        """
        Decomposes the signal to find the strongest underlying frequencies.
        """
        # Step 1: Remove the mean (detrending) so the 0Hz component doesn't dominate
        standardized_signal = signal - np.mean(signal)
        
        # Step 2: Apply Real Fast Fourier Transform
        fft_vals = np.fft.rfft(standardized_signal)
        fft_freq = np.fft.rfftfreq(len(signal))
        
        # Step 3: Get magnitudes (amplitudes)
        amplitudes = np.abs(fft_vals)
        
        # Step 4: Find the indices of the 'top_k' highest amplitudes
        # We skip index 0 because that's the constant/mean component
        indices = np.argsort(amplitudes[1:])[-self.top_k:] + 1
        
        return {
            "amplitudes": amplitudes[indices].tolist(),
            "frequencies": fft_freq[indices].tolist(),
            "periods": [1/f if f != 0 else 0 for f in fft_freq[indices]]
        }


    def extract_feature_vector(self, window):
            """
            Converts the Fourier analysis into a flat list of numbers 
            that a Machine Learning model can process.
            """
            feats = self.get_dominant_features(window)
            vector = []
            # We pair [Amp1, Freq1, Amp2, Freq2...]
            for a, f in zip(feats['amplitudes'], feats['frequencies']):
                vector.extend([a, f])
            return vector