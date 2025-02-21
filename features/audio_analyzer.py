# features/audio_analyzer.py
import numpy as np
import librosa
import torch
import sounddevice as sd
from typing import Dict, List, Any, Tuple
from scipy.signal import spectrogram
import python_speech_features


class AudioAnalyzer:
    """
    Comprehensive audio analysis system for teaching assessment.

    This analyzer evaluates multiple dimensions of speech:
    1. Prosodic Features (tone, pitch, rhythm)
    2. Voice Quality (clarity, projection)
    3. Speech Patterns (pace, pauses, emphasis)
    4. Emotional Content (engagement, confidence)

    The system maintains historical context to track patterns over time
    and provides teaching-specific metrics for assessment.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the audio analyzer with configurable parameters.

        Args:
            config: Configuration dictionary containing:
                - sample_rate: Audio sampling rate (default: 16000 Hz)
                - frame_duration: Analysis frame duration (default: 25ms)
                - window_size: Context window size (default: 2s)
                - feature_extraction: Feature extraction configuration
        """
        self.config = config or {}
        self.sample_rate = self.config.get('sample_rate', 16000)
        self.frame_duration = self.config.get('frame_duration', 0.025)
        self.window_size = self.config.get('window_size', 2.0)

        # Initialize feature extractors
        self._init_feature_extractors()

        # Historical data for pattern analysis
        self.history = {
            'pitch': [],
            'energy': [],
            'speech_rate': [],
            'emotions': []
        }

        # Teaching-specific thresholds and parameters
        self.teaching_params = {
            'optimal_speech_rate': (130, 160),  # Words per minute
            'pause_threshold': 0.5,  # Seconds
            'emphasis_energy_threshold': 1.5,  # Relative to mean
            'clarity_threshold': 0.7  # Minimum clarity score
        }

    def _init_feature_extractors(self):
        """
        Initialize various feature extraction components.
        Each component is specialized for different aspects of speech analysis.
        """
        # Pitch tracking configuration
        self.pitch_params = {
            'fmin': 50,  # Minimum frequency in Hz
            'fmax': 500,  # Maximum frequency in Hz
            'frame_length': int(self.frame_duration * self.sample_rate),
            'hop_length': int(self.frame_duration * self.sample_rate / 4)
        }

        # MFCC configuration for voice quality analysis
        self.mfcc_params = {
            'numcep': 13,
            'nfilt': 26,
            'nfft': 512,
            'winlen': self.frame_duration,
            'winstep': self.frame_duration / 2
        }

    def analyze_audio_chunk(self, audio_chunk: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive analysis on an audio chunk.

        Args:
            audio_chunk: Raw audio data (mono, floating-point)

        Returns:
            Dictionary containing multiple analysis dimensions:
            - prosody_metrics: Tone and rhythm analysis
            - voice_quality: Clarity and projection metrics
            - speech_patterns: Temporal speech characteristics
            - emotional_content: Detected emotional indicators
            - teaching_metrics: Teaching-specific assessments
        """
        # Ensure correct audio format
        audio_chunk = self._preprocess_audio(audio_chunk)

        # Extract fundamental features
        features = self._extract_base_features(audio_chunk)

        # Analyze different dimensions
        prosody = self._analyze_prosody(features)
        voice_quality = self._analyze_voice_quality(features)
        speech_patterns = self._analyze_speech_patterns(features)
        emotions = self._analyze_emotional_content(features)

        # Calculate teaching-specific metrics
        teaching_metrics = self._calculate_teaching_metrics(
            prosody, voice_quality, speech_patterns, emotions
        )

        # Update historical data
        self._update_history(features)

        return {
            'prosody_metrics': prosody,
            'voice_quality': voice_quality,
            'speech_patterns': speech_patterns,
            'emotional_content': emotions,
            'teaching_metrics': teaching_metrics
        }

    def _extract_base_features(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract fundamental audio features used in higher-level analysis.

        Features extracted:
        1. Pitch contour (F0)
        2. Spectral features (MFCCs)
        3. Energy envelope
        4. Rhythm features
        5. Voice activity detection
        """
        # Pitch extraction using robust algorithm
        pitch, pitch_confidence = librosa.piptrack(
            y=audio,
            sr=self.sample_rate,
            **self.pitch_params
        )

        # Spectral features
        mfccs = python_speech_features.mfcc(
            audio,
            samplerate=self.sample_rate,
            **self.mfcc_params
        )

        # Energy calculations
        energy = librosa.feature.rms(y=audio)[0]

        # Rhythm features
        tempo, beats = librosa.beat.beat_track(y=audio, sr=self.sample_rate)

        # Voice activity detection
        voiced_segments = self._detect_voice_activity(audio)

        return {
            'pitch': pitch,
            'pitch_confidence': pitch_confidence,
            'mfccs': mfccs,
            'energy': energy,
            'tempo': tempo,
            'beats': beats,
            'voiced_segments': voiced_segments
        }

    def _analyze_prosody(self, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze prosodic elements of speech.

        Evaluates:
        1. Pitch variation (intonation)
        2. Rhythm patterns
        3. Stress patterns
        4. Melodic contours
        """
        pitch_stats = self._calculate_pitch_statistics(features['pitch'])
        rhythm_stats = self._calculate_rhythm_statistics(
            features['energy'], features['beats']
        )

        return {
            'pitch_variety': pitch_stats['variety'],
            'pitch_range': pitch_stats['range'],
            'rhythm_regularity': rhythm_stats['regularity'],
            'emphasis_patterns': rhythm_stats['emphasis'],
            'melodic_score': self._calculate_melodic_score(features)
        }

    def _analyze_voice_quality(self, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze voice quality characteristics.

        Evaluates:
        1. Clarity of articulation
        2. Voice projection
        3. Vocal stability
        4. Resonance
        """
        spectral_features = self._analyze_spectral_features(features['mfccs'])

        return {
            'clarity': self._calculate_clarity_score(spectral_features),
            'projection': self._calculate_projection_score(features['energy']),
            'stability': self._calculate_stability_score(features),
            'resonance': self._calculate_resonance_score(spectral_features)
        }

    def _analyze_speech_patterns(self, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze temporal speech patterns.

        Evaluates:
        1. Speaking rate
        2. Pause patterns
        3. Emphasis distribution
        4. Fluency indicators
        """
        voiced_segments = features['voiced_segments']

        return {
            'speech_rate': self._calculate_speech_rate(voiced_segments),
            'pause_patterns': self._analyze_pauses(voiced_segments),
            'emphasis_distribution': self._analyze_emphasis(features['energy']),
            'fluency_score': self._calculate_fluency_score(features)
        }

    def _analyze_emotional_content(self, features: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Analyze emotional indicators in speech.

        Evaluates:
        1. Engagement level
        2. Confidence indicators
        3. Enthusiasm markers
        4. Stress indicators
        """
        # Combine multiple feature sets for emotion analysis
        emotional_indicators = self._extract_emotional_indicators(features)

        return {
            'engagement': self._calculate_engagement_score(emotional_indicators),
            'confidence': self._calculate_confidence_score(emotional_indicators),
            'enthusiasm': self._calculate_enthusiasm_score(emotional_indicators),
            'stress_level': self._calculate_stress_level(emotional_indicators)
        }

    def _calculate_teaching_metrics(self,
                                    prosody: Dict[str, float],
                                    voice_quality: Dict[str, float],
                                    speech_patterns: Dict[str, float],
                                    emotions: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate teaching-specific metrics from various analysis dimensions.

        Returns:
        - clarity_score: How clearly concepts are explained
        - engagement_score: How engaging the delivery is
        - pacing_score: How well-paced the teaching is
        - effectiveness_score: Overall teaching effectiveness
        """
        clarity_score = self._calculate_clarity_metric(voice_quality, speech_patterns)
        engagement_score = self._calculate_engagement_metric(prosody, emotions)
        pacing_score = self._calculate_pacing_metric(speech_patterns)

        # Weighted combination for overall effectiveness
        effectiveness_score = (
                clarity_score * 0.4 +
                engagement_score * 0.3 +
                pacing_score * 0.3
        )

        return {
            'clarity': clarity_score,
            'engagement': engagement_score,
            'pacing': pacing_score,
            'overall_effectiveness': effectiveness_score
        }