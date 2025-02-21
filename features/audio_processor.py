# eqly_assessment/features/audio_processor.py
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from typing import Dict, Tuple, List, Any



class AudioProcessor:
    def __init__(self, config: Dict):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h").to(self.device)
        self.sample_rate = config.get('sample_rate', 16000)

    def extract_features(self, audio_path: str) -> Dict[str, torch.Tensor]:
        waveform, sample_rate = torchaudio.load(audio_path)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)

        features = self._extract_speech_features(waveform)
        patterns = self._analyze_speech_patterns(features)
        emotions = self._detect_speech_emotions(features)
        clarity = self._measure_speech_clarity(features)

        return {
            'speech_features': features,
            'speech_patterns': patterns,
            'speech_emotions': emotions,
            'speech_clarity': clarity
        }

    def _extract_speech_features(self, waveform: torch.Tensor) -> torch.Tensor:
        inputs = self.processor(waveform, return_tensors="pt", padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs.logits

    def _analyze_speech_patterns(self, features: torch.Tensor) -> Dict[str, float]:
        pace = self._calculate_speaking_pace(features)
        pauses = self._analyze_pauses(features)
        emphasis = self._detect_emphasis(features)

        return {
            'speaking_pace': pace,
            'pause_patterns': pauses,
            'emphasis_points': emphasis
        }

    def _calculate_speaking_pace(self, features: torch.Tensor) -> float:
        # Calculate words per minute
        word_timestamps = self._get_word_timestamps(features)
        total_words = len(word_timestamps)
        duration = word_timestamps[-1]['end'] - word_timestamps[0]['start']
        return total_words / (duration / 60)

    def _analyze_pauses(self, features: torch.Tensor) -> Dict[str, Any]:
        word_timestamps = self._get_word_timestamps(features)
        pauses = []
        for i in range(len(word_timestamps) - 1):
            pause_duration = word_timestamps[i + 1]['start'] - word_timestamps[i]['end']
            if pause_duration > 0.2:  # Threshold for significant pause
                pauses.append({
                    'duration': pause_duration,
                    'position': i
                })
        return {
            'count': len(pauses),
            'avg_duration': np.mean([p['duration'] for p in pauses]),
            'positions': [p['position'] for p in pauses]
        }

    def _detect_emphasis(self, features: torch.Tensor) -> Dict[str, List[int]]:
        energy = torch.norm(features, dim=-1)
        peaks = self._find_peaks(energy)
        return {
            'emphasis_positions': peaks.tolist(),
            'emphasis_strength': energy[peaks].tolist()
        }

    def _detect_speech_emotions(self, features: torch.Tensor) -> Dict[str, float]:
        confidence = self._measure_confidence(features)
        engagement = self._measure_engagement(features)
        enthusiasm = self._measure_enthusiasm(features)

        return {
            'confidence': confidence,
            'engagement': engagement,
            'enthusiasm': enthusiasm
        }

    def _measure_confidence(self, features: torch.Tensor) -> float:
        # Analyze pitch stability and voice steadiness
        pitch_stability = self._analyze_pitch_stability(features)
        voice_steadiness = self._analyze_voice_steadiness(features)
        return 0.6 * pitch_stability + 0.4 * voice_steadiness

    def _measure_engagement(self, features: torch.Tensor) -> float:
        # Analyze variation in tone and energy
        tone_variation = self._analyze_tone_variation(features)
        energy_variation = self._analyze_energy_variation(features)
        return 0.5 * tone_variation + 0.5 * energy_variation

    def _measure_enthusiasm(self, features: torch.Tensor) -> float:
        # Analyze speaking pace and energy levels
        pace_score = self._analyze_pace_score(features)
        energy_score = self._analyze_energy_score(features)
        return 0.4 * pace_score + 0.6 * energy_score

    def _measure_speech_clarity(self, features: torch.Tensor) -> Dict[str, float]:
        articulation = self._measure_articulation(features)
        pronunciation = self._measure_pronunciation(features)
        volume = self._analyze_volume(features)

        return {
            'articulation': articulation,
            'pronunciation': pronunciation,
            'volume_consistency': volume
        }

    def _measure_articulation(self, features: torch.Tensor) -> float:
        # Analyze phoneme clarity and transitions
        phoneme_clarity = self._analyze_phoneme_clarity(features)
        transition_quality = self._analyze_transitions(features)
        return 0.7 * phoneme_clarity + 0.3 * transition_quality

    def _measure_pronunciation(self, features: torch.Tensor) -> float:
        # Compare against reference pronunciations
        phoneme_scores = self._compare_phonemes(features)
        consistency = self._measure_pronunciation_consistency(features)
        return 0.8 * phoneme_scores + 0.2 * consistency

    def _analyze_volume(self, features: torch.Tensor) -> float:
        # Analyze volume consistency and appropriate levels
        energy = torch.norm(features, dim=-1)
        consistency = 1.0 - torch.std(energy) / torch.mean(energy)
        return float(consistency)

    def _get_word_timestamps(self, features: torch.Tensor) -> List[Dict[str, float]]:
        # Process features to get word-level timestamps
        logits = features.argmax(dim=-1)
        timestamps = []
        current_word = {'start': 0, 'end': 0}

        for i, logit in enumerate(logits[0]):
            if logit == 0:  # Blank token
                if current_word['start'] != current_word['end']:
                    timestamps.append(current_word)
                    current_word = {'start': i + 1, 'end': i + 1}
            else:
                current_word['end'] = i

        if current_word['start'] != current_word['end']:
            timestamps.append(current_word)

        return timestamps

    def _find_peaks(self, signal: torch.Tensor, min_distance: int = 5) -> torch.Tensor:
        # Find peaks in the signal with minimum distance between peaks
        peaks = []
        last_peak = -min_distance

        for i in range(1, len(signal) - 1):
            if signal[i - 1] < signal[i] > signal[i + 1]:
                if i - last_peak >= min_distance:
                    peaks.append(i)
                    last_peak = i

        return torch.tensor(peaks)

    # Additional helper methods for various analyses
    def _analyze_pitch_stability(self, features: torch.Tensor) -> float:
        pitch = self._extract_pitch(features)
        stability = 1.0 - torch.std(pitch) / torch.mean(pitch)
        return float(stability)

    def _analyze_voice_steadiness(self, features: torch.Tensor) -> float:
        energy = torch.norm(features, dim=-1)
        steadiness = 1.0 - torch.std(energy) / torch.mean(energy)
        return float(steadiness)

    def _analyze_tone_variation(self, features: torch.Tensor) -> float:
        pitch = self._extract_pitch(features)
        variation = torch.std(pitch) / torch.mean(pitch)
        return min(float(variation) * 2, 1.0)  # Normalize to [0, 1]

    def _analyze_energy_variation(self, features: torch.Tensor) -> float:
        energy = torch.norm(features, dim=-1)
        variation = torch.std(energy) / torch.mean(energy)
        return min(float(variation) * 2, 1.0)  # Normalize to [0, 1]

    def _analyze_pace_score(self, features: torch.Tensor) -> float:
        word_timestamps = self._get_word_timestamps(features)
        words_per_minute = len(word_timestamps) / (word_timestamps[-1]['end'] / self.sample_rate / 60)
        # Normalize to [0, 1] assuming optimal pace is 150 wpm
        return min(words_per_minute / 150, 1.0)

    def _analyze_energy_score(self, features: torch.Tensor) -> float:
        energy = torch.norm(features, dim=-1)
        return float(torch.mean(energy) / torch.max(energy))

    def _analyze_phoneme_clarity(self, features: torch.Tensor) -> float:
        logits = torch.softmax(features, dim=-1)
        confidence = torch.max(logits, dim=-1)[0]
        return float(torch.mean(confidence))

    def _analyze_transitions(self, features: torch.Tensor) -> float:
        # Analyze smoothness of transitions between phonemes
        transitions = torch.diff(features, dim=1)
        smoothness = 1.0 - torch.mean(torch.abs(transitions)) / torch.max(torch.abs(transitions))
        return float(smoothness)

    def _compare_phonemes(self, features: torch.Tensor) -> float:
        # Compare predicted phonemes with reference
        logits = torch.softmax(features, dim=-1)
        max_probs = torch.max(logits, dim=-1)[0]
        return float(torch.mean(max_probs))

    def _measure_pronunciation_consistency(self, features: torch.Tensor) -> float:
        # Measure consistency in pronunciation across similar phonemes
        logits = torch.softmax(features, dim=-1)
        consistency = 1.0 - torch.std(torch.max(logits, dim=-1)[0])
        return float(consistency)

    def _extract_pitch(self, features: torch.Tensor) -> torch.Tensor:
        # Extract pitch information from features
        frequency_features = torch.fft.rfft(features, dim=1)
        magnitude = torch.abs(frequency_features)
        pitch = torch.argmax(magnitude, dim=-1).float()
        return pitch