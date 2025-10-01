# Copyright 2025 THU-BPM MarkDiffusion.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import numpy as np
from typing import Tuple, Type
from scipy.special import erf
from detection.base import BaseDetector
from ldpc import bp_decoder
from galois import FieldArray
import sys        
from Levenshtein import distance

class VideoMarkDetector(BaseDetector):
    
    def __init__(self,
                 message_sequence: np.ndarray,
                 watermark: np.ndarray,
                 num_frames: int,
                 var: int,
                 decoding_key: tuple,
                 GF: Type[FieldArray],
                 threshold: float,
                 device: torch.device):
        super().__init__(threshold, device)

        self.message_sequence = message_sequence
        self.watermark = watermark
        self.num_frames = num_frames
        self.var = var
        self.decoding_key = decoding_key
        self.GF = GF
        
    def _recover_posteriors(self, z, basis=None, variances=None):
        if variances is None:
            default_variance = 1.5
            denominators = np.sqrt(2 * default_variance * (1+default_variance)) * torch.ones_like(z)
        elif type(variances) is float:
            denominators = np.sqrt(2 * variances * (1 + variances))
        else:
            denominators = torch.sqrt(2 * variances * (1 + variances))

        if basis is None:
            return erf(z / denominators)
        else:
            return erf((z @ basis) / denominators)
        
    def _detect_watermark(self, posteriors: torch.Tensor) -> Tuple[bool, float]:
        """Detect watermark in posteriors."""
        generator_matrix, parity_check_matrix, one_time_pad, false_positive_rate, noise_rate, test_bits, g, max_bp_iter, t = self.decoding_key
        posteriors = (1 - 2 * noise_rate) * (1 - 2 * np.array(one_time_pad, dtype=float)) * posteriors.numpy(force=True)
        
        r = parity_check_matrix.shape[0]
        Pi = np.prod(posteriors[parity_check_matrix.indices.reshape(r, t)], axis=1)
        log_plus = np.log((1 + Pi) / 2)
        log_minus = np.log((1 - Pi) / 2)
        log_prod = log_plus + log_minus
        
        const = 0.5 * np.sum(np.power(log_plus, 2) + np.power(log_minus, 2) - 0.5 * np.power(log_prod, 2))
        threshold = np.sqrt(2 * const * np.log(1 / false_positive_rate)) + 0.5 * log_prod.sum()
        #print(f"threshold: {threshold}")
        return log_plus.sum() >= threshold#, log_plus.sum()
        
    def _boolean_row_reduce(self, A, print_progress=False):
        """Given a GF(2) matrix, do row elimination and return the first k rows of A that form an invertible matrix

        Args:
            A (np.ndarray): A GF(2) matrix
            print_progress (bool, optional): Whether to print the progress. Defaults to False.

        Returns:
            np.ndarray: The first k rows of A that form an invertible matrix
        """
        n, k = A.shape
        A_rr = A.copy()
        perm = np.arange(n)
        for j in range(k):
            idxs = j + np.nonzero(A_rr[j:, j])[0]
            if idxs.size == 0:
                print("The given matrix is not invertible")
                return None
            A_rr[[j, idxs[0]]] = A_rr[[idxs[0], j]]  # For matrices you have to swap them this way
            (perm[j], perm[idxs[0]]) = (perm[idxs[0]], perm[j])  # Weirdly, this is MUCH faster if you swap this way instead of using perm[[i,j]]=perm[[j,i]]
            A_rr[idxs[1:]] += A_rr[j]
            if print_progress and (j%5==0 or j+1==k):
                sys.stdout.write(f'\rDecoding progress: {j + 1} / {k}')
                sys.stdout.flush()
        if print_progress: print()
        return perm[:k]
        
    def _decode_message(self, posteriors, print_progress=False, max_bp_iter=None):
        generator_matrix, parity_check_matrix, one_time_pad, false_positive_rate_key, noise_rate, test_bits, g, max_bp_iter_key, t = self.decoding_key
        if max_bp_iter is None:
            max_bp_iter = max_bp_iter_key

        posteriors = (1 - 2 * noise_rate) * (1 - 2 * np.array(one_time_pad, dtype=float)) * posteriors.numpy(force=True)
        channel_probs = (1 - np.abs(posteriors)) / 2
        x_recovered = (1 - np.sign(posteriors)) // 2
        

        # Apply the belief-propagation decoder.
        if print_progress:
            print("Running belief propagation...")
        bpd = bp_decoder(parity_check_matrix, channel_probs=channel_probs, max_iter=max_bp_iter, bp_method="product_sum")
        x_decoded = bpd.decode(x_recovered)

        # Compute a confidence score.
        bpd_probs = 1 / (1 + np.exp(bpd.log_prob_ratios))
        confidences = 2 * np.abs(0.5 - bpd_probs)

        # Order codeword bits by confidence.
        confidence_order = np.argsort(-confidences)
        ordered_generator_matrix = generator_matrix[confidence_order]
        ordered_x_decoded = x_decoded[confidence_order].astype(int)

        # Find the first (according to the confidence order) linearly independent set of rows of the generator matrix.
        top_invertible_rows = self._boolean_row_reduce(ordered_generator_matrix, print_progress=print_progress)
        if top_invertible_rows is None:
            return None

        # Solve the system.
        if print_progress:
            print("Solving linear system...")
        recovered_string = np.linalg.solve(ordered_generator_matrix[top_invertible_rows], self.GF(ordered_x_decoded[top_invertible_rows]))

        return np.array(recovered_string[len(test_bits) + g:])
        
    def bits_to_string(self, bits):
        return ''.join(map(str, bits))

    def recover(self, idx_list, message_list, distance_list, message_length):
        """Recover the original message sequence from indices, messages, and distances."""
        valid_entries = [(idx, msg, dist) 
                        for idx, msg, dist in zip(idx_list, message_list, distance_list) 
                        if idx != -1]
        valid_entries.sort(key=lambda x: x[0])

        sorted_order, recovered_message, recovered_distance = [], [], []
        valid_idx = 0

        for idx in idx_list:
            if idx == -1:
                sorted_order.append(-1)
                recovered_message.append([-1] * message_length)
                recovered_distance.append(message_length)
            else:
                sorted_order.append(valid_entries[valid_idx][0])
                recovered_message.append(valid_entries[valid_idx][1])
                recovered_distance.append(valid_entries[valid_idx][2])
                valid_idx += 1

        return (np.array(sorted_order),
                np.array(recovered_message),
                np.array(recovered_distance))



    def eval_watermark(self, reversed_latents: torch.Tensor, reference_latents: torch.Tensor = None, detector_type: str = "is_watermark") -> float:
        """Evaluate watermark in reversed latents."""
        if detector_type != 'bit_acc':
            raise ValueError(f'Detector type {detector_type} is not supported for VideoMark. Use "bit_acc" instead.')
        idx_list, message_list, distance_list = [], [], []
        message_length = self.message_sequence.shape[1]
        message_sequence_str = [self.bits_to_string(msg) for msg in self.message_sequence]

        for frame_index in range(1, self.num_frames):

            reversed_prc = self._recover_posteriors(
                reversed_latents[:, :, frame_index].to(torch.float64).flatten().cpu(),
                variances=self.var
            ).flatten().cpu()
            self.recovered_prc = reversed_prc

            # Detect & Decode
            if not self._detect_watermark(reversed_prc):
                decode_message = np.full((1, message_length), -1)
                decode_message_str = "<message_placeholder>"
            else:
                decode_message = self._decode_message(reversed_prc)
                decode_message_str = self.bits_to_string(decode_message)

            # Temporal Message Matching (TMM)
            distances = np.array([distance(decode_message_str, msg) for msg in message_sequence_str])
            min_distance = distances.min()
            idx = -1 if np.all(distances == distances[0]) else distances.argmin()

            message_list.append(decode_message)
            distance_list.append(min_distance)
            idx_list.append(idx)

        recovered_index, recovered_message, recovered_distance = self.recover(
            idx_list, message_list, distance_list, message_length
        )
        bit_acc = np.mean(recovered_message == self.watermark[1:])

        return {
            "bit_acc": float(bit_acc),
            "recovered_index": recovered_index,
            "recovered_message": recovered_message,
            "recovered_distance": recovered_distance,
        }
    
