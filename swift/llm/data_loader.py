from typing import Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from swift.llm import to_device
from swift.utils import get_logger

logger = get_logger()


class BatchSamplerShard:

    def __init__(
        self,
        total_samples: int,
        batch_size: int,
        shuffle: bool,
        drop_last: bool,
        data_seed: Optional[int],
        tp_size: int = 1,
        group_by_length: bool = False,
        lengths=None,
    ):
        self.tp_size = tp_size
        self.total_samples = total_samples // self.world_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.base_seed = data_seed or 0
        self.curr_seed = self.base_seed
        self.group_by_length = group_by_length
        if group_by_length and not shuffle:
            raise ValueError('shuffle must be True when group_by_length is True')
        self.lengths = lengths
        if self.lengths is not None:
            self.lengths = [max(length) if isinstance(length, list) else length for length in self.lengths]

    @property
    def rank(self):
        return (dist.get_rank() // self.tp_size) if dist.is_initialized() else 0

    @property
    def world_size(self):
        return (dist.get_world_size() // self.tp_size) if dist.is_initialized() else 1

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.curr_seed)
            if self.group_by_length:
                from transformers.trainer_pt_utils import get_length_grouped_indices
                total_idx = get_length_grouped_indices(
                    self.lengths, self.batch_size * self.world_size, generator=generator)
            else:
                total_idx = torch.randperm(self.total_samples * self.world_size, generator=generator).tolist()
            total_idx = total_idx[self.rank::self.world_size]
        else:
            total_idx = range(self.rank, self.total_samples * self.world_size, self.world_size)

        batch = []
        # Last batch if not complete will be dropped.
        for idx in total_idx:
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if not self.drop_last and len(batch) > 0:
            yield batch
        return

    def set_epoch(self, epoch: int):
        self.curr_seed = self.base_seed + epoch

    def __len__(self) -> int:
        if self.drop_last:
            return self.total_samples // self.batch_size
        else:
            return (self.total_samples + self.batch_size - 1) // self.batch_size


class ImageSizeBucketSampler:
    """Sampler that groups samples by image size into buckets.
    
    This ensures that samples in the same batch have the same image size,
    reducing padding overhead in multimodal training.
    
    Args:
        total_samples: Total number of samples in the dataset
        batch_size: Number of samples per batch
        image_sizes: List of image sizes, each size is a tuple (height, width)
        shuffle: Whether to shuffle samples within and across buckets
        drop_last: Whether to drop the last incomplete batch in each bucket
        data_seed: Random seed for shuffling
        tp_size: Tensor parallel size (for multi-GPU training)
    """
    def __init__(
        self,
        total_samples: int,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = False,
        data_seed: Optional[int] = None,
        tp_size: int = 1,
        image_sizes = [],
    ):
        self.tp_size = tp_size
        self.total_samples = total_samples // self.world_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.base_seed = data_seed or 0
        self.curr_seed = self.base_seed
        
        # Group indices by image size
        self._build_buckets(image_sizes)
        
    def _build_buckets(self, image_sizes):
        """Group sample indices by their image sizes."""
        from collections import defaultdict
        assert len(image_sizes) > 0
        
        # Create buckets: size -> [list of indices]
        size_to_indices = defaultdict(list)
        for idx, size in enumerate(image_sizes):
            if size is not None:
                # Normalize size to tuple
                if isinstance(size, (list, tuple)):
                    size = tuple(size)
                size_to_indices[size].append(idx)
        
        # Convert to list of (size, indices) and sort by size for determinism
        self.buckets = sorted(size_to_indices.items(), key=lambda x: x[0])
        
        # Log bucket statistics
        total_samples_in_buckets = sum(len(indices) for _, indices in self.buckets)
        logger.info(f'ImageSizeBucketSampler: created {len(self.buckets)} buckets from {total_samples_in_buckets} samples')
        
        # Show top 5 largest buckets
        bucket_sizes = sorted([(size, len(indices)) for size, indices in self.buckets], 
                             key=lambda x: x[1], reverse=True)
        for i, (size, count) in enumerate(bucket_sizes):
            logger.debug(f'Bucket {i + 1}: size={size}, samples={count}')
    
    @property
    def rank(self):
        return (dist.get_rank() // self.tp_size) if dist.is_initialized() else 0

    @property
    def world_size(self):
        return (dist.get_world_size() // self.tp_size) if dist.is_initialized() else 1

    def __iter__(self):
        """Generate batches where each batch contains samples of the same size.
        
        Strategy for multi-GPU training:
        All ranks process the same image size at the same step to ensure:
        - Balanced forward pass time
        - Balanced memory usage
        - No waiting between GPUs
        """
        generator = torch.Generator()
        generator.manual_seed(self.curr_seed)
        
        all_batches = []
        
        # Process each bucket
        for size, indices in self.buckets:
            indices = list(indices)
            
            # Shuffle indices within bucket if needed
            if self.shuffle:
                perm = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[i] for i in perm]
            
            # Calculate how many complete mega-batches (batch_size * world_size) we can make
            mega_batch_size = self.batch_size * self.world_size
            num_mega_batches = len(indices) // mega_batch_size
            
            if not self.drop_last:
                # Include the last incomplete mega-batch
                if len(indices) % mega_batch_size > 0:
                    num_mega_batches += 1
            
            # Split into mega-batches, then distribute to ranks
            for mega_batch_idx in range(num_mega_batches):
                start = mega_batch_idx * mega_batch_size
                end = min(start + mega_batch_size, len(indices))
                mega_batch = indices[start:end]
                
                # Distribute mega-batch across ranks using chunk split (not stride)
                # This ensures all ranks process the same size image at the same step
                samples_per_rank = len(mega_batch) // self.world_size
                rank_start = self.rank * samples_per_rank
                rank_end = rank_start + samples_per_rank
                
                # Handle remainder samples for the last rank
                if self.rank == self.world_size - 1:
                    rank_end = len(mega_batch)
                
                rank_batch = mega_batch[rank_start:rank_end]
                
                # Only add if we have a complete batch (or drop_last=False)
                if len(rank_batch) == self.batch_size or (not self.drop_last and len(rank_batch) > 0):
                    all_batches.append(rank_batch)
        
        # Shuffle batches across buckets if needed
        # Important: Use same seed on all ranks to get the same shuffle order
        if self.shuffle:
            perm = torch.randperm(len(all_batches), generator=generator).tolist()
            all_batches = [all_batches[i] for i in perm]
        
        # Yield batches
        for batch in all_batches:
            yield batch
    
    def set_epoch(self, epoch: int):
        """Update seed for new epoch to get different shuffling."""
        self.curr_seed = self.base_seed + epoch

    def __len__(self) -> int:
        """Return total number of batches."""
        total_batches = 0
        for _, indices in self.buckets:
            num_samples = len(indices) // self.world_size
            if self.drop_last:
                total_batches += num_samples // self.batch_size
            else:
                total_batches += (num_samples + self.batch_size - 1) // self.batch_size
        return total_batches


class DataLoaderShard(DataLoader):

    def __init__(self, dataset, device=None, **dataloader_params):
        self.device = device
        super().__init__(dataset, **dataloader_params)

    def set_epoch(self, epoch: int):
        if self.batch_sampler is not None and hasattr(self.batch_sampler, 'set_epoch'):
            self.batch_sampler.set_epoch(epoch)
        elif self.sampler is not None and hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)

    def __iter__(self):
        for item in super().__iter__():
            if self.device:
                item = to_device(item, self.device)
            yield item


class DataLoaderDispatcher:

    def __init__(self, base_dataloader, device=None, skip_batches: int = 0):
        self.base_dataloader = base_dataloader
        self.device = device
        self.skip_batches = skip_batches

    @property
    def rank(self):
        return dist.get_rank(self.group) if dist.is_initialized() else 0

    @property
    def world_size(self):
        return dist.get_world_size(self.group) if dist.is_initialized() else 1

    @property
    def group(self):
        return dist.group.WORLD if dist.is_initialized() else 1

    def _scatter_object_list(self, inputs):
        if not dist.is_initialized():
            return inputs[0]
        outputs = [None]
        global_src_rank = dist.get_global_rank(self.group, 0)
        dist.scatter_object_list(outputs, inputs, global_src_rank, group=self.group)
        return outputs[0]

    def _skip_batches(self, base_iter):
        if self.rank == 0 and self.skip_batches > 0:
            for _ in tqdm(range(self.skip_batches), dynamic_ncols=True, desc='Skip Batches: '):
                [next(base_iter) for _ in range(self.world_size)]

    def __iter__(self):
        base_iter = iter(self.base_dataloader)
        self._skip_batches(base_iter)
        while True:
            if self.rank == 0:
                try:
                    data = [next(base_iter) for _ in range(self.world_size)]
                except StopIteration:
                    data = [None] * self.world_size
                data = self._scatter_object_list(data)
            else:
                data = self._scatter_object_list(None)
            if data is None:
                break
            if self.device:
                data = to_device(data, self.device)
            yield data
