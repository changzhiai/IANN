from fairchem.core.common.data_parallel import BalancedBatchSampler

sampler = BalancedBatchSampler(
            dataset,
            batch_size=batch_size,
            num_replicas=distutils.get_world_size(),
            rank=distutils.get_rank(),
            device=self.device,
            mode=balancing_mode,
            shuffle=shuffle,
            force_balancing=force_balancing,
        )