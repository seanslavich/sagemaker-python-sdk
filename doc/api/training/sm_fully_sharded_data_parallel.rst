#################################################################
SageMaker Fully Sharded Data Parallel (FSDP) Library Overview
#################################################################

SageMaker's Fully Sharded Data Parallel library (FSDP) extends SageMaker's training capabilities on large deep learning models with improved memory efficiency and scalability, enabling training of models that would otherwise be too large to fit in GPU memory.

When training very large models, especially those with billions of parameters, machine learning practitioners often face memory constraints that limit the size of models that can be trained on a single GPU. FSDP addresses this challenge by sharding model parameters, gradients, and optimizer states across data parallel workers, allowing for training of larger models with minimal code changes.

SageMaker's FSDP library addresses memory constraints and improves scalability in several ways:

The library shards model parameters, gradients, and optimizer states across multiple GPUs, reducing the memory footprint on each device.

- It implements efficient communication primitives to synchronize sharded parameters during the forward and backward passes.

- The library integrates seamlessly with SageMaker's distributed training infrastructure, leveraging AWS's network capabilities for optimal performance.

- It provides various sharding strategies and configurations to balance between memory savings and computational overhead.

To learn more about the core features of this library, see Introduction to SageMaker's Fully Sharded Data Parallel Library in the SageMaker Developer Guide.
