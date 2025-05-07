Troubleshooting Guide
===================

This guide covers common issues and their solutions when using IANN.

Training Issues
-------------

1. **Memory Issues**

   * **Problem**: Out of Memory (OOM) errors during training
   * **Solutions**:
     * Reduce batch size
     * Use gradient checkpointing
     * Enable mixed precision training
     * Clear GPU cache between runs

2. **Training Instability**

   * **Problem**: Loss becomes NaN or training diverges
   * **Solutions**:
     * Reduce learning rate
     * Enable gradient clipping
     * Check data normalization
     * Verify input data quality

3. **Slow Training**

   * **Problem**: Training is slower than expected
   * **Solutions**:
     * Increase batch size if memory allows
     * Use multiple workers for data loading
     * Enable mixed precision training
     * Profile training to identify bottlenecks

DDP Issues
---------

1. **Gradient Strides Warning**

   * **Problem**: Warning about gradient strides not matching bucket view strides
   * **Solution**: This is a known PyTorch DDP warning that can be safely ignored. It doesn't affect training accuracy.

2. **Communication Errors**

   * **Problem**: DDP communication failures
   * **Solutions**:
     * Check network connectivity
     * Verify NCCL installation
     * Increase DDP timeout
     * Check GPU compatibility

3. **Synchronization Issues**

   * **Problem**: Models on different GPUs become desynchronized
   * **Solutions**:
     * Use consistent random seeds
     * Check data loading order
     * Verify batch size consistency
     * Monitor gradient norms

Prediction Issues
--------------

1. **Incorrect Predictions**

   * **Problem**: Model predictions are inaccurate
   * **Solutions**:
     * Verify model loading
     * Check input data normalization
     * Ensure cutoff radius matches training
     * Validate atomic numbers

2. **Performance Issues**

   * **Problem**: Slow prediction speed
   * **Solutions**:
     * Use batch processing
     * Enable CUDA if available
     * Optimize data loading
     * Consider model quantization

LAMMPS Integration
---------------

1. **Model Loading**

   * **Problem**: LAMMPS fails to load the model
   * **Solutions**:
     * Verify model export format
     * Check file permissions
     * Ensure correct LAMMPS version
     * Validate model compatibility

2. **Energy/Force Issues**

   * **Problem**: Incorrect energies or forces in LAMMPS
   * **Solutions**:
     * Check unit conversion
     * Verify cutoff radius
     * Validate energy/force scaling
     * Test with simple systems

3. **Performance Problems**

   * **Problem**: Slow MD simulations
   * **Solutions**:
     * Optimize neighbor list settings
     * Adjust communication settings
     * Use appropriate parallelization
     * Profile simulation

General Tips
----------

1. **Debugging**

   * Enable debug logging
   * Use smaller test cases
   * Check intermediate outputs
   * Monitor memory usage

2. **Performance Optimization**

   * Profile your code
   * Use appropriate batch sizes
   * Enable mixed precision
   * Optimize data loading

3. **Best Practices**

   * Keep track of model versions
   * Document configuration changes
   * Use version control
   * Regular testing

For more specific issues or if you need additional help, please:

1. Check the GitHub issues page
2. Review the API documentation
3. Contact the maintainers 

Maintainers
----------

Maintainer ``Dr. Changzhi Ai`` (changzhi@stanford.edu) at SUNCAT center, Stanford University and SLAC, who is supervised by Dr. Johannes Voss and Dr. Frank Abild-Pedersen.