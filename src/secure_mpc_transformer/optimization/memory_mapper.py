"""
Memory-mapped file support for efficient large model loading and management.

This module provides memory-mapped access to model files, enabling efficient
loading of large models without consuming excessive RAM.
"""

import torch
import numpy as np
import mmap
import os
import struct
import json
import pickle
from typing import Dict, List, Optional, Any, Tuple, Union, BinaryIO
from dataclasses import dataclass
from pathlib import Path
import logging
import threading
import hashlib
from contextlib import contextmanager
from abc import ABC, abstractmethod
import tempfile

logger = logging.getLogger(__name__)


@dataclass
class MemoryMapConfig:
    """Configuration for memory-mapped model loading."""
    enable_mmap: bool = True
    preload_critical_layers: bool = True
    cache_hot_tensors: bool = True
    lazy_loading: bool = True
    mmap_threshold_mb: int = 100  # Only mmap files larger than this
    page_size: int = 4096
    prefetch_size: int = 1024 * 1024  # 1MB prefetch
    enable_compression: bool = False
    compression_algorithm: str = "lz4"  # lz4, gzip, zstd


class MemoryMappedTensor:
    """A tensor backed by memory-mapped storage."""
    
    def __init__(self, file_path: str, offset: int, shape: Tuple[int, ...], 
                 dtype: torch.dtype, device: Optional[torch.device] = None):
        self.file_path = file_path
        self.offset = offset
        self.shape = shape
        self.dtype = dtype
        self.device = device or torch.device('cpu')
        
        # Memory mapping
        self._mmap: Optional[mmap.mmap] = None
        self._file: Optional[BinaryIO] = None
        self._cached_tensor: Optional[torch.Tensor] = None
        self._lock = threading.Lock()
        
        # Metadata
        self.numel = int(np.prod(shape))
        self.element_size = torch.tensor([], dtype=dtype).element_size()
        self.total_bytes = self.numel * self.element_size
        
        logger.debug(f"Created memory-mapped tensor: {shape}, {dtype}, {self.total_bytes} bytes")
    
    def _open_mmap(self):
        """Open memory-mapped file."""
        if self._mmap is None:
            self._file = open(self.file_path, 'rb')
            self._mmap = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
    
    def _close_mmap(self):
        """Close memory-mapped file."""
        if self._mmap:
            self._mmap.close()
            self._mmap = None
        if self._file:
            self._file.close()
            self._file = None
    
    def load(self, cache: bool = True) -> torch.Tensor:
        """Load tensor from memory-mapped storage."""
        with self._lock:
            # Return cached tensor if available
            if self._cached_tensor is not None:
                return self._cached_tensor
            
            # Open memory map if needed
            self._open_mmap()
            
            # Read data from memory map
            data_bytes = self._mmap[self.offset:self.offset + self.total_bytes]
            
            # Convert to numpy array
            np_array = np.frombuffer(data_bytes, dtype=self._torch_to_numpy_dtype(self.dtype))
            np_array = np_array.reshape(self.shape)
            
            # Convert to tensor
            tensor = torch.from_numpy(np_array.copy()).to(dtype=self.dtype, device=self.device)
            
            # Cache if requested
            if cache:
                self._cached_tensor = tensor
            
            return tensor
    
    def _torch_to_numpy_dtype(self, torch_dtype: torch.dtype) -> np.dtype:
        """Convert torch dtype to numpy dtype."""
        mapping = {
            torch.float32: np.float32,
            torch.float64: np.float64,
            torch.float16: np.float16,
            torch.int32: np.int32,
            torch.int64: np.int64,
            torch.int16: np.int16,
            torch.int8: np.int8,
            torch.uint8: np.uint8,
            torch.bool: np.bool_
        }
        return mapping.get(torch_dtype, np.float32)
    
    def prefetch(self):
        """Prefetch tensor data into memory."""
        with self._lock:
            if self._mmap is None:
                self._open_mmap()
            
            # Touch the memory pages to trigger loading
            page_size = 4096
            for i in range(self.offset, self.offset + self.total_bytes, page_size):
                end_offset = min(i + page_size, self.offset + self.total_bytes)
                _ = self._mmap[i:end_offset]
    
    def __del__(self):
        """Cleanup memory mapping on destruction."""
        self._close_mmap()


class ModelFileFormat(ABC):
    """Abstract base class for model file formats."""
    
    @abstractmethod
    def can_handle(self, file_path: str) -> bool:
        """Check if this format can handle the file."""
        pass
    
    @abstractmethod
    def load_metadata(self, file_path: str) -> Dict[str, Any]:
        """Load model metadata."""
        pass
    
    @abstractmethod
    def create_mmap_tensors(self, file_path: str, config: MemoryMapConfig) -> Dict[str, MemoryMappedTensor]:
        """Create memory-mapped tensors from file."""
        pass


class PyTorchFormat(ModelFileFormat):
    """Handler for PyTorch .pth/.pt files."""
    
    def can_handle(self, file_path: str) -> bool:
        return file_path.endswith(('.pth', '.pt', '.bin'))
    
    def load_metadata(self, file_path: str) -> Dict[str, Any]:
        """Load PyTorch model metadata."""
        try:
            # Load just the metadata without loading tensors
            checkpoint = torch.load(file_path, map_location='cpu', weights_only=False)
            
            metadata = {
                'format': 'pytorch',
                'file_size': os.path.getsize(file_path),
                'tensors': {},
                'total_parameters': 0
            }
            
            if isinstance(checkpoint, dict):
                # Handle state_dict format
                state_dict = checkpoint.get('model_state_dict', checkpoint)
                
                for name, tensor in state_dict.items():
                    if isinstance(tensor, torch.Tensor):
                        metadata['tensors'][name] = {
                            'shape': list(tensor.shape),
                            'dtype': str(tensor.dtype),
                            'numel': tensor.numel(),
                            'size_bytes': tensor.numel() * tensor.element_size()
                        }
                        metadata['total_parameters'] += tensor.numel()
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch metadata: {e}")
            return {'format': 'pytorch', 'error': str(e)}
    
    def create_mmap_tensors(self, file_path: str, config: MemoryMapConfig) -> Dict[str, MemoryMappedTensor]:
        """Create memory-mapped tensors from PyTorch file."""
        # Note: PyTorch's native format doesn't easily support memory mapping
        # This would require a custom serialization format or conversion
        logger.warning("Native PyTorch format doesn't support efficient memory mapping")
        return {}


class SafeTensorsFormat(ModelFileFormat):
    """Handler for SafeTensors format files."""
    
    def can_handle(self, file_path: str) -> bool:
        return file_path.endswith('.safetensors')
    
    def load_metadata(self, file_path: str) -> Dict[str, Any]:
        """Load SafeTensors metadata."""
        try:
            # SafeTensors has a JSON header at the beginning
            with open(file_path, 'rb') as f:
                # Read header size
                header_size = struct.unpack('<Q', f.read(8))[0]
                # Read header JSON
                header_bytes = f.read(header_size)
                header = json.loads(header_bytes.decode('utf-8'))
            
            metadata = {
                'format': 'safetensors',
                'file_size': os.path.getsize(file_path),
                'header_size': header_size,
                'tensors': {},
                'total_parameters': 0
            }
            
            for name, info in header.items():
                if name != '__metadata__':
                    metadata['tensors'][name] = info
                    # Calculate parameters from shape
                    if 'shape' in info:
                        numel = int(np.prod(info['shape']))
                        metadata['total_parameters'] += numel
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to load SafeTensors metadata: {e}")
            return {'format': 'safetensors', 'error': str(e)}
    
    def create_mmap_tensors(self, file_path: str, config: MemoryMapConfig) -> Dict[str, MemoryMappedTensor]:
        """Create memory-mapped tensors from SafeTensors file."""
        mmap_tensors = {}
        
        try:
            metadata = self.load_metadata(file_path)
            header_size = metadata.get('header_size', 0)
            data_offset = 8 + header_size  # Skip header size bytes + header
            
            for name, info in metadata['tensors'].items():
                dtype_str = info['dtype']
                shape = tuple(info['shape'])
                offset = data_offset + info['data_offsets'][0]
                
                # Convert dtype string to torch dtype
                torch_dtype = self._parse_dtype(dtype_str)
                
                mmap_tensor = MemoryMappedTensor(
                    file_path=file_path,
                    offset=offset,
                    shape=shape,
                    dtype=torch_dtype
                )
                
                mmap_tensors[name] = mmap_tensor
            
            logger.info(f"Created {len(mmap_tensors)} memory-mapped tensors from SafeTensors file")
            
        except Exception as e:
            logger.error(f"Failed to create SafeTensors memory maps: {e}")
        
        return mmap_tensors
    
    def _parse_dtype(self, dtype_str: str) -> torch.dtype:
        """Parse dtype string to torch dtype."""
        dtype_mapping = {
            'F32': torch.float32,
            'F64': torch.float64,
            'F16': torch.float16,
            'I32': torch.int32,
            'I64': torch.int64,
            'I16': torch.int16,
            'I8': torch.int8,
            'U8': torch.uint8,
            'BOOL': torch.bool
        }
        return dtype_mapping.get(dtype_str, torch.float32)


class CustomFormat(ModelFileFormat):
    """Handler for custom binary format optimized for memory mapping."""
    
    MAGIC_NUMBER = b'SMPT'  # Secure MPC Transformer
    VERSION = 1
    
    def can_handle(self, file_path: str) -> bool:
        try:
            with open(file_path, 'rb') as f:
                magic = f.read(4)
                return magic == self.MAGIC_NUMBER
        except:
            return False
    
    def load_metadata(self, file_path: str) -> Dict[str, Any]:
        """Load custom format metadata."""
        try:
            with open(file_path, 'rb') as f:
                # Read header
                magic = f.read(4)
                version = struct.unpack('<I', f.read(4))[0]
                metadata_size = struct.unpack('<Q', f.read(8))[0]
                
                # Read metadata
                metadata_bytes = f.read(metadata_size)
                metadata = pickle.loads(metadata_bytes)
                
                metadata.update({
                    'format': 'custom',
                    'version': version,
                    'file_size': os.path.getsize(file_path)
                })
                
                return metadata
                
        except Exception as e:
            logger.error(f"Failed to load custom format metadata: {e}")
            return {'format': 'custom', 'error': str(e)}
    
    def create_mmap_tensors(self, file_path: str, config: MemoryMapConfig) -> Dict[str, MemoryMappedTensor]:
        """Create memory-mapped tensors from custom format file."""
        mmap_tensors = {}
        
        try:
            metadata = self.load_metadata(file_path)
            
            for name, info in metadata['tensors'].items():
                mmap_tensor = MemoryMappedTensor(
                    file_path=file_path,
                    offset=info['offset'],
                    shape=tuple(info['shape']),
                    dtype=getattr(torch, info['dtype'])
                )
                
                mmap_tensors[name] = mmap_tensor
            
            logger.info(f"Created {len(mmap_tensors)} memory-mapped tensors from custom format")
            
        except Exception as e:
            logger.error(f"Failed to create custom format memory maps: {e}")
        
        return mmap_tensors
    
    @classmethod
    def save_model(cls, model_dict: Dict[str, torch.Tensor], file_path: str, 
                   metadata: Optional[Dict[str, Any]] = None):
        """Save model in custom memory-mappable format."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Prepare metadata
        model_metadata = {
            'tensors': {},
            'total_parameters': 0,
            'creation_time': time.time()
        }
        
        if metadata:
            model_metadata.update(metadata)
        
        with open(file_path, 'wb') as f:
            # Write header
            f.write(cls.MAGIC_NUMBER)
            f.write(struct.pack('<I', cls.VERSION))
            
            # Calculate offsets
            header_size = 4 + 4 + 8  # magic + version + metadata_size
            current_offset = header_size
            
            # Serialize metadata to get its size
            metadata_bytes = pickle.dumps(model_metadata)
            current_offset += len(metadata_bytes)
            
            # Calculate tensor offsets
            for name, tensor in model_dict.items():
                tensor_bytes = tensor.numel() * tensor.element_size()
                
                model_metadata['tensors'][name] = {
                    'shape': list(tensor.shape),
                    'dtype': str(tensor.dtype).split('.')[-1],  # Extract just the type name
                    'offset': current_offset,
                    'size_bytes': tensor_bytes
                }
                
                model_metadata['total_parameters'] += tensor.numel()
                current_offset += tensor_bytes
            
            # Update metadata with calculated offsets
            metadata_bytes = pickle.dumps(model_metadata)
            
            # Write metadata size and metadata
            f.write(struct.pack('<Q', len(metadata_bytes)))
            f.write(metadata_bytes)
            
            # Write tensors
            for name, tensor in model_dict.items():
                tensor_cpu = tensor.detach().cpu()
                f.write(tensor_cpu.numpy().tobytes())
        
        logger.info(f"Saved model in custom format: {file_path}")


class MemoryMappedModelLoader:
    """Main class for loading models with memory mapping support."""
    
    def __init__(self, config: Optional[MemoryMapConfig] = None):
        self.config = config or MemoryMapConfig()
        
        # Format handlers
        self.format_handlers: List[ModelFileFormat] = [
            CustomFormat(),
            SafeTensorsFormat(),
            PyTorchFormat()
        ]
        
        # Loaded models cache
        self._loaded_models: Dict[str, Dict[str, MemoryMappedTensor]] = {}
        self._model_metadata: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        
        logger.info("Initialized memory-mapped model loader")
    
    def load_model(self, file_path: str, preload_layers: Optional[List[str]] = None,
                   device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """Load model with memory mapping support."""
        file_path = str(Path(file_path).resolve())
        
        with self._lock:
            if file_path in self._loaded_models:
                logger.info(f"Using cached memory-mapped model: {file_path}")
                cached_model = self._loaded_models[file_path]
            else:
                cached_model = self._load_model_internal(file_path, device)
        
        # Convert memory-mapped tensors to regular tensors
        model_dict = {}
        
        for name, mmap_tensor in cached_model.items():
            if preload_layers and name in preload_layers:
                # Preload and cache critical layers
                tensor = mmap_tensor.load(cache=True)
            else:
                # Lazy load other layers
                tensor = mmap_tensor.load(cache=self.config.cache_hot_tensors)
            
            if device is not None and device != tensor.device:
                tensor = tensor.to(device)
            
            model_dict[name] = tensor
        
        logger.info(f"Loaded model with {len(model_dict)} tensors")
        return model_dict
    
    def _load_model_internal(self, file_path: str, device: Optional[torch.device]) -> Dict[str, MemoryMappedTensor]:
        """Internal method to load model with memory mapping."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model file not found: {file_path}")
        
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb < self.config.mmap_threshold_mb:
            logger.info(f"File too small for memory mapping ({file_size_mb:.1f}MB < {self.config.mmap_threshold_mb}MB)")
            return self._load_model_fallback(file_path, device)
        
        # Find appropriate format handler
        handler = None
        for format_handler in self.format_handlers:
            if format_handler.can_handle(file_path):
                handler = format_handler
                break
        
        if handler is None:
            logger.warning(f"No memory mapping handler found for {file_path}, falling back to regular loading")
            return self._load_model_fallback(file_path, device)
        
        # Load metadata
        metadata = handler.load_metadata(file_path)
        self._model_metadata[file_path] = metadata
        
        # Create memory-mapped tensors
        mmap_tensors = handler.create_mmap_tensors(file_path, self.config)
        
        # Set device for all tensors
        if device is not None:
            for tensor in mmap_tensors.values():
                tensor.device = device
        
        # Cache the loaded model
        self._loaded_models[file_path] = mmap_tensors
        
        logger.info(f"Created memory-mapped model with {len(mmap_tensors)} tensors")
        return mmap_tensors
    
    def _load_model_fallback(self, file_path: str, device: Optional[torch.device]) -> Dict[str, MemoryMappedTensor]:
        """Fallback to regular PyTorch loading."""
        # For fallback, we create fake memory-mapped tensors that just load normally
        fallback_tensors = {}
        
        try:
            state_dict = torch.load(file_path, map_location='cpu', weights_only=True)
            
            for name, tensor in state_dict.items():
                # Create a simple wrapper that returns the tensor
                class SimpleTensor:
                    def __init__(self, tensor):
                        self.tensor = tensor
                    
                    def load(self, cache=True):
                        return self.tensor.to(device) if device else self.tensor
                
                fallback_tensors[name] = SimpleTensor(tensor)
            
        except Exception as e:
            logger.error(f"Fallback loading failed: {e}")
            raise
        
        return fallback_tensors
    
    def prefetch_model(self, file_path: str, layer_names: Optional[List[str]] = None):
        """Prefetch model data into memory."""
        file_path = str(Path(file_path).resolve())
        
        if file_path not in self._loaded_models:
            self.load_model(file_path)
        
        mmap_tensors = self._loaded_models[file_path]
        
        layers_to_prefetch = layer_names or list(mmap_tensors.keys())
        
        for layer_name in layers_to_prefetch:
            if layer_name in mmap_tensors:
                mmap_tensors[layer_name].prefetch()
        
        logger.info(f"Prefetched {len(layers_to_prefetch)} layers")
    
    @contextmanager
    def temporary_model_loading(self, file_path: str, device: Optional[torch.device] = None):
        """Context manager for temporary model loading."""
        model_dict = self.load_model(file_path, device=device)
        try:
            yield model_dict
        finally:
            # Cleanup could be implemented here if needed
            pass
    
    def get_model_info(self, file_path: str) -> Dict[str, Any]:
        """Get information about a model file."""
        file_path = str(Path(file_path).resolve())
        
        if file_path in self._model_metadata:
            return self._model_metadata[file_path]
        
        # Load metadata without full loading
        for handler in self.format_handlers:
            if handler.can_handle(file_path):
                metadata = handler.load_metadata(file_path)
                self._model_metadata[file_path] = metadata
                return metadata
        
        return {'error': 'No suitable format handler found'}
    
    def clear_cache(self, file_path: Optional[str] = None):
        """Clear cached models."""
        with self._lock:
            if file_path:
                file_path = str(Path(file_path).resolve())
                if file_path in self._loaded_models:
                    del self._loaded_models[file_path]
                    del self._model_metadata[file_path]
                    logger.info(f"Cleared cache for {file_path}")
            else:
                self._loaded_models.clear()
                self._model_metadata.clear()
                logger.info("Cleared all model cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_models = len(self._loaded_models)
            total_tensors = sum(len(model) for model in self._loaded_models.values())
            
            return {
                'cached_models': total_models,
                'total_tensors': total_tensors,
                'model_files': list(self._loaded_models.keys()),
                'memory_mapped_enabled': self.config.enable_mmap
            }