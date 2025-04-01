import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit  # Handles CUDA context creation
import os
from consts import *
class TensorRTInference:
    def __init__(self, engine_path: str):
        """
        Initialize TensorRT engine for inference.
        
        Args:
            engine_path (str): Path to the serialized TensorRT engine file
        """
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self._setup_bindings()

    def _load_engine(self, engine_path: str) -> trt.ICudaEngine:
        """Load and deserialize a TensorRT engine from file."""
        try:
            with open(engine_path, "rb") as f:
                runtime = trt.Runtime(self.logger)
                return runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorRT engine: {e}")

    def _setup_bindings(self):
        """Setup input/output bindings with proper memory allocation."""
        self.bindings = []
        self.input_shape = None
        self.output_shapes = []

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            
            # Allocate device memory
            size = trt.volume(shape) * np.dtype(dtype).itemsize
            device_mem = cuda.mem_alloc(size)
            self.bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                self.input_shape = shape
            else:
                self.output_shapes.append(shape)

    def infer(self, input_batch: np.ndarray) -> list[np.ndarray]:
        """
        Perform inference on input batch.
        
        Args:
            input_batch (np.ndarray): Input image batch in NCHW format
            
        Returns:
            list[np.ndarray]: List of output arrays
        """
        # Validate input shape
        if input_batch.shape != self.input_shape:
            raise ValueError(
                f"Input shape mismatch. Expected {self.input_shape}, got {input_batch.shape}"
            )

        # Create output buffers
        outputs = [
            np.empty(shape, dtype=trt.nptype(self.engine.get_binding_dtype(binding)))
            for shape in self.output_shapes
        ]

        # Transfer data to GPU and run inference
        cuda.memcpy_htod(self.bindings[0], input_batch)
        self.context.execute_v2(bindings=self.bindings)
        
        # Transfer results back to CPU
        for i, output in enumerate(outputs, start=1):
            cuda.memcpy_dtoh(output, self.bindings[i])

        return outputs

    def __del__(self):
        """Clean up CUDA resources."""
        if hasattr(self, 'bindings'):
            for binding in self.bindings:
                cuda.mem_free(binding)

def build_engine(onnx_path: str, engine_path: str, fp16: bool = True):
    """
    Build and save a TensorRT engine from ONNX model.
    
    Args:
        onnx_path (str): Path to ONNX model
        engine_path (str): Path to save TensorRT engine
        fp16 (bool): Enable FP16 precision
    """
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse ONNX model
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError(f"ONNX parsing failed: {parser.get_error(0)}")

    # Build configuration
    config = builder.create_builder_config()
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Optimize for TensorRT
    config.max_workspace_size = 1 << 30  # 1GB
    engine = builder.build_engine(network, config)

    # Save engine
    with open(engine_path, "wb") as f:
        f.write(engine.serialize())

# Example Usage
if __name__ == "__main__":
    # Step 1: Convert ONNX to TensorRT engine (do this once)
    build_engine(os.path.join(MODEL_WEIGHTS,"yolo11n.onnx"), os.path.join(MODEL_WEIGHTS,"yolov11n.engine"))
    
    # Step 2: Run inference
    trt_model = TensorRTInference(os.path.join(MODEL_WEIGHTS,"yolo11n.onnx"))
    
    # Create dummy input (replace with actual preprocessed image)
    input_batch = np.random.randn(1, 3, 640, 640).astype(np.float32)  # Example shape for YOLO
    results = trt_model.infer(input_batch)
    
    print(f"Inference completed. Output shapes: {[r.shape for r in results]}")