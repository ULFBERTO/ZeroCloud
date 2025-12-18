/**
 * WebGPU Compute Sharing - Distributed Inference Protocol
 * 
 * Arquitectura de hiperconvergencia P2P para ejecutar modelos LLM
 * distribuyendo el cómputo entre múltiples dispositivos en la red local.
 */

// === GPU Node Capabilities ===

export interface GPUCapabilities {
  vendor: string;
  architecture: string;
  maxBufferSize: number;       // bytes
  maxComputeWorkgroupsPerDimension: number;
  supportsF16: boolean;
  estimatedTFLOPS: number;     // Teraflops estimados
  availableVRAM: number;       // MB estimados disponibles
}

export interface ComputeNode {
  peerId: string;
  deviceName: string;
  gpu: GPUCapabilities;
  status: 'idle' | 'computing' | 'syncing' | 'offline' | 'error';
  currentLoad: number;         // 0-100%
  layersAssigned: number[];    // Índices de capas del modelo asignadas
  lastHeartbeat: number;
  latencyMs: number;           // Latencia de red al nodo
  benchmarkScore: number;      // Score de rendimiento normalizado
}

// === Model Distribution ===

export interface ModelLayerInfo {
  layerIndex: number;
  layerType: 'attention' | 'ffn' | 'embedding' | 'norm' | 'output';
  parameterCount: number;
  estimatedVRAM: number;       // MB
  computeIntensity: number;    // 1-10
}

export interface ModelDistributionPlan {
  modelId: string;
  totalLayers: number;
  layerAssignments: Map<string, number[]>;  // peerId -> layer indices
  pipelineOrder: string[];                   // Orden de ejecución de peers
  estimatedLatencyMs: number;
  redundancyLevel: number;                   // 0 = sin redundancia, 1+ = réplicas
}

// === Tensor Transfer Protocol ===

export interface TensorMetadata {
  id: string;
  shape: number[];
  dtype: 'float32' | 'float16' | 'int32' | 'uint8';
  byteSize: number;
  layerIndex: number;
  sequencePosition: number;
}

export interface TensorChunk {
  tensorId: string;
  chunkIndex: number;
  totalChunks: number;
  data: ArrayBuffer;
  checksum: string;
}

export interface TensorTransferRequest {
  type: 'tensor-request';
  taskId: string;
  tensorId: string;
  fromLayer: number;
  toLayer: number;
  metadata: TensorMetadata;
}

export interface TensorTransferResponse {
  type: 'tensor-response';
  taskId: string;
  tensorId: string;
  chunks: TensorChunk[];
  processingTimeMs: number;
}

// === Distributed Inference Messages ===

export type DistributedComputeMessage =
  | { type: 'gpu-capabilities'; capabilities: GPUCapabilities }
  | { type: 'join-cluster'; nodeInfo: Partial<ComputeNode> }
  | { type: 'leave-cluster'; peerId: string }
  | { type: 'heartbeat'; load: number; status: ComputeNode['status'] }
  | { type: 'distribution-plan'; plan: ModelDistributionPlan }
  | { type: 'layer-assignment'; layers: number[]; modelId: string }
  | { type: 'compute-request'; taskId: string; inputTensor: TensorMetadata; targetLayers: number[] }
  | { type: 'compute-result'; taskId: string; outputTensor: TensorMetadata; data: ArrayBuffer }
  | { type: 'compute-error'; taskId: string; error: string }
  | { type: 'sync-weights'; layerIndex: number; checksum: string }
  | { type: 'benchmark-request' }
  | { type: 'benchmark-result'; score: number; details: GPUBenchmarkResult };

// === Benchmarking ===

export interface GPUBenchmarkResult {
  matmulTFLOPS: number;        // Matrix multiplication throughput
  memoryBandwidthGBps: number; // Memory bandwidth
  latencyMs: number;           // Compute latency
  sustainedLoad: number;       // % de rendimiento sostenido
}

// === Cluster State ===

export interface ClusterState {
  isCoordinator: boolean;
  nodes: Map<string, ComputeNode>;
  currentPlan: ModelDistributionPlan | null;
  totalComputePower: number;   // TFLOPS agregados
  totalVRAM: number;           // MB agregados
  activeTaskCount: number;
  clusterHealth: 'healthy' | 'degraded' | 'critical';
}

// === Pipeline Execution ===

export interface PipelineStage {
  stageIndex: number;
  assignedNode: string;
  layers: number[];
  inputTensorId: string | null;
  outputTensorId: string | null;
  status: 'pending' | 'computing' | 'completed' | 'failed';
  startTime?: number;
  endTime?: number;
}

export interface DistributedInferenceTask {
  taskId: string;
  prompt: string;
  modelId: string;
  pipeline: PipelineStage[];
  status: 'queued' | 'running' | 'completed' | 'failed';
  createdAt: number;
  completedAt?: number;
  result?: string;
  error?: string;
}
