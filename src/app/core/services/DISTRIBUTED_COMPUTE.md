# WebGPU Compute Sharing - Arquitectura de Hiperconvergencia P2P

## Visión General

Este sistema implementa una arquitectura de **hiperconvergencia distribuida** que permite ejecutar modelos LLM distribuyendo el cómputo entre múltiples dispositivos conectados a una red P2P local.

```
┌─────────────────────────────────────────────────────────────────┐
│                    Red P2P (WebRTC/PeerJS)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Nodo A     │    │   Nodo B     │    │   Nodo C     │      │
│  │  RTX 4090    │◄──►│  RTX 3080    │◄──►│  Intel Arc   │      │
│  │  Capas 0-10  │    │  Capas 11-20 │    │  Capas 21-31 │      │
│  └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                   │                   │               │
│         ▼                   ▼                   ▼               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Pipeline de Inferencia Distribuida          │   │
│  │  Input → [Embedding] → [Attention] → [FFN] → Output     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Componentes Principales

### 1. GPUClusterService
Gestiona el cluster de GPUs en la red P2P.

**Responsabilidades:**
- Detección de capacidades GPU locales via WebGPU
- Registro y descubrimiento de nodos en la red
- Heartbeat y monitoreo de salud del cluster
- Benchmarking de rendimiento (TFLOPS)
- Creación de planes de distribución de capas

**Métricas recolectadas:**
- TFLOPS estimados por GPU
- VRAM disponible
- Soporte de F16
- Latencia de red entre nodos

### 2. DistributedInferenceService
Orquesta la ejecución distribuida de inferencia.

**Responsabilidades:**
- Gestión del pipeline de ejecución
- Transferencia de tensores entre nodos
- Chunking de datos grandes (>1MB)
- Fallback a inferencia local
- Tracking de tareas activas

### 3. Protocolo de Comunicación

```typescript
// Mensajes del Cluster
type ClusterMessage =
  | { type: 'join-cluster'; nodeInfo: ComputeNode }
  | { type: 'leave-cluster'; peerId: string }
  | { type: 'heartbeat'; load: number; status: string }
  | { type: 'distribution-plan'; plan: ModelDistributionPlan }
  | { type: 'benchmark-request' }
  | { type: 'benchmark-result'; score: number }

// Mensajes de Compute
type ComputeMessage =
  | { type: 'compute-request'; taskId: string; inputTensor: TensorMetadata }
  | { type: 'compute-result'; taskId: string; outputTensor: TensorMetadata }
  | { type: 'tensor-chunk'; chunk: TensorChunk }
  | { type: 'compute-error'; taskId: string; error: string }
```

## Flujo de Ejecución

### 1. Inicialización del Cluster

```
Usuario A (Host)                    Usuario B (Cliente)
     │                                    │
     │ createRoom()                       │
     │────────────────────────────────────│
     │                                    │
     │                              joinRoom(code)
     │◄───────────────────────────────────│
     │                                    │
     │ joinCluster()                joinCluster()
     │────────────────────────────────────│
     │                                    │
     │◄──── gpu-capabilities ────────────►│
     │                                    │
     │ createDistributionPlan()           │
     │────────────────────────────────────│
     │                                    │
     │──── distribution-plan ────────────►│
```

### 2. Inferencia Distribuida

```
Nodo A (Capas 0-10)    Nodo B (Capas 11-20)    Nodo C (Capas 21-31)
       │                       │                       │
       │ tokenize(prompt)      │                       │
       │                       │                       │
       │ forward(layers 0-10)  │                       │
       │                       │                       │
       │──── tensor ──────────►│                       │
       │                       │                       │
       │                       │ forward(layers 11-20) │
       │                       │                       │
       │                       │──── tensor ──────────►│
       │                       │                       │
       │                       │                       │ forward(layers 21-31)
       │                       │                       │
       │                       │                       │ decode()
       │                       │                       │
       │◄──────────────────────┼───── result ─────────│
```

## Plan de Distribución

El sistema asigna capas del modelo proporcionalmente al rendimiento de cada GPU:

```typescript
// Ejemplo con 3 nodos
const plan = {
  modelId: 'llama-3.2-1b',
  totalLayers: 32,
  layerAssignments: {
    'peer-rtx4090': [0, 1, 2, ..., 15],    // 50% - GPU más potente
    'peer-rtx3080': [16, 17, ..., 25],     // 31% - GPU media
    'peer-arc':     [26, 27, ..., 31],     // 19% - GPU menos potente
  },
  pipelineOrder: ['peer-rtx4090', 'peer-rtx3080', 'peer-arc'],
  estimatedLatencyMs: 150,
};
```

## Transferencia de Tensores

Para tensores grandes, el sistema usa chunking:

```typescript
// Tensor de 4MB dividido en chunks de 1MB
const chunks = [
  { tensorId: 'hidden-123', chunkIndex: 0, totalChunks: 4, data: ArrayBuffer },
  { tensorId: 'hidden-123', chunkIndex: 1, totalChunks: 4, data: ArrayBuffer },
  { tensorId: 'hidden-123', chunkIndex: 2, totalChunks: 4, data: ArrayBuffer },
  { tensorId: 'hidden-123', chunkIndex: 3, totalChunks: 4, data: ArrayBuffer },
];
```

## Benchmarking

El benchmark mide el rendimiento real de cada GPU:

```wgsl
// Shader de benchmark: multiplicación de matrices 1024x1024
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // Matrix multiplication C = A * B
  var sum: f32 = 0.0;
  for (var k: u32 = 0u; k < 1024u; k = k + 1u) {
    sum = sum + a[row * 1024u + k] * b[k * 1024u + col];
  }
  c[row * 1024u + col] = sum;
}
```

**Métricas:**
- `matmulTFLOPS`: Rendimiento en operaciones de punto flotante
- `memoryBandwidthGBps`: Ancho de banda de memoria
- `latencyMs`: Latencia de cómputo

## Limitaciones Actuales

1. **WebGPU no expone memoria compartida** entre dispositivos
2. **Latencia de red** añade overhead en cada salto del pipeline
3. **Sincronización de pesos** requiere que todos los nodos tengan el modelo
4. **No hay tensor parallelism real** - es pipeline parallelism

## Mejoras Futuras

1. **Speculative Decoding**: Nodos pequeños generan tokens candidatos
2. **Redundancia**: Réplicas de capas para tolerancia a fallos
3. **Caché de KV distribuido**: Compartir contexto entre nodos
4. **Compresión de tensores**: Reducir overhead de transferencia
5. **Quantización dinámica**: Adaptar precisión según ancho de banda

## Uso

```typescript
// 1. Conectar a red P2P
await p2pSync.createRoom(); // o joinRoom(code)

// 2. Unirse al cluster GPU
await gpuCluster.joinCluster();

// 3. Ejecutar benchmark (opcional)
await gpuCluster.runBenchmark();

// 4. Crear plan de distribución
gpuCluster.createDistributionPlan('model-id', 32);

// 5. Ejecutar inferencia distribuida
const result = await distributedInference.runDistributedInference(prompt);
```
