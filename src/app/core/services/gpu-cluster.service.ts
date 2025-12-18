import { Injectable, signal, computed, inject } from '@angular/core';
import { P2PSyncService } from './p2p-sync.service';
import {
  GPUCapabilities,
  ComputeNode,
  ClusterState,
  ModelDistributionPlan,
  GPUBenchmarkResult,
  DistributedComputeMessage,
} from '../interfaces/distributed-compute.interface';

// WebGPU type declarations for environments without @webgpu/types
declare global {
  interface GPU {
    requestAdapter(options?: { powerPreference?: string }): Promise<GPUAdapterType | null>;
  }
  interface GPUAdapterType {
    requestDevice(): Promise<GPUDeviceType>;
    requestAdapterInfo?(): Promise<GPUAdapterInfoType>;
  }
  interface GPUAdapterInfoType {
    vendor?: string;
    architecture?: string;
    device?: string;
  }
  interface GPUDeviceType {
    createBuffer(descriptor: { size: number; usage: number }): GPUBufferType;
    createShaderModule(descriptor: { code: string }): GPUShaderModuleType;
    createComputePipeline(descriptor: unknown): GPUComputePipelineType;
    createCommandEncoder(): GPUCommandEncoderType;
    createBindGroup(descriptor: unknown): GPUBindGroupType;
    queue: { submit(buffers: GPUCommandBufferType[]): void };
    destroy(): void;
  }
  interface GPUBufferType {
    destroy(): void;
  }
  interface GPUShaderModuleType {}
  interface GPUComputePipelineType {
    getBindGroupLayout(index: number): unknown;
  }
  interface GPUCommandEncoderType {
    beginComputePass(): GPUComputePassEncoderType;
    finish(): GPUCommandBufferType;
  }
  interface GPUComputePassEncoderType {
    setPipeline(pipeline: GPUComputePipelineType): void;
    setBindGroup(index: number, bindGroup: GPUBindGroupType): void;
    dispatchWorkgroups(x: number, y?: number, z?: number): void;
    end(): void;
  }
  interface GPUBindGroupType {}
  interface GPUCommandBufferType {}
}

const GPU_BUFFER_USAGE_STORAGE = 0x80;
const GPU_BUFFER_USAGE_COPY_DST = 0x08;
const GPU_BUFFER_USAGE_COPY_SRC = 0x04;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type GPUDeviceForBenchmark = any;

const HEARTBEAT_INTERVAL = 5000;
const NODE_TIMEOUT = 15000;

@Injectable({ providedIn: 'root' })
export class GPUClusterService {
  private readonly p2p = inject(P2PSyncService);
  
  private heartbeatTimer: ReturnType<typeof setInterval> | null = null;
  private localCapabilities: GPUCapabilities | null = null;
  private localBenchmark: GPUBenchmarkResult | null = null;

  // === State ===
  private readonly _clusterState = signal<ClusterState>({
    isCoordinator: false,
    nodes: new Map(),
    currentPlan: null,
    totalComputePower: 0,
    totalVRAM: 0,
    activeTaskCount: 0,
    clusterHealth: 'healthy',
  });

  private readonly _localNode = signal<ComputeNode | null>(null);
  private readonly _isBenchmarking = signal(false);

  // === Public Signals ===
  readonly clusterState = this._clusterState.asReadonly();
  readonly localNode = this._localNode.asReadonly();
  readonly isBenchmarking = this._isBenchmarking.asReadonly();

  readonly clusterNodes = computed(() => 
    Array.from(this._clusterState().nodes.values())
  );
  
  readonly activeNodes = computed(() =>
    this.clusterNodes().filter(n => n.status !== 'offline' && n.status !== 'error')
  );

  readonly totalClusterTFLOPS = computed(() =>
    this.activeNodes().reduce((sum, n) => sum + n.gpu.estimatedTFLOPS, 0) +
    (this.localNode()?.gpu.estimatedTFLOPS || 0)
  );

  readonly totalClusterVRAM = computed(() =>
    this.activeNodes().reduce((sum, n) => sum + n.gpu.availableVRAM, 0) +
    (this.localNode()?.gpu.availableVRAM || 0)
  );

  readonly canDistribute = computed(() => this.activeNodes().length > 0);

  constructor() {
    this.setupMessageHandlers();
  }

  // === Initialization ===

  async initializeLocalGPU(): Promise<GPUCapabilities | null> {
    const nav = navigator as Navigator & { gpu?: { requestAdapter: (opts?: { powerPreference?: string }) => Promise<GPUAdapterType | null> } };
    if (!nav.gpu) return null;

    try {
      const adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) return null;

      const device = await adapter.requestDevice() as GPUDeviceType & { limits: { maxBufferSize: number; maxComputeWorkgroupsPerDimension: number } };
      const info = await adapter.requestAdapterInfo?.() || {} as GPUAdapterInfoType;

      this.localCapabilities = {
        vendor: info.vendor || 'Unknown',
        architecture: info.architecture || 'Unknown',
        maxBufferSize: device.limits.maxBufferSize,
        maxComputeWorkgroupsPerDimension: device.limits.maxComputeWorkgroupsPerDimension,
        supportsF16: false, // Simplified
        estimatedTFLOPS: this.estimateTFLOPS(info),
        availableVRAM: this.estimateVRAM(device),
      };

      device.destroy();
      return this.localCapabilities;
    } catch (e) {
      console.error('Error initializing GPU:', e);
      return null;
    }
  }

  private estimateTFLOPS(info: GPUAdapterInfoType): number {
    const vendor = (info.vendor || '').toLowerCase();
    const arch = (info.architecture || '').toLowerCase();
    
    // Estimaciones basadas en arquitectura conocida
    if (vendor.includes('nvidia')) {
      if (arch.includes('ada') || arch.includes('40')) return 40;
      if (arch.includes('ampere') || arch.includes('30')) return 25;
      if (arch.includes('turing') || arch.includes('20')) return 12;
      return 8;
    }
    if (vendor.includes('amd')) {
      if (arch.includes('rdna3') || arch.includes('7')) return 30;
      if (arch.includes('rdna2') || arch.includes('6')) return 20;
      return 10;
    }
    if (vendor.includes('intel')) {
      if (arch.includes('arc')) return 15;
      return 2; // Integrada
    }
    if (vendor.includes('apple')) {
      return 10; // M1/M2 aproximado
    }
    return 5; // Default conservador
  }

  private estimateVRAM(device: GPUDeviceType & { limits: { maxBufferSize: number } }): number {
    // WebGPU no expone VRAM directamente, estimamos por maxBufferSize
    const maxBuffer = device.limits.maxBufferSize;
    // Asumimos que maxBufferSize es ~25% de VRAM total
    return Math.round((maxBuffer * 4) / (1024 * 1024));
  }

  // === Cluster Management ===

  async joinCluster(): Promise<void> {
    if (!this.localCapabilities) {
      await this.initializeLocalGPU();
    }
    if (!this.localCapabilities) {
      throw new Error('No GPU capabilities detected');
    }

    const peerId = this.p2p.peerId();
    if (!peerId) {
      throw new Error('Not connected to P2P network');
    }

    // Crear nodo local
    const localNode: ComputeNode = {
      peerId,
      deviceName: this.p2p.getDeviceInfo().name,
      gpu: this.localCapabilities,
      status: 'idle',
      currentLoad: 0,
      layersAssigned: [],
      lastHeartbeat: Date.now(),
      latencyMs: 0,
      benchmarkScore: this.localBenchmark?.matmulTFLOPS || this.localCapabilities.estimatedTFLOPS,
    };

    this._localNode.set(localNode);

    // Determinar si somos coordinador (el host de la sala)
    const isHost = this.p2p.connectionMode() === 'hosting';
    this._clusterState.update(s => ({ ...s, isCoordinator: isHost }));

    // Broadcast capabilities a la red
    this.broadcastMessage({
      type: 'join-cluster',
      nodeInfo: localNode,
    });

    // Iniciar heartbeat
    this.startHeartbeat();
  }

  leaveCluster(): void {
    this.stopHeartbeat();
    
    const peerId = this.p2p.peerId();
    if (peerId) {
      this.broadcastMessage({ type: 'leave-cluster', peerId });
    }

    this._localNode.set(null);
    this._clusterState.set({
      isCoordinator: false,
      nodes: new Map(),
      currentPlan: null,
      totalComputePower: 0,
      totalVRAM: 0,
      activeTaskCount: 0,
      clusterHealth: 'healthy',
    });
  }

  // === Heartbeat ===

  private startHeartbeat(): void {
    this.stopHeartbeat();
    
    this.heartbeatTimer = setInterval(() => {
      const local = this._localNode();
      if (local) {
        this.broadcastMessage({
          type: 'heartbeat',
          load: local.currentLoad,
          status: local.status,
        });
      }
      this.checkNodeTimeouts();
    }, HEARTBEAT_INTERVAL);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  private checkNodeTimeouts(): void {
    const now = Date.now();
    this._clusterState.update(state => {
      const nodes = new Map(state.nodes);
      let changed = false;

      nodes.forEach((node, peerId) => {
        if (now - node.lastHeartbeat > NODE_TIMEOUT && node.status !== 'offline') {
          nodes.set(peerId, { ...node, status: 'offline' });
          changed = true;
        }
      });

      if (!changed) return state;

      const activeCount = Array.from(nodes.values()).filter(n => n.status !== 'offline').length;
      const health = activeCount === 0 ? 'critical' : 
                     activeCount < nodes.size / 2 ? 'degraded' : 'healthy';

      return { ...state, nodes, clusterHealth: health };
    });
  }

  // === Message Handling ===

  private setupMessageHandlers(): void {
    window.addEventListener('p2p-cluster-message', ((event: CustomEvent) => {
      const { message, fromId } = event.detail;
      this.handleClusterMessage(message as DistributedComputeMessage, fromId);
    }) as EventListener);
  }

  private handleClusterMessage(msg: DistributedComputeMessage, fromId: string): void {
    switch (msg.type) {
      case 'join-cluster':
        this.handleNodeJoin(msg.nodeInfo as ComputeNode, fromId);
        break;

      case 'leave-cluster':
        this.handleNodeLeave(msg.peerId);
        break;

      case 'heartbeat':
        this.handleHeartbeat(fromId, msg.load, msg.status);
        break;

      case 'gpu-capabilities':
        this.handleCapabilitiesUpdate(fromId, msg.capabilities);
        break;

      case 'benchmark-request':
        this.runBenchmark().then(result => {
          if (result) {
            this.sendToPeer(fromId, { type: 'benchmark-result', score: result.matmulTFLOPS, details: result });
          }
        });
        break;

      case 'benchmark-result':
        this.handleBenchmarkResult(fromId, msg.score, msg.details);
        break;

      case 'distribution-plan':
        this.handleDistributionPlan(msg.plan);
        break;

      case 'layer-assignment':
        this.handleLayerAssignment(msg.layers, msg.modelId);
        break;
    }
  }

  private handleNodeJoin(nodeInfo: ComputeNode, fromId: string): void {
    this._clusterState.update(state => {
      const nodes = new Map(state.nodes);
      nodes.set(fromId, {
        ...nodeInfo,
        peerId: fromId,
        lastHeartbeat: Date.now(),
        status: 'idle',
      });

      const totalPower = Array.from(nodes.values())
        .reduce((sum, n) => sum + n.gpu.estimatedTFLOPS, 0);
      const totalVRAM = Array.from(nodes.values())
        .reduce((sum, n) => sum + n.gpu.availableVRAM, 0);

      return {
        ...state,
        nodes,
        totalComputePower: totalPower,
        totalVRAM,
        clusterHealth: 'healthy',
      };
    });

    // Si somos coordinador, enviar nuestras capabilities de vuelta
    if (this._clusterState().isCoordinator && this.localCapabilities) {
      this.sendToPeer(fromId, {
        type: 'gpu-capabilities',
        capabilities: this.localCapabilities,
      });
    }
  }

  private handleNodeLeave(peerId: string): void {
    this._clusterState.update(state => {
      const nodes = new Map(state.nodes);
      nodes.delete(peerId);
      return { ...state, nodes };
    });
  }

  private handleHeartbeat(fromId: string, load: number, status: ComputeNode['status']): void {
    this._clusterState.update(state => {
      const nodes = new Map(state.nodes);
      const node = nodes.get(fromId);
      if (node) {
        nodes.set(fromId, {
          ...node,
          currentLoad: load,
          status,
          lastHeartbeat: Date.now(),
        });
      }
      return { ...state, nodes };
    });
  }

  private handleCapabilitiesUpdate(fromId: string, capabilities: GPUCapabilities): void {
    this._clusterState.update(state => {
      const nodes = new Map(state.nodes);
      const node = nodes.get(fromId);
      if (node) {
        nodes.set(fromId, { ...node, gpu: capabilities });
      }
      return { ...state, nodes };
    });
  }

  private handleBenchmarkResult(fromId: string, score: number, details: GPUBenchmarkResult): void {
    this._clusterState.update(state => {
      const nodes = new Map(state.nodes);
      const node = nodes.get(fromId);
      if (node) {
        nodes.set(fromId, { ...node, benchmarkScore: score });
      }
      return { ...state, nodes };
    });
  }

  private handleDistributionPlan(plan: ModelDistributionPlan): void {
    this._clusterState.update(s => ({ ...s, currentPlan: plan }));
    
    // Aplicar asignación de capas local
    const myPeerId = this.p2p.peerId();
    if (myPeerId && plan.layerAssignments.has(myPeerId)) {
      const myLayers = plan.layerAssignments.get(myPeerId)!;
      this._localNode.update(n => n ? { ...n, layersAssigned: myLayers } : n);
    }
  }

  private handleLayerAssignment(layers: number[], _modelId: string): void {
    this._localNode.update(n => n ? { ...n, layersAssigned: layers } : n);
  }

  // === Communication Helpers ===

  private broadcastMessage(msg: DistributedComputeMessage): void {
    this.p2p.broadcastToPeers({ ...msg, _cluster: true });
  }

  private sendToPeer(peerId: string, msg: DistributedComputeMessage): void {
    this.p2p.sendToPeer(peerId, { ...msg, _cluster: true });
  }

  // === Benchmarking ===

  async runBenchmark(): Promise<GPUBenchmarkResult | null> {
    if (this._isBenchmarking()) return null;
    this._isBenchmarking.set(true);

    try {
      const nav = navigator as Navigator & { gpu?: { requestAdapter: (opts?: { powerPreference?: string }) => Promise<GPUAdapterType | null> } };
      if (!nav.gpu) return null;

      const adapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) return null;

      const device = await adapter.requestDevice();
      const result = await this.runMatmulBenchmark(device as unknown as GPUDeviceForBenchmark);
      
      device.destroy();
      this.localBenchmark = result;

      // Actualizar nodo local con benchmark score
      this._localNode.update(n => n ? { ...n, benchmarkScore: result.matmulTFLOPS } : n);

      return result;
    } catch (e) {
      console.error('Benchmark failed:', e);
      return null;
    } finally {
      this._isBenchmarking.set(false);
    }
  }

  private async runMatmulBenchmark(device: GPUDeviceForBenchmark): Promise<GPUBenchmarkResult> {
    // Benchmark simple: multiplicación de matrices 1024x1024
    const size = 1024;
    const iterations = 10;
    
    const shaderCode = `
      @group(0) @binding(0) var<storage, read> a: array<f32>;
      @group(0) @binding(1) var<storage, read> b: array<f32>;
      @group(0) @binding(2) var<storage, read_write> c: array<f32>;
      
      @compute @workgroup_size(16, 16)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let row = gid.x;
        let col = gid.y;
        if (row >= ${size}u || col >= ${size}u) { return; }
        
        var sum: f32 = 0.0;
        for (var k: u32 = 0u; k < ${size}u; k = k + 1u) {
          sum = sum + a[row * ${size}u + k] * b[k * ${size}u + col];
        }
        c[row * ${size}u + col] = sum;
      }
    `;

    const shaderModule = device.createShaderModule({ code: shaderCode });
    const pipeline = device.createComputePipeline({
      layout: 'auto',
      compute: { module: shaderModule, entryPoint: 'main' },
    });

    const bufferSize = size * size * 4;
    const bufferA = device.createBuffer({ size: bufferSize, usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST });
    const bufferB = device.createBuffer({ size: bufferSize, usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_DST });
    const bufferC = device.createBuffer({ size: bufferSize, usage: GPU_BUFFER_USAGE_STORAGE | GPU_BUFFER_USAGE_COPY_SRC });

    // Inicializar con datos aleatorios
    const dataA = new Float32Array(size * size).map(() => Math.random());
    const dataB = new Float32Array(size * size).map(() => Math.random());
    device.queue.writeBuffer(bufferA, 0, dataA);
    device.queue.writeBuffer(bufferB, 0, dataB);

    const bindGroup = device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: bufferA } },
        { binding: 1, resource: { buffer: bufferB } },
        { binding: 2, resource: { buffer: bufferC } },
      ],
    });

    // Warmup
    const warmupEncoder = device.createCommandEncoder();
    const warmupPass = warmupEncoder.beginComputePass();
    warmupPass.setPipeline(pipeline);
    warmupPass.setBindGroup(0, bindGroup);
    warmupPass.dispatchWorkgroups(Math.ceil(size / 16), Math.ceil(size / 16));
    warmupPass.end();
    device.queue.submit([warmupEncoder.finish()]);
    await device.queue.onSubmittedWorkDone();

    // Benchmark
    const startTime = performance.now();
    for (let i = 0; i < iterations; i++) {
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(size / 16), Math.ceil(size / 16));
      pass.end();
      device.queue.submit([encoder.finish()]);
    }
    await device.queue.onSubmittedWorkDone();
    const endTime = performance.now();

    // Cleanup
    bufferA.destroy();
    bufferB.destroy();
    bufferC.destroy();

    const totalTimeMs = endTime - startTime;
    const avgTimeMs = totalTimeMs / iterations;
    // FLOPS = 2 * N^3 para matmul
    const flops = 2 * Math.pow(size, 3);
    const tflops = (flops * iterations) / (totalTimeMs / 1000) / 1e12;

    return {
      matmulTFLOPS: Math.round(tflops * 100) / 100,
      memoryBandwidthGBps: Math.round((bufferSize * 3 * iterations) / (totalTimeMs / 1000) / 1e9 * 100) / 100,
      latencyMs: Math.round(avgTimeMs * 100) / 100,
      sustainedLoad: 95, // Asumimos buen rendimiento sostenido
    };
  }

  // === Distribution Planning ===

  createDistributionPlan(modelId: string, totalLayers: number): ModelDistributionPlan {
    const nodes = this.activeNodes();
    const localNode = this._localNode();
    
    const allNodes = localNode ? [...nodes, localNode] : nodes;
    if (allNodes.length === 0) {
      throw new Error('No compute nodes available');
    }

    // Ordenar nodos por benchmark score (mayor primero)
    const sortedNodes = [...allNodes].sort((a, b) => b.benchmarkScore - a.benchmarkScore);

    // Calcular peso total
    const totalScore = sortedNodes.reduce((sum, n) => sum + n.benchmarkScore, 0);

    // Asignar capas proporcionalmente al rendimiento
    const layerAssignments = new Map<string, number[]>();
    let currentLayer = 0;

    sortedNodes.forEach((node, idx) => {
      const proportion = node.benchmarkScore / totalScore;
      const layerCount = idx === sortedNodes.length - 1 
        ? totalLayers - currentLayer  // Último nodo toma el resto
        : Math.floor(totalLayers * proportion);
      
      const layers: number[] = [];
      for (let i = 0; i < layerCount && currentLayer < totalLayers; i++) {
        layers.push(currentLayer++);
      }
      
      if (layers.length > 0) {
        layerAssignments.set(node.peerId, layers);
      }
    });

    // Pipeline order basado en índice de capas
    const pipelineOrder = Array.from(layerAssignments.entries())
      .sort((a, b) => (a[1][0] || 0) - (b[1][0] || 0))
      .map(([peerId]) => peerId);

    const plan: ModelDistributionPlan = {
      modelId,
      totalLayers,
      layerAssignments,
      pipelineOrder,
      estimatedLatencyMs: this.estimatePipelineLatency(pipelineOrder),
      redundancyLevel: 0,
    };

    // Broadcast plan si somos coordinador
    if (this._clusterState().isCoordinator) {
      this.broadcastMessage({ type: 'distribution-plan', plan });
    }

    this._clusterState.update(s => ({ ...s, currentPlan: plan }));
    return plan;
  }

  private estimatePipelineLatency(pipelineOrder: string[]): number {
    // Estimar latencia basada en número de saltos de red
    const nodes = this._clusterState().nodes;
    let totalLatency = 0;
    
    for (let i = 1; i < pipelineOrder.length; i++) {
      const node = nodes.get(pipelineOrder[i]);
      totalLatency += node?.latencyMs || 50; // Default 50ms
    }
    
    return totalLatency;
  }

  // === Status Updates ===

  updateLocalStatus(status: ComputeNode['status'], load?: number): void {
    this._localNode.update(n => {
      if (!n) return n;
      return {
        ...n,
        status,
        currentLoad: load ?? n.currentLoad,
      };
    });
  }
}
