import { Injectable, signal, computed, inject } from '@angular/core';
import { P2PSyncService } from './p2p-sync.service';
import { GPUClusterService } from './gpu-cluster.service';
import { WebLLMService } from './webllm.service';
import {
  TensorMetadata,
  TensorChunk,
  DistributedInferenceTask,
  PipelineStage,
  ModelDistributionPlan,
} from '../interfaces/distributed-compute.interface';

const CHUNK_SIZE = 1024 * 1024; // 1MB chunks para transferencia

interface PendingTask {
  task: DistributedInferenceTask;
  resolve: (result: string) => void;
  reject: (error: Error) => void;
  tensorBuffer: Map<string, ArrayBuffer[]>;
}

@Injectable({ providedIn: 'root' })
export class DistributedInferenceService {
  private readonly p2p = inject(P2PSyncService);
  private readonly cluster = inject(GPUClusterService);
  private readonly webllm = inject(WebLLMService);

  private pendingTasks = new Map<string, PendingTask>();
  private tensorCache = new Map<string, ArrayBuffer>();

  // === State ===
  private readonly _activeTasks = signal<DistributedInferenceTask[]>([]);
  private readonly _isProcessing = signal(false);
  private readonly _currentTaskId = signal<string | null>(null);
  private readonly _transferProgress = signal<{ sent: number; total: number } | null>(null);

  readonly activeTasks = this._activeTasks.asReadonly();
  readonly isProcessing = this._isProcessing.asReadonly();
  readonly currentTaskId = this._currentTaskId.asReadonly();
  readonly transferProgress = this._transferProgress.asReadonly();

  readonly canUseDistributed = computed(() => 
    this.cluster.canDistribute() && this.cluster.clusterState().currentPlan !== null
  );

  constructor() {
    this.setupMessageHandlers();
  }

  // === Message Handlers ===

  private setupMessageHandlers(): void {
    window.addEventListener('p2p-compute-message', ((event: CustomEvent) => {
      const { message, fromId } = event.detail;
      this.handleComputeMessage(message, fromId);
    }) as EventListener);
  }

  private handleComputeMessage(msg: Record<string, unknown>, fromId: string): void {
    switch (msg['type']) {
      case 'compute-request':
        this.handleComputeRequest(
          msg['taskId'] as string,
          msg['inputTensor'] as TensorMetadata,
          msg['tensorData'] as ArrayBuffer,
          msg['targetLayers'] as number[],
          fromId
        );
        break;

      case 'compute-result':
        this.handleComputeResult(
          msg['taskId'] as string,
          msg['outputTensor'] as TensorMetadata,
          msg['data'] as ArrayBuffer,
          fromId
        );
        break;

      case 'tensor-chunk':
        this.handleTensorChunk(
          msg['taskId'] as string,
          msg['chunk'] as TensorChunk,
          fromId
        );
        break;

      case 'compute-error':
        this.handleComputeError(
          msg['taskId'] as string,
          msg['error'] as string
        );
        break;

      case 'pipeline-progress':
        this.handlePipelineProgress(
          msg['taskId'] as string,
          msg['stageIndex'] as number,
          msg['status'] as PipelineStage['status']
        );
        break;
    }
  }

  // === Distributed Inference ===

  async runDistributedInference(prompt: string): Promise<string> {
    const plan = this.cluster.clusterState().currentPlan;
    if (!plan) {
      throw new Error('No distribution plan available. Create one first.');
    }

    const taskId = crypto.randomUUID();
    const pipeline = this.createPipeline(plan);

    const task: DistributedInferenceTask = {
      taskId,
      prompt,
      modelId: plan.modelId,
      pipeline,
      status: 'queued',
      createdAt: Date.now(),
    };

    return new Promise((resolve, reject) => {
      this.pendingTasks.set(taskId, {
        task,
        resolve,
        reject,
        tensorBuffer: new Map(),
      });

      this._activeTasks.update(tasks => [...tasks, task]);
      this._currentTaskId.set(taskId);
      this._isProcessing.set(true);

      this.startPipelineExecution(taskId, prompt, plan).catch(err => {
        this.handleComputeError(taskId, err.message);
      });
    });
  }

  private createPipeline(plan: ModelDistributionPlan): PipelineStage[] {
    return plan.pipelineOrder.map((peerId, index) => ({
      stageIndex: index,
      assignedNode: peerId,
      layers: plan.layerAssignments.get(peerId) || [],
      inputTensorId: index === 0 ? null : `tensor-${index - 1}`,
      outputTensorId: `tensor-${index}`,
      status: 'pending' as const,
    }));
  }

  private async startPipelineExecution(
    taskId: string,
    prompt: string,
    plan: ModelDistributionPlan
  ): Promise<void> {
    const pending = this.pendingTasks.get(taskId);
    if (!pending) return;

    // Actualizar estado
    pending.task.status = 'running';
    this.updateTaskState(taskId, { status: 'running' });

    const myPeerId = this.p2p.peerId();
    const firstNode = plan.pipelineOrder[0];

    if (firstNode === myPeerId) {
      // Somos el primer nodo, procesar embedding/primeras capas
      await this.processLocalStage(taskId, prompt, 0, plan);
    } else {
      // Enviar al primer nodo
      const inputTensor = await this.tokenizePrompt(prompt);
      this.sendComputeRequest(taskId, firstNode, inputTensor, plan.layerAssignments.get(firstNode) || []);
    }
  }

  private async processLocalStage(
    taskId: string,
    input: string | ArrayBuffer,
    stageIndex: number,
    plan: ModelDistributionPlan
  ): Promise<void> {
    const pending = this.pendingTasks.get(taskId);
    if (!pending) return;

    const myPeerId = this.p2p.peerId()!;
    const myLayers = plan.layerAssignments.get(myPeerId) || [];

    // Actualizar estado del stage
    this.updateStageStatus(taskId, stageIndex, 'computing');
    this.cluster.updateLocalStatus('computing', 80);

    try {
      // Procesar con WebLLM local (simulado para capas específicas)
      // En una implementación real, aquí se ejecutarían solo las capas asignadas
      const isLastStage = stageIndex === plan.pipelineOrder.length - 1;

      if (isLastStage) {
        // Última etapa: generar respuesta final
        const result = await this.generateFinalOutput(input, myLayers);
        this.completeTask(taskId, result);
      } else {
        // Etapa intermedia: procesar y pasar al siguiente nodo
        const outputTensor = await this.processLayers(input, myLayers);
        const nextNode = plan.pipelineOrder[stageIndex + 1];
        const nextLayers = plan.layerAssignments.get(nextNode) || [];

        this.updateStageStatus(taskId, stageIndex, 'completed');
        this.sendComputeRequest(taskId, nextNode, outputTensor, nextLayers);
      }
    } catch (error) {
      this.handleComputeError(taskId, error instanceof Error ? error.message : 'Unknown error');
    } finally {
      this.cluster.updateLocalStatus('idle', 0);
    }
  }

  // === Tensor Operations ===

  private async tokenizePrompt(prompt: string): Promise<TensorMetadata> {
    // Convertir prompt a tensor de tokens
    // En implementación real, usaría el tokenizer del modelo
    const encoder = new TextEncoder();
    const encoded = encoder.encode(prompt);
    
    const tensorId = `input-${Date.now()}`;
    const buffer = encoded.buffer as ArrayBuffer;
    this.tensorCache.set(tensorId, buffer);

    return {
      id: tensorId,
      shape: [1, encoded.length],
      dtype: 'uint8',
      byteSize: buffer.byteLength,
      layerIndex: -1,
      sequencePosition: 0,
    };
  }

  private async processLayers(
    input: string | ArrayBuffer,
    _layers: number[]
  ): Promise<TensorMetadata> {
    // Simulación de procesamiento de capas
    // En implementación real, ejecutaría forward pass parcial
    
    const inputBuffer = typeof input === 'string' 
      ? new TextEncoder().encode(input).buffer as ArrayBuffer
      : input;

    // Simular procesamiento con delay proporcional a capas
    await new Promise(resolve => setTimeout(resolve, _layers.length * 10));

    const tensorId = `hidden-${Date.now()}`;
    this.tensorCache.set(tensorId, inputBuffer);

    return {
      id: tensorId,
      shape: [1, 4096], // Hidden size típico
      dtype: 'float32',
      byteSize: inputBuffer.byteLength,
      layerIndex: _layers[_layers.length - 1] || 0,
      sequencePosition: 0,
    };
  }

  private async generateFinalOutput(
    input: string | ArrayBuffer,
    _layers: number[]
  ): Promise<string> {
    // En implementación real, procesaría las últimas capas y decodificaría
    // Por ahora, usamos WebLLM completo como fallback
    
    if (typeof input === 'string') {
      return await this.webllm.sendMessage(input);
    }

    // Decodificar tensor a texto (simulado)
    const decoder = new TextDecoder();
    const text = decoder.decode(input);
    return await this.webllm.sendMessage(text);
  }

  // === Network Communication ===

  private sendComputeRequest(
    taskId: string,
    targetPeerId: string,
    tensor: TensorMetadata,
    targetLayers: number[]
  ): void {
    const tensorData = this.tensorCache.get(tensor.id);
    
    if (tensorData && tensorData.byteLength > CHUNK_SIZE) {
      // Enviar en chunks
      this.sendTensorChunked(taskId, targetPeerId, tensor, tensorData);
    } else {
      // Enviar directo
      this.p2p.sendToPeer(targetPeerId, {
        type: 'compute-request',
        _compute: true,
        taskId,
        inputTensor: tensor,
        tensorData,
        targetLayers,
      });
    }
  }

  private sendTensorChunked(
    taskId: string,
    targetPeerId: string,
    tensor: TensorMetadata,
    data: ArrayBuffer
  ): void {
    const totalChunks = Math.ceil(data.byteLength / CHUNK_SIZE);
    
    this._transferProgress.set({ sent: 0, total: totalChunks });

    for (let i = 0; i < totalChunks; i++) {
      const start = i * CHUNK_SIZE;
      const end = Math.min(start + CHUNK_SIZE, data.byteLength);
      const chunkData = data.slice(start, end);

      const chunk: TensorChunk = {
        tensorId: tensor.id,
        chunkIndex: i,
        totalChunks,
        data: chunkData,
        checksum: this.computeChecksum(chunkData),
      };

      this.p2p.sendToPeer(targetPeerId, {
        type: 'tensor-chunk',
        _compute: true,
        taskId,
        chunk,
        metadata: i === 0 ? tensor : undefined,
      });

      this._transferProgress.update(p => p ? { ...p, sent: i + 1 } : null);
    }

    this._transferProgress.set(null);
  }

  private computeChecksum(data: ArrayBuffer): string {
    // Simple checksum (en producción usar algo más robusto)
    const view = new Uint8Array(data);
    let sum = 0;
    for (let i = 0; i < view.length; i++) {
      sum = (sum + view[i]) & 0xFFFFFFFF;
    }
    return sum.toString(16);
  }

  // === Incoming Message Handlers ===

  private async handleComputeRequest(
    taskId: string,
    inputTensor: TensorMetadata,
    tensorData: ArrayBuffer,
    targetLayers: number[],
    fromId: string
  ): Promise<void> {
    this.cluster.updateLocalStatus('computing', 90);

    try {
      // Almacenar tensor recibido
      if (tensorData) {
        this.tensorCache.set(inputTensor.id, tensorData);
      }

      // Procesar capas asignadas
      const outputTensor = await this.processLayers(tensorData, targetLayers);
      const outputData = this.tensorCache.get(outputTensor.id);

      // Determinar siguiente paso
      const plan = this.cluster.clusterState().currentPlan;
      if (!plan) {
        throw new Error('No distribution plan');
      }

      const myPeerId = this.p2p.peerId()!;
      const myIndex = plan.pipelineOrder.indexOf(myPeerId);
      const isLast = myIndex === plan.pipelineOrder.length - 1;

      if (isLast) {
        // Generar resultado final y enviar de vuelta
        const result = await this.generateFinalOutput(tensorData, targetLayers);
        this.p2p.sendToPeer(fromId, {
          type: 'compute-result',
          _compute: true,
          taskId,
          outputTensor,
          result,
        });
      } else {
        // Pasar al siguiente nodo en el pipeline
        const nextNode = plan.pipelineOrder[myIndex + 1];
        const nextLayers = plan.layerAssignments.get(nextNode) || [];
        this.sendComputeRequest(taskId, nextNode, outputTensor, nextLayers);

        // Notificar progreso
        this.p2p.sendToPeer(fromId, {
          type: 'pipeline-progress',
          _compute: true,
          taskId,
          stageIndex: myIndex,
          status: 'completed',
        });
      }
    } catch (error) {
      this.p2p.sendToPeer(fromId, {
        type: 'compute-error',
        _compute: true,
        taskId,
        error: error instanceof Error ? error.message : 'Unknown error',
      });
    } finally {
      this.cluster.updateLocalStatus('idle', 0);
    }
  }

  private handleComputeResult(
    taskId: string,
    _outputTensor: TensorMetadata,
    _data: ArrayBuffer,
    _fromId: string
  ): void {
    const pending = this.pendingTasks.get(taskId);
    if (!pending) return;

    // El resultado viene como string en el mensaje
    const msg = arguments[arguments.length - 1] as Record<string, unknown>;
    const result = msg['result'] as string;

    if (result) {
      this.completeTask(taskId, result);
    }
  }

  private handleTensorChunk(
    taskId: string,
    chunk: TensorChunk,
    _fromId: string
  ): void {
    const pending = this.pendingTasks.get(taskId);
    if (!pending) return;

    // Acumular chunks
    if (!pending.tensorBuffer.has(chunk.tensorId)) {
      pending.tensorBuffer.set(chunk.tensorId, new Array(chunk.totalChunks));
    }

    const chunks = pending.tensorBuffer.get(chunk.tensorId)!;
    chunks[chunk.chunkIndex] = chunk.data;

    // Verificar si tenemos todos los chunks
    const complete = chunks.every(c => c !== undefined);
    if (complete) {
      // Reconstruir tensor
      const totalSize = chunks.reduce((sum, c) => sum + c.byteLength, 0);
      const fullBuffer = new ArrayBuffer(totalSize);
      const view = new Uint8Array(fullBuffer);
      
      let offset = 0;
      for (const c of chunks) {
        view.set(new Uint8Array(c), offset);
        offset += c.byteLength;
      }

      this.tensorCache.set(chunk.tensorId, fullBuffer);
      pending.tensorBuffer.delete(chunk.tensorId);
    }
  }

  private handleComputeError(taskId: string, error: string): void {
    const pending = this.pendingTasks.get(taskId);
    if (!pending) return;

    pending.task.status = 'failed';
    pending.task.error = error;
    
    this.updateTaskState(taskId, { status: 'failed', error });
    pending.reject(new Error(error));
    
    this.cleanupTask(taskId);
  }

  private handlePipelineProgress(
    taskId: string,
    stageIndex: number,
    status: PipelineStage['status']
  ): void {
    this.updateStageStatus(taskId, stageIndex, status);
  }

  // === Task Management ===

  private completeTask(taskId: string, result: string): void {
    const pending = this.pendingTasks.get(taskId);
    if (!pending) return;

    pending.task.status = 'completed';
    pending.task.result = result;
    pending.task.completedAt = Date.now();

    this.updateTaskState(taskId, { 
      status: 'completed', 
      result,
      completedAt: Date.now(),
    });

    pending.resolve(result);
    this.cleanupTask(taskId);
  }

  private cleanupTask(taskId: string): void {
    this.pendingTasks.delete(taskId);
    this._currentTaskId.set(null);
    this._isProcessing.set(false);

    // Limpiar tensores del cache después de un delay
    setTimeout(() => {
      this.tensorCache.forEach((_, key) => {
        if (key.includes(taskId)) {
          this.tensorCache.delete(key);
        }
      });
    }, 5000);
  }

  private updateTaskState(taskId: string, updates: Partial<DistributedInferenceTask>): void {
    this._activeTasks.update(tasks => 
      tasks.map(t => t.taskId === taskId ? { ...t, ...updates } : t)
    );
  }

  private updateStageStatus(
    taskId: string,
    stageIndex: number,
    status: PipelineStage['status']
  ): void {
    this._activeTasks.update(tasks =>
      tasks.map(t => {
        if (t.taskId !== taskId) return t;
        const pipeline = [...t.pipeline];
        if (pipeline[stageIndex]) {
          pipeline[stageIndex] = {
            ...pipeline[stageIndex],
            status,
            ...(status === 'computing' ? { startTime: Date.now() } : {}),
            ...(status === 'completed' ? { endTime: Date.now() } : {}),
          };
        }
        return { ...t, pipeline };
      })
    );
  }

  // === Fallback to Local ===

  async runWithFallback(prompt: string): Promise<string> {
    if (this.canUseDistributed()) {
      try {
        return await this.runDistributedInference(prompt);
      } catch (error) {
        console.warn('Distributed inference failed, falling back to local:', error);
      }
    }
    
    // Fallback a inferencia local
    return await this.webllm.sendMessage(prompt);
  }

  // === Cleanup ===

  clearTensorCache(): void {
    this.tensorCache.clear();
  }

  cancelTask(taskId: string): void {
    const pending = this.pendingTasks.get(taskId);
    if (pending) {
      pending.reject(new Error('Task cancelled'));
      this.cleanupTask(taskId);
    }
  }
}
