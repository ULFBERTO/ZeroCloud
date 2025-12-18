import { Component, inject, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { GPUClusterService } from '../../../core/services/gpu-cluster.service';
import { DistributedInferenceService } from '../../../core/services/distributed-inference.service';
import { P2PSyncService } from '../../../core/services/p2p-sync.service';

@Component({
  selector: 'app-cluster-panel',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="bg-gradient-to-br from-purple-900/30 to-indigo-900/30 rounded-xl p-4 border border-purple-500/30">
      <div class="flex items-center justify-between mb-4">
        <h3 class="text-lg font-semibold text-purple-300 flex items-center gap-2">
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" 
              d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"/>
          </svg>
          GPU Cluster
        </h3>
        <span [class]="healthBadgeClass()">
          {{ clusterHealth() }}
        </span>
      </div>

      <!-- Cluster Stats -->
      <div class="grid grid-cols-2 gap-3 mb-4">
        <div class="bg-black/30 rounded-lg p-3">
          <div class="text-xs text-gray-400 mb-1">Total TFLOPS</div>
          <div class="text-xl font-bold text-green-400">
            {{ totalTFLOPS() | number:'1.1-1' }}
          </div>
        </div>
        <div class="bg-black/30 rounded-lg p-3">
          <div class="text-xs text-gray-400 mb-1">Total VRAM</div>
          <div class="text-xl font-bold text-blue-400">
            {{ totalVRAM() | number:'1.0-0' }} MB
          </div>
        </div>
      </div>

      <!-- Local Node -->
      @if (localNode(); as node) {
        <div class="mb-4">
          <div class="text-xs text-gray-400 mb-2">Tu GPU</div>
          <div class="bg-black/40 rounded-lg p-3 border border-green-500/30">
            <div class="flex items-center justify-between mb-2">
              <span class="text-sm font-medium text-white">{{ node.deviceName }}</span>
              <span [class]="getStatusClass(node.status)">{{ node.status }}</span>
            </div>
            <div class="text-xs text-gray-400">
              {{ node.gpu.vendor }} {{ node.gpu.architecture }}
            </div>
            <div class="flex gap-4 mt-2 text-xs">
              <span class="text-purple-400">{{ node.gpu.estimatedTFLOPS }} TFLOPS</span>
              <span class="text-blue-400">{{ node.gpu.availableVRAM }} MB</span>
              @if (node.gpu.supportsF16) {
                <span class="text-green-400">F16 ✓</span>
              }
            </div>
            @if (node.layersAssigned.length > 0) {
              <div class="mt-2 text-xs text-yellow-400">
                Capas asignadas: {{ node.layersAssigned[0] }}-{{ node.layersAssigned[node.layersAssigned.length - 1] }}
              </div>
            }
          </div>
        </div>
      }

      <!-- Remote Nodes -->
      @if (remoteNodes().length > 0) {
        <div class="mb-4">
          <div class="text-xs text-gray-400 mb-2">Nodos Remotos ({{ remoteNodes().length }})</div>
          <div class="space-y-2 max-h-48 overflow-y-auto">
            @for (node of remoteNodes(); track node.peerId) {
              <div class="bg-black/30 rounded-lg p-3 border border-gray-700">
                <div class="flex items-center justify-between mb-1">
                  <span class="text-sm text-white">{{ node.deviceName }}</span>
                  <span [class]="getStatusClass(node.status)">{{ node.status }}</span>
                </div>
                <div class="text-xs text-gray-500">{{ node.gpu.vendor }}</div>
                <div class="flex gap-3 mt-1 text-xs">
                  <span class="text-purple-400">{{ node.benchmarkScore | number:'1.1-1' }} TFLOPS</span>
                  <span class="text-gray-400">{{ node.latencyMs }}ms</span>
                </div>
                @if (node.currentLoad > 0) {
                  <div class="mt-2 h-1 bg-gray-700 rounded-full overflow-hidden">
                    <div class="h-full bg-purple-500 transition-all" 
                         [style.width.%]="node.currentLoad"></div>
                  </div>
                }
              </div>
            }
          </div>
        </div>
      }

      <!-- Actions -->
      <div class="space-y-2">
        @if (!isInCluster()) {
          <button 
            (click)="joinCluster()"
            [disabled]="!canJoin()"
            class="w-full py-2 px-4 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 
                   disabled:text-gray-500 rounded-lg text-sm font-medium transition-colors">
            Unirse al Cluster GPU
          </button>
        } @else {
          <button 
            (click)="runBenchmark()"
            [disabled]="isBenchmarking()"
            class="w-full py-2 px-4 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-700
                   rounded-lg text-sm font-medium transition-colors">
            @if (isBenchmarking()) {
              <span class="flex items-center justify-center gap-2">
                <svg class="animate-spin h-4 w-4" viewBox="0 0 24 24">
                  <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" fill="none"/>
                  <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"/>
                </svg>
                Benchmarking...
              </span>
            } @else {
              Ejecutar Benchmark
            }
          </button>

          @if (canCreatePlan()) {
            <button 
              (click)="createDistributionPlan()"
              class="w-full py-2 px-4 bg-green-600 hover:bg-green-700 
                     rounded-lg text-sm font-medium transition-colors">
              Crear Plan de Distribución
            </button>
          }

          <button 
            (click)="leaveCluster()"
            class="w-full py-2 px-4 bg-red-600/20 hover:bg-red-600/40 text-red-400
                   rounded-lg text-sm font-medium transition-colors">
            Salir del Cluster
          </button>
        }
      </div>

      <!-- Distribution Plan -->
      @if (currentPlan(); as plan) {
        <div class="mt-4 p-3 bg-black/40 rounded-lg border border-yellow-500/30">
          <div class="text-xs text-yellow-400 mb-2">Plan de Distribución Activo</div>
          <div class="text-xs text-gray-400 space-y-1">
            <div>Modelo: {{ plan.modelId }}</div>
            <div>Capas: {{ plan.totalLayers }}</div>
            <div>Pipeline: {{ plan.pipelineOrder.length }} etapas</div>
            <div>Latencia estimada: {{ plan.estimatedLatencyMs }}ms</div>
          </div>
        </div>
      }

      <!-- Active Tasks -->
      @if (activeTasks().length > 0) {
        <div class="mt-4">
          <div class="text-xs text-gray-400 mb-2">Tareas Activas</div>
          @for (task of activeTasks(); track task.taskId) {
            <div class="bg-black/30 rounded-lg p-2 mb-2">
              <div class="flex items-center justify-between text-xs">
                <span class="text-gray-300 truncate max-w-[150px]">{{ task.prompt }}</span>
                <span [class]="getTaskStatusClass(task.status)">{{ task.status }}</span>
              </div>
              @if (task.status === 'running') {
                <div class="mt-2 flex gap-1">
                  @for (stage of task.pipeline; track stage.stageIndex) {
                    <div 
                      class="flex-1 h-2 rounded-full"
                      [class]="getStageClass(stage.status)">
                    </div>
                  }
                </div>
              }
            </div>
          }
        </div>
      }
    </div>
  `,
})
export class ClusterPanelComponent {
  private readonly cluster = inject(GPUClusterService);
  private readonly distributed = inject(DistributedInferenceService);
  private readonly p2p = inject(P2PSyncService);

  // Signals
  readonly localNode = this.cluster.localNode;
  readonly remoteNodes = this.cluster.clusterNodes;
  readonly totalTFLOPS = this.cluster.totalClusterTFLOPS;
  readonly totalVRAM = this.cluster.totalClusterVRAM;
  readonly isBenchmarking = this.cluster.isBenchmarking;
  readonly activeTasks = this.distributed.activeTasks;

  readonly clusterHealth = computed(() => this.cluster.clusterState().clusterHealth);
  readonly currentPlan = computed(() => this.cluster.clusterState().currentPlan);
  readonly isInCluster = computed(() => this.localNode() !== null);
  readonly canJoin = computed(() => this.p2p.isConnected());
  readonly canCreatePlan = computed(() => this.remoteNodes().length > 0 || this.localNode() !== null);

  healthBadgeClass = computed(() => {
    const health = this.clusterHealth();
    const base = 'px-2 py-1 rounded-full text-xs font-medium';
    switch (health) {
      case 'healthy': return `${base} bg-green-500/20 text-green-400`;
      case 'degraded': return `${base} bg-yellow-500/20 text-yellow-400`;
      case 'critical': return `${base} bg-red-500/20 text-red-400`;
      default: return `${base} bg-gray-500/20 text-gray-400`;
    }
  });

  getStatusClass(status: string): string {
    const base = 'px-2 py-0.5 rounded text-xs';
    switch (status) {
      case 'idle': return `${base} bg-green-500/20 text-green-400`;
      case 'computing': return `${base} bg-purple-500/20 text-purple-400`;
      case 'syncing': return `${base} bg-blue-500/20 text-blue-400`;
      case 'offline': return `${base} bg-gray-500/20 text-gray-400`;
      case 'error': return `${base} bg-red-500/20 text-red-400`;
      default: return `${base} bg-gray-500/20 text-gray-400`;
    }
  }

  getTaskStatusClass(status: string): string {
    const base = 'px-2 py-0.5 rounded';
    switch (status) {
      case 'queued': return `${base} bg-gray-500/20 text-gray-400`;
      case 'running': return `${base} bg-purple-500/20 text-purple-400`;
      case 'completed': return `${base} bg-green-500/20 text-green-400`;
      case 'failed': return `${base} bg-red-500/20 text-red-400`;
      default: return `${base} bg-gray-500/20 text-gray-400`;
    }
  }

  getStageClass(status: string): string {
    switch (status) {
      case 'pending': return 'bg-gray-600';
      case 'computing': return 'bg-purple-500 animate-pulse';
      case 'completed': return 'bg-green-500';
      case 'failed': return 'bg-red-500';
      default: return 'bg-gray-600';
    }
  }

  async joinCluster(): Promise<void> {
    try {
      await this.cluster.joinCluster();
    } catch (e) {
      console.error('Failed to join cluster:', e);
    }
  }

  leaveCluster(): void {
    this.cluster.leaveCluster();
  }

  async runBenchmark(): Promise<void> {
    await this.cluster.runBenchmark();
  }

  createDistributionPlan(): void {
    // Usar valores típicos de un modelo LLM pequeño
    this.cluster.createDistributionPlan('current-model', 32);
  }
}
