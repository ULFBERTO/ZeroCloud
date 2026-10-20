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
    <div class="bg-gradient-to-br from-indigo-900/20 to-purple-900/20 rounded-3xl p-5 border border-indigo-500/20 shadow-glass">
      <div class="flex items-center justify-between mb-5">
        <div class="flex items-center gap-2">
          <div class="w-8 h-8 rounded-xl bg-indigo-500/20 flex items-center justify-center text-indigo-400">
            <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2"
                d="M9 3v2m6-2v2M9 19v2m6-2v2M5 9H3m2 6H3m18-6h-2m2 6h-2M7 19h10a2 2 0 002-2V7a2 2 0 00-2-2H7a2 2 0 00-2 2v10a2 2 0 002 2zM9 9h6v6H9V9z"/>
            </svg>
          </div>
          <h3 class="text-lg font-bold text-slate-100">GPU Cluster</h3>
        </div>
        <span [class]="healthBadgeClass()">
          {{ clusterHealth() }}
        </span>
      </div>

      <!-- Cluster Stats -->
      <div class="grid grid-cols-2 gap-4 mb-5">
        <div class="p-3 rounded-2xl bg-slate-900/50 border border-slate-700/50">
          <div class="text-[10px] font-medium text-slate-500 uppercase tracking-wider mb-1">Total TFLOPS</div>
          <div class="text-xl font-bold text-green-400">{{ totalTFLOPS() | number:'1.1-1' }}</div>
        </div>
        <div class="p-3 rounded-2xl bg-slate-900/50 border border-slate-700/50">
          <div class="text-[10px] font-medium text-slate-500 uppercase tracking-wider mb-1">Total VRAM</div>
          <div class="text-xl font-bold text-indigo-400">{{ totalVRAM() | number:'1.0-0' }} MB</div>
        </div>
      </div>

      <!-- Local Node -->
      @if (localNode(); as node) {
        <div class="mb-5">
          <div class="text-[10px] font-medium text-slate-500 uppercase tracking-wider mb-2">Tu GPU</div>
          <div class="p-3 rounded-2xl bg-slate-800/50 border border-green-500/30">
            <div class="flex items-center justify-between mb-2">
              <span class="text-sm font-medium text-slate-200">{{ node.deviceName }}</span>
              <span [class]="statusClass(node.status)">{{ node.status }}</span>
            </div>
            <div class="text-xs text-slate-500 mb-2">{{ node.gpu.vendor }} {{ node.gpu.architecture }}</div>
            <div class="flex gap-4 text-xs">
              <span class="text-purple-400">{{ node.gpu.estimatedTFLOPS }} TFLOPS</span>
              <span class="text-indigo-400">{{ node.gpu.availableVRAM }} MB</span>
              @if (node.gpu.supportsF16) {
                <span class="text-green-400">F16 ✓</span>
              }
            </div>
            @if (node.layersAssigned.length > 0) {
              <div class="mt-2.5 text-[10px] text-amber-400">
                Capas asignadas: {{ node.layersAssigned[0] }}-{{ node.layersAssigned[node.layersAssigned.length - 1] }}
              </div>
            }
          </div>
        </div>
      }

      <!-- Remote Nodes -->
      @if (remoteNodes().length > 0) {
        <div class="mb-5">
          <div class="text-[10px] font-medium text-slate-500 uppercase tracking-wider mb-2">Nodos Remotos ({{ remoteNodes().length }})</div>
          <div class="space-y-2 max-h-48 overflow-y-auto">
            @for (node of remoteNodes(); track node.peerId) {
              <div class="p-3 rounded-2xl bg-slate-800/50 border border-slate-700/50">
                <div class="flex items-center justify-between mb-1">
                  <span class="text-sm text-slate-200">{{ node.deviceName }}</span>
                  <span [class]="statusClass(node.status)">{{ node.status }}</span>
                </div>
                <div class="text-[10px] text-slate-500 mb-1">{{ node.gpu.vendor }}</div>
                <div class="flex gap-3 text-[10px]">
                  <span class="text-purple-400">{{ node.benchmarkScore | number:'1.1-1' }} TFLOPS</span>
                  <span class="text-slate-500">{{ node.latencyMs }}ms</span>
                </div>
                @if (node.currentLoad > 0) {
                  <div class="mt-2 h-1.5 bg-slate-700 rounded-full overflow-hidden">
                    <div class="h-full bg-purple-500 transition-all duration-300" [style.width.%]="node.currentLoad"></div>
                  </div>
                }
              </div>
            }
          </div>
        </div>
      }

      <!-- Actions -->
      <div class="space-y-2.5">
        @if (!isInCluster()) {
          <button
            (click)="joinCluster()"
            [disabled]="!canJoin()"
            class="w-full py-3 px-4 bg-indigo-600 text-slate-100 font-medium rounded-2xl hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-500 disabled:cursor-not-allowed transition-all duration-200 shadow-lg shadow-indigo-500/20">
            Unirse al Cluster GPU
          </button>
        } @else {
          <button
            (click)="runBenchmark()"
            [disabled]="isBenchmarking()"
            class="w-full py-3 px-4 bg-purple-600 text-slate-100 font-medium rounded-2xl hover:bg-purple-500 disabled:bg-slate-800 disabled:text-slate-500 disabled:cursor-not-allowed transition-all duration-200 shadow-lg shadow-purple-500/20">
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
              class="w-full py-3 px-4 bg-green-600 text-slate-100 font-medium rounded-2xl hover:bg-green-500 transition-all duration-200 shadow-lg shadow-green-500/20">
              Crear Plan de Distribución
            </button>
          }

          <button
            (click)="leaveCluster()"
            class="w-full py-3 px-4 text-sm text-red-500 hover:text-red-400 hover:bg-red-500/10 rounded-xl transition-colors">
            Salir del Cluster
          </button>
        }
      </div>

      <!-- Distribution Plan -->
      @if (currentPlan(); as plan) {
        <div class="mt-5 p-3.5 rounded-2xl bg-slate-900/50 border border-amber-500/30">
          <div class="text-[10px] font-medium text-amber-400 uppercase tracking-wider mb-2">Plan de Distribución Activo</div>
          <div class="text-[10px] text-slate-500 space-y-1.5">
            <div class="flex justify-between">
              <span>Modelo</span>
              <span class="text-slate-300">{{ plan.modelId }}</span>
            </div>
            <div class="flex justify-between">
              <span>Capas</span>
              <span class="text-slate-300">{{ plan.totalLayers }}</span>
            </div>
            <div class="flex justify-between">
              <span>Pipeline</span>
              <span class="text-slate-300">{{ plan.pipelineOrder.length }} etapas</span>
            </div>
            <div class="flex justify-between">
              <span>Latencia estimada</span>
              <span class="text-slate-300">{{ plan.estimatedLatencyMs }}ms</span>
            </div>
          </div>
        </div>
      }

      <!-- Active Tasks -->
      @if (activeTasks().length > 0) {
        <div class="mt-5">
          <div class="text-[10px] font-medium text-slate-500 uppercase tracking-wider mb-2">Tareas Activas</div>
          @for (task of activeTasks(); track task.taskId) {
            <div class="p-2.5 rounded-2xl bg-slate-800/50 border border-slate-700/50 mb-2">
              <div class="flex items-center justify-between text-[10px] mb-1">
                <span class="text-slate-300 truncate max-w-[160px]">{{ task.prompt }}</span>
                <span [class]="taskStatusClass(task.status)">{{ task.status }}</span>
              </div>
              @if (task.status === 'running') {
                <div class="mt-2 flex gap-1">
                  @for (stage of task.pipeline; track stage.stageIndex) {
                    <div class="flex-1 h-1.5 rounded-full" [class]="stageClass(stage.status)"></div>
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
    const base = 'px-2 py-0.5 rounded-full text-[10px] font-medium';
    switch (health) {
      case 'healthy': return `${base} bg-green-500/20 text-green-400 border border-green-500/30`;
      case 'degraded': return `${base} bg-amber-500/20 text-amber-400 border border-amber-500/30`;
      case 'critical': return `${base} bg-red-500/20 text-red-400 border border-red-500/30`;
      default: return `${base} bg-slate-700/20 text-slate-500 border border-slate-600/30`;
    }
  });

  statusClass(status: string): string {
    const base = 'px-2 py-0.5 rounded text-xs';
    switch (status) {
      case 'idle': return `${base} bg-green-500/20 text-green-400`;
      case 'computing': return `${base} bg-purple-500/20 text-purple-400`;
      case 'syncing': return `${base} bg-indigo-500/20 text-indigo-400`;
      case 'offline': return `${base} bg-slate-700/20 text-slate-500`;
      case 'error': return `${base} bg-red-500/20 text-red-400`;
      default: return `${base} bg-slate-700/20 text-slate-500`;
    }
  }

  taskStatusClass(status: string): string {
    const base = 'px-2 py-0.5 rounded';
    switch (status) {
      case 'queued': return `${base} bg-slate-700/20 text-slate-500`;
      case 'running': return `${base} bg-purple-500/20 text-purple-400`;
      case 'completed': return `${base} bg-green-500/20 text-green-400`;
      case 'failed': return `${base} bg-red-500/20 text-red-400`;
      default: return `${base} bg-slate-700/20 text-slate-500`;
    }
  }

  stageClass(status: string): string {
    switch (status) {
      case 'pending': return 'bg-slate-700';
      case 'computing': return 'bg-purple-500 animate-pulse';
      case 'completed': return 'bg-green-500';
      case 'failed': return 'bg-red-500';
      default: return 'bg-slate-700';
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
    this.cluster.createDistributionPlan('current-model', 32);
  }
}
