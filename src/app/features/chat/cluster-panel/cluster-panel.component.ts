import { CommonModule } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, inject } from '@angular/core';
import { GPUClusterService } from '../../../core/services/gpu-cluster.service';
import { DistributedInferenceService } from '../../../core/services/distributed-inference.service';
import { P2PSyncService } from '../../../core/services/p2p-sync.service';
import { BadgeComponent, IconComponent } from '../../../shared/components';

@Component({
  selector: 'app-cluster-panel',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [CommonModule, BadgeComponent, IconComponent],
  templateUrl: './cluster-panel.component.html',
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
  readonly canCreatePlan = computed(
    () => this.remoteNodes().length > 0 || this.localNode() !== null,
  );

  readonly healthVariant = computed<'success' | 'warning' | 'error' | 'neutral'>(() => {
    switch (this.clusterHealth()) {
      case 'healthy':
        return 'success';
      case 'degraded':
        return 'warning';
      case 'critical':
        return 'error';
      default:
        return 'neutral';
    }
  });

  readonly healthLabel = computed(() => {
    switch (this.clusterHealth()) {
      case 'healthy':
        return 'Saludable';
      case 'degraded':
        return 'Degradado';
      case 'critical':
        return 'Crítico';
      default:
        return 'Inactivo';
    }
  });

  statusClass(status: string): string {
    const base =
      'inline-flex items-center border px-1.5 py-[2px] font-mono text-[8px] uppercase tracking-[0.08em]';
    switch (status) {
      case 'idle':
        return `${base} border-positive/25 bg-positive-soft text-positive`;
      case 'computing':
        return `${base} border-ink bg-ink text-paper`;
      case 'syncing':
        return `${base} border-accent/20 bg-accent-soft text-accent`;
      case 'error':
        return `${base} border-negative/25 bg-negative-soft text-negative`;
      default:
        return `${base} border-line bg-surface text-muted`;
    }
  }

  taskStatusClass(status: string): string {
    const base =
      'inline-flex items-center border px-1.5 py-[2px] font-mono text-[8px] uppercase tracking-[0.08em]';
    switch (status) {
      case 'running':
        return `${base} border-warning/25 bg-warning-soft text-warning`;
      case 'completed':
        return `${base} border-positive/25 bg-positive-soft text-positive`;
      case 'failed':
        return `${base} border-negative/25 bg-negative-soft text-negative`;
      default:
        return `${base} border-line bg-surface text-muted`;
    }
  }

  stageClass(status: string): string {
    switch (status) {
      case 'computing':
        return 'bg-warning animate-pulse';
      case 'completed':
        return 'bg-positive';
      case 'failed':
        return 'bg-negative';
      default:
        return 'bg-line';
    }
  }

  async joinCluster(): Promise<void> {
    try {
      await this.cluster.joinCluster();
    } catch (error) {
      console.error('Failed to join cluster:', error);
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
