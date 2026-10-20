import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { Router } from '@angular/router';
import { BadgeComponent, ButtonComponent, IconComponent } from '../../shared/components';
import type { IconName } from '../../shared/components';
import { RouterLink } from '@angular/router';
import { ModelInfo } from '../../core/models/model.model';
import { ChatHistoryService } from '../../core/services/chat-history.service';
import { HuggingFaceModelService } from '../../core/services/huggingface-model.service';
import { ModelManagerService } from '../../core/services/model-manager.service';
import { WebLLMService } from '../../core/services/webllm.service';
import { copyText } from '../../core/utils/clipboard';
import { formatBytes, percentage } from '../../core/utils/format';

interface Capability {
  index: string;
  label: string;
  title: string;
  description: string;
  icon: IconName;
  path: string;
  cta: string;
}

@Component({
  selector: 'app-home',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [RouterLink, BadgeComponent, ButtonComponent, IconComponent],
  templateUrl: './home.component.html',
})
export class HomeComponent {
  private readonly router = inject(Router);
  private readonly modelManager = inject(ModelManagerService);
  private readonly webllm = inject(WebLLMService);
  private readonly chatHistory = inject(ChatHistoryService);
  private readonly hfService = inject(HuggingFaceModelService);

  readonly models = this.modelManager.models;
  readonly isChecking = this.modelManager.isChecking;
  readonly downloadedCount = this.modelManager.downloadedCount;
  readonly selectedModel = this.modelManager.selectedModel;
  readonly selectedModelId = this.modelManager.selectedModelId;
  readonly hfDefaultModel = this.hfService.defaultModel;
  readonly totalSessions = computed(() => this.chatHistory.sessions().length);
  readonly totalMessages = this.chatHistory.totalMessages;

  readonly storage = signal<{ used: string; quota: string; percent: number } | null>(null);
  readonly webgpuState = signal<'checking' | 'ready' | 'unsupported'>('checking');
  readonly copyState = signal<'idle' | 'copied'>('idle');

  readonly quickstart = [
    'git clone https://github.com/ULFBERTO/ZeroCloud.git',
    'cd ZeroCloud',
    'npm install',
    'npm start',
  ].join('\n');

  readonly featuredModels = computed(() => this.models().slice(0, 4));

  readonly gpuName = computed(() => {
    const gpu = this.webllm.availableGPUs()[0];
    return gpu?.name || 'Adaptador por defecto';
  });

  readonly runtimeLabel = computed(() => {
    switch (this.webgpuState()) {
      case 'ready':
        return 'WebGPU listo';
      case 'unsupported':
        return 'WebGPU no disponible';
      default:
        return 'Detectando runtime';
    }
  });

  readonly heroMetrics = computed(() => [
    { label: 'Runtime', value: 'WebGPU', hint: '100% navegador' },
    { label: 'Catálogo', value: `${this.models().length} modelos`, hint: `${this.downloadedCount()} en caché` },
    { label: 'GPU', value: this.gpuName(), hint: 'adaptador activo' },
    { label: 'Privacidad', value: 'On-device', hint: '0 telemetría' },
  ]);

  readonly capabilities: Capability[] = [
    {
      index: '01',
      label: 'Local first',
      title: 'Privacidad por defecto',
      description:
        'Los pesos y tus conversaciones permanecen en el dispositivo. No hay backend que pueda leerlos.',
      icon: 'shield',
      path: '/docs',
      cta: 'Arquitectura',
    },
    {
      index: '02',
      label: 'Runtime',
      title: 'WebGPU + WebLLM',
      description:
        'Aprovecha la GPU para inferencia en streaming con modelos MLC, ONNX y TensorFlow.js.',
      icon: 'gpu',
      path: '/models',
      cta: 'Ver modelos',
    },
    {
      index: '03',
      label: 'Distribuido',
      title: 'Clúster P2P',
      description:
        'Conecta dispositivos por WebRTC y reparte tareas de inferencia sin un coordinador central.',
      icon: 'nodes',
      path: '/playground',
      cta: 'Abrir playground',
    },
    {
      index: '04',
      label: 'State space',
      title: 'OxideLLM SSM',
      description:
        'Un experimento Mamba-like con escaneo lineal. Menos memoria, contexto más largo.',
      icon: 'zap',
      path: '/ssm-chat',
      cta: 'Probar SSM',
    },
  ];

  constructor() {
    void this.loadRuntimeInfo();
  }

  runtimeBadgeClass(): string {
    const base =
      'inline-flex items-center gap-1.5 rounded-[2px] border px-2 py-1 font-mono text-[8px] uppercase tracking-[0.12em]';
    switch (this.webgpuState()) {
      case 'ready':
        return `${base} border-positive/25 bg-positive-soft text-positive`;
      case 'unsupported':
        return `${base} border-negative/25 bg-negative-soft text-negative`;
      default:
        return `${base} border-line bg-surface text-muted`;
    }
  }

  async copyQuickstart(): Promise<void> {
    const copied = await copyText(this.quickstart);
    if (!copied) return;
    this.copyState.set('copied');
    window.setTimeout(() => this.copyState.set('idle'), 1800);
  }

  openPlayground(): void {
    this.router.navigate(['/playground']);
  }

  openModels(): void {
    this.router.navigate(['/models']);
  }

  openDocs(): void {
    this.router.navigate(['/docs']);
  }

  openPath(path: string): void {
    this.router.navigate([path]);
  }

  useModel(model: ModelInfo): void {
    if (this.isModelDisabled(model)) {
      this.router.navigate(['/models']);
      return;
    }
    this.modelManager.selectModel(model.id);
    this.router.navigate(['/playground']);
  }

  isModelDisabled(model: ModelInfo): boolean {
    return Boolean(model.requiresF16 && !this.webllm.supportsF16());
  }

  private async loadRuntimeInfo(): Promise<void> {
    const gpuDetection = this.webllm.detectAvailableGPUs();
    const [gpuSupport, storage] = await Promise.all([
      this.webllm.checkWebGPUSupport(),
      this.modelManager.getStorageUsage(),
    ]);
    await gpuDetection;

    this.webgpuState.set(gpuSupport.supported ? 'ready' : 'unsupported');

    if (storage) {
      this.storage.set({
        used: formatBytes(storage.used),
        quota: formatBytes(storage.quota),
        percent: percentage(storage.used, storage.quota),
      });
    }
  }
}
