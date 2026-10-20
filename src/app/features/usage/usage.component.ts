import { ChangeDetectionStrategy, Component, OnInit, computed, inject, signal } from '@angular/core';
import { RouterLink } from '@angular/router';
import { ChatHistoryService } from '../../core/services/chat-history.service';
import { ModelManagerService } from '../../core/services/model-manager.service';
import { WebLLMService } from '../../core/services/webllm.service';
import { formatBytes, percentage } from '../../core/utils/format';
import { BadgeComponent, IconComponent } from '../../shared/components';

@Component({
  selector: 'app-usage',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [RouterLink, BadgeComponent, IconComponent],
  templateUrl: './usage.component.html',
})
export class UsageComponent implements OnInit {
  private readonly modelManager = inject(ModelManagerService);
  private readonly chatHistory = inject(ChatHistoryService);
  private readonly webllm = inject(WebLLMService);

  readonly models = this.modelManager.models;
  readonly selectedModel = this.modelManager.selectedModel;
  readonly downloadedModels = this.modelManager.downloadedModels;
  readonly sessions = this.chatHistory.sortedSessions;
  readonly totalMessages = this.chatHistory.totalMessages;
  readonly availableGPUs = this.webllm.availableGPUs;
  readonly supportsF16 = this.webllm.supportsF16;

  readonly storage = signal<{ used: number; quota: number; percent: number } | null>(null);
  readonly isRefreshing = signal(false);

  readonly customModelCount = computed(
    () => this.models().filter((model) => model.isCustom).length,
  );

  readonly storageLabel = computed(() => {
    const value = this.storage();
    return value ? `${formatBytes(value.used)} de ${formatBytes(value.quota)}` : 'Calculando…';
  });

  readonly recentSessions = computed(() => this.sessions().slice(0, 5));

  ngOnInit(): void {
    void this.webllm.detectAvailableGPUs();
    void this.refreshStorage();
  }

  async refreshStorage(): Promise<void> {
    this.isRefreshing.set(true);
    try {
      const estimate = await this.modelManager.getStorageUsage();
      this.storage.set(
        estimate
          ? {
              ...estimate,
              percent: percentage(estimate.used, estimate.quota),
            }
          : null,
      );
    } finally {
      this.isRefreshing.set(false);
    }
  }

  formatSize(bytes: number): string {
    return formatBytes(bytes);
  }

  formatModelSize(size: string | undefined): string {
    return size ?? '—';
  }

  formatDate(value: Date): string {
    return new Intl.DateTimeFormat('es-ES', {
      day: '2-digit',
      month: 'short',
      hour: '2-digit',
      minute: '2-digit',
    }).format(new Date(value));
  }
}
