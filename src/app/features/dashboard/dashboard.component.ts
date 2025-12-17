import { Component, inject, signal, computed } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { ModelManagerService, ModelInfo } from '../../core/services/model-manager.service';
import { WebLLMService } from '../../core/services/webllm.service';

@Component({
  selector: 'app-dashboard',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './dashboard.component.html',
})
export class DashboardComponent {
  private readonly modelManager = inject(ModelManagerService);
  private readonly webllm = inject(WebLLMService);
  private readonly router = inject(Router);

  readonly models = this.modelManager.models;
  readonly selectedModelId = this.modelManager.selectedModelId;
  readonly isChecking = this.modelManager.isChecking;
  readonly downloadedModels = this.modelManager.downloadedModels;

  readonly storageInfo = signal<{ used: string; quota: string; percent: number } | null>(null);
  readonly showAddModal = signal(false);
  readonly showRegistryModal = signal(false);
  readonly customModelId = signal('');
  readonly isDeleting = signal<string | null>(null);
  readonly registryModels = signal<string[]>([]);
  readonly registryFilter = signal('');

  readonly filteredRegistryModels = computed(() => {
    const filter = this.registryFilter().toLowerCase();
    return this.registryModels().filter((m) => m.toLowerCase().includes(filter)).slice(0, 50);
  });

  constructor() {
    this.loadStorageInfo();
  }

  async loadStorageInfo(): Promise<void> {
    const info = await this.modelManager.getStorageUsage();
    if (info) {
      this.storageInfo.set({
        used: this.formatBytes(info.used),
        quota: this.formatBytes(info.quota),
        percent: Math.round((info.used / info.quota) * 100),
      });
    }
  }

  private formatBytes(bytes: number): string {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }


  selectModel(modelId: string): void {
    this.modelManager.selectModel(modelId);
  }

  async deleteModel(model: ModelInfo): Promise<void> {
    if (!confirm(`¿Eliminar "${model.name}" del caché? Tendrás que descargarlo de nuevo.`)) {
      return;
    }

    this.isDeleting.set(model.id);
    try {
      await this.modelManager.deleteModel(model.id);
      await this.loadStorageInfo();
    } catch (error) {
      console.error('Error deleting model:', error);
    } finally {
      this.isDeleting.set(null);
    }
  }

  startChat(): void {
    this.router.navigate(['/chat']);
  }

  openAddModal(): void {
    this.showAddModal.set(true);
  }

  closeAddModal(): void {
    this.showAddModal.set(false);
    this.customModelId.set('');
  }

  openRegistryModal(): void {
    this.registryModels.set(this.modelManager.getAvailableModelsFromRegistry());
    this.showRegistryModal.set(true);
  }

  closeRegistryModal(): void {
    this.showRegistryModal.set(false);
    this.registryFilter.set('');
  }

  addFromRegistry(modelId: string): void {
    this.modelManager.addCustomModel({ modelId, modelUrl: '' });
    this.closeRegistryModal();
  }

  addCustomModel(): void {
    const modelId = this.customModelId().trim();
    if (!modelId) return;

    this.modelManager.addCustomModel({ modelId, modelUrl: '' });
    this.closeAddModal();
  }

  async refreshModels(): Promise<void> {
    await this.modelManager.loadModels();
    await this.loadStorageInfo();
  }
}
