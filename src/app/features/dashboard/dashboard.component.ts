import { Component, inject, signal, computed } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { ModelManagerService, ModelInfo } from '../../core/services/model-manager.service';
import { WebLLMService } from '../../core/services/webllm.service';
import { HuggingFaceModelService, HFModelInfo } from '../../core/services/huggingface-model.service';

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
  private readonly hfService = inject(HuggingFaceModelService);

  readonly models = this.modelManager.models;
  readonly selectedModelId = this.modelManager.selectedModelId;
  readonly isChecking = this.modelManager.isChecking;
  readonly downloadedModels = this.modelManager.downloadedModels;
  readonly supportsF16 = this.webllm.supportsF16;

  // HuggingFace models
  readonly hfModels = this.hfService.models;
  readonly hfDefaultModelId = this.hfService.defaultModelId;
  readonly hfIsLoading = this.hfService.isLoading;

  constructor() {
    // Detectar GPUs para saber si soporta F16
    this.webllm.detectAvailableGPUs();
    this.loadStorageInfo();
  }

  readonly storageInfo = signal<{ used: string; quota: string; percent: number } | null>(null);
  readonly showAddModal = signal(false);
  readonly showRegistryModal = signal(false);
  readonly showHFModal = signal(false);
  readonly showHFDownloadModal = signal(false);
  readonly customModelId = signal('');
  readonly hfRepoId = signal('');
  readonly hfSearchQuery = signal('');
  readonly hfSearchResults = signal<HFModelInfo[]>([]);
  readonly selectedHFModel = signal<HFModelInfo | null>(null);
  readonly isDeleting = signal<string | null>(null);
  readonly registryModels = signal<string[]>([]);
  readonly registryFilter = signal('');

  readonly filteredRegistryModels = computed(() => {
    const filter = this.registryFilter().toLowerCase();
    return this.registryModels().filter((m) => m.toLowerCase().includes(filter)).slice(0, 50);
  });



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
    // No permitir seleccionar modelos F16 si no hay soporte
    const model = this.models().find(m => m.id === modelId);
    if (model?.requiresF16 && !this.supportsF16()) {
      return;
    }
    this.modelManager.selectModel(modelId);
  }

  isModelDisabled(model: ModelInfo): boolean {
    return !!model.requiresF16 && !this.supportsF16();
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

  goToTFJSChat(): void {
    this.router.navigate(['/tfjs-chat']);
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

  // ========== HuggingFace Methods ==========

  openHFModal(): void {
    this.showHFModal.set(true);
  }

  closeHFModal(): void {
    this.showHFModal.set(false);
    this.hfRepoId.set('');
    this.hfSearchQuery.set('');
    this.hfSearchResults.set([]);
  }

  async searchHFModels(): Promise<void> {
    const query = this.hfSearchQuery().trim();
    if (!query) return;
    
    const results = await this.hfService.searchModels(query);
    this.hfSearchResults.set(results);
  }

  addHFModel(): void {
    const repoId = this.hfRepoId().trim();
    if (!repoId) return;

    this.hfService.addModel({
      repoId,
      name: repoId.split('/').pop() || repoId,
      description: 'Modelo agregado manualmente',
    });
    this.closeHFModal();
  }

  addHFModelFromSearch(model: HFModelInfo): void {
    this.hfService.addModel(model);
    this.closeHFModal();
  }

  removeHFModel(repoId: string): void {
    if (confirm('¿Eliminar este modelo de la lista?')) {
      this.hfService.removeModel(repoId);
    }
  }

  setDefaultHFModel(repoId: string): void {
    this.hfService.setDefaultModel(repoId);
  }

  showDownloadInstructions(model: HFModelInfo): void {
    this.selectedHFModel.set(model);
    this.showHFDownloadModal.set(true);
  }

  closeHFDownloadModal(): void {
    this.showHFDownloadModal.set(false);
    this.selectedHFModel.set(null);
  }

  getCloneCommand(repoId: string): string {
    return this.hfService.getCloneCommand(repoId);
  }

  getPythonCode(repoId: string): string {
    return this.hfService.getPythonDownloadCode(repoId);
  }

  getModelUrl(repoId: string): string {
    return this.hfService.getModelUrl(repoId);
  }

  copyToClipboard(text: string): void {
    navigator.clipboard.writeText(text);
  }
}
