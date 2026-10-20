import { ChangeDetectionStrategy, Component, DestroyRef, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { HFModelInfo, ModelInfo } from '../../core/models/model.model';
import { HuggingFaceModelService } from '../../core/services/huggingface-model.service';
import { ModelManagerService } from '../../core/services/model-manager.service';
import { WebLLMService } from '../../core/services/webllm.service';
import { copyText } from '../../core/utils/clipboard';
import { formatBytes, percentage } from '../../core/utils/format';
import {
  BadgeComponent,
  ButtonComponent,
  DialogComponent,
  IconComponent,
  InputComponent,
  SheetComponent,
} from '../../shared/components';

@Component({
  selector: 'app-models',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [
    FormsModule,
    ButtonComponent,
    BadgeComponent,
    IconComponent,
    InputComponent,
    SheetComponent,
    DialogComponent,
  ],
  templateUrl: './models.component.html',
})
export class ModelsComponent {
  private readonly modelManager = inject(ModelManagerService);
  private readonly webllm = inject(WebLLMService);
  private readonly hfService = inject(HuggingFaceModelService);
  private readonly router = inject(Router);
  private readonly destroyRef = inject(DestroyRef);

  readonly models = this.modelManager.models;
  readonly selectedModelId = this.modelManager.selectedModelId;
  readonly selectedModel = this.modelManager.selectedModel;
  readonly isChecking = this.modelManager.isChecking;
  readonly downloadedModels = this.modelManager.downloadedModels;
  readonly supportsF16 = this.webllm.supportsF16;

  readonly hfModels = this.hfService.models;
  readonly hfDefaultModelId = this.hfService.defaultModelId;
  readonly hfIsLoading = this.hfService.isLoading;
  readonly hfSearchError = this.hfService.searchError;

  readonly storageInfo = signal<{ used: string; quota: string; percent: number } | null>(null);
  readonly searchQuery = signal('');
  readonly showAddModal = signal(false);
  readonly showRegistryModal = signal(false);
  readonly showHfModal = signal(false);
  readonly showHfDownloadModal = signal(false);
  readonly customModelId = signal('');
  readonly hfRepoId = signal('');
  readonly hfSearchQuery = signal('');
  readonly hfSearchResults = signal<HFModelInfo[]>([]);
  readonly selectedHfModel = signal<HFModelInfo | null>(null);
  readonly isDeleting = signal<string | null>(null);
  readonly pendingDelete = signal<ModelInfo | null>(null);
  readonly pendingHfDelete = signal<HFModelInfo | null>(null);
  readonly registryModels = signal<string[]>([]);
  readonly registryFilter = signal('');
  readonly notice = signal<string | null>(null);

  private noticeTimer: number | null = null;

  readonly filteredModels = computed(() => {
    const query = this.searchQuery().trim().toLowerCase();
    if (!query) return this.models();
    return this.models().filter(
      (model) =>
        model.name.toLowerCase().includes(query) || model.id.toLowerCase().includes(query),
    );
  });

  readonly filteredRegistryModels = computed(() => {
    const filter = this.registryFilter().trim().toLowerCase();
    return this.registryModels()
      .filter((modelId) => modelId.toLowerCase().includes(filter))
      .slice(0, 60);
  });

  constructor() {
    void this.webllm.detectAvailableGPUs();
    void this.loadStorageInfo();

    this.destroyRef.onDestroy(() => {
      if (this.noticeTimer !== null) clearTimeout(this.noticeTimer);
    });
  }

  isModelDisabled(model: ModelInfo): boolean {
    return Boolean(model.requiresF16 && !this.supportsF16());
  }

  selectModel(modelId: string): void {
    const model = this.models().find((item) => item.id === modelId);
    if (model && this.isModelDisabled(model)) {
      this.showNotice('Este modelo requiere soporte float16.');
      return;
    }
    this.modelManager.selectModel(modelId);
    this.showNotice(`Modelo activo: ${model?.name ?? modelId}`);
  }

  useModel(modelId: string): void {
    this.selectModel(modelId);
    this.router.navigate(['/playground']);
  }

  openPlayground(): void {
    this.router.navigate(['/playground']);
  }

  requestDelete(model: ModelInfo): void {
    this.pendingDelete.set(model);
  }

  async confirmDelete(): Promise<void> {
    const model = this.pendingDelete();
    if (!model) return;

    this.isDeleting.set(model.id);
    try {
      await this.modelManager.deleteModel(model.id);
      await this.loadStorageInfo();
      this.showNotice(`"${model.name}" eliminado de la caché.`);
    } catch {
      this.showNotice('No se pudo eliminar el modelo.');
    } finally {
      this.isDeleting.set(null);
      this.pendingDelete.set(null);
    }
  }

  openAddModal(): void {
    this.showAddModal.set(true);
  }

  closeAddModal(): void {
    this.showAddModal.set(false);
    this.customModelId.set('');
  }

  addCustomModel(): void {
    const modelId = this.customModelId().trim();
    if (!modelId) return;

    if (this.models().some((model) => model.id === modelId)) {
      this.showNotice('Ese modelo ya está en el catálogo.');
      return;
    }

    this.modelManager.addCustomModel({ modelId, modelUrl: '' });
    this.showNotice('Modelo agregado al catálogo.');
    this.closeAddModal();
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
    if (this.models().some((model) => model.id === modelId)) {
      this.showNotice('Ese modelo ya está en el catálogo.');
      return;
    }
    this.modelManager.addCustomModel({ modelId, modelUrl: '' });
    this.showNotice('Modelo agregado desde el registro.');
    this.closeRegistryModal();
  }

  async refreshModels(): Promise<void> {
    await this.modelManager.loadModels();
    await this.loadStorageInfo();
    this.showNotice('Catálogo actualizado.');
  }

  openHfModal(): void {
    this.showHfModal.set(true);
  }

  closeHfModal(): void {
    this.showHfModal.set(false);
    this.hfRepoId.set('');
    this.hfSearchQuery.set('');
    this.hfSearchResults.set([]);
  }

  async searchHfModels(): Promise<void> {
    const query = this.hfSearchQuery().trim();
    if (!query) return;
    this.hfSearchResults.set(await this.hfService.searchModels(query));
  }

  addHfModel(): void {
    const repoId = this.hfRepoId().trim();
    if (!repoId) return;

    this.hfService.addModel({
      repoId,
      name: repoId.split('/').pop() || repoId,
      description: 'Modelo agregado manualmente.',
    });
    this.showNotice('Modelo de Hugging Face agregado.');
    this.closeHfModal();
  }

  addHfModelFromSearch(model: HFModelInfo): void {
    this.hfService.addModel(model);
    this.showNotice('Modelo de Hugging Face agregado.');
    this.closeHfModal();
  }

  requestHfDelete(model: HFModelInfo): void {
    this.pendingHfDelete.set(model);
  }

  confirmHfDelete(): void {
    const model = this.pendingHfDelete();
    if (!model) return;
    this.hfService.removeModel(model.repoId);
    this.pendingHfDelete.set(null);
    this.showNotice(`"${model.name}" eliminado de la lista.`);
  }

  setDefaultHfModel(repoId: string): void {
    this.hfService.setDefaultModel(repoId);
    this.showNotice('Modelo predeterminado actualizado.');
  }

  showDownloadInstructions(model: HFModelInfo): void {
    this.selectedHfModel.set(model);
    this.showHfDownloadModal.set(true);
  }

  closeHfDownloadModal(): void {
    this.showHfDownloadModal.set(false);
    this.selectedHfModel.set(null);
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

  async copy(value: string, label = 'Contenido'): Promise<void> {
    const copied = await copyText(value);
    this.showNotice(copied ? `${label} copiado al portapapeles.` : 'No se pudo copiar.');
  }

  private async loadStorageInfo(): Promise<void> {
    const info = await this.modelManager.getStorageUsage();
    if (!info) return;
    this.storageInfo.set({
      used: formatBytes(info.used),
      quota: formatBytes(info.quota),
      percent: percentage(info.used, info.quota),
    });
  }

  private showNotice(message: string): void {
    this.notice.set(message);
    if (this.noticeTimer !== null) clearTimeout(this.noticeTimer);
    this.noticeTimer = window.setTimeout(() => this.notice.set(null), 2400);
  }
}
