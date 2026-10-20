import { Injectable, computed, signal } from '@angular/core';
import { deleteModelAllInfoInCache, hasModelInCache, prebuiltAppConfig } from '@mlc-ai/web-llm';
import { RECOMMENDED_MODELS } from '../config/model-catalog';
import { CustomModelConfig, ModelInfo } from '../models/model.model';
import { readJson, readStorage, writeJson, writeStorage } from '../utils/storage';

const STORAGE_KEY = 'webllm_selected_model';
const CUSTOM_MODELS_KEY = 'webllm_custom_models';
const DEFAULT_MODEL_ID = RECOMMENDED_MODELS[0].id;

@Injectable({ providedIn: 'root' })
export class ModelManagerService {
  private readonly _models = signal<ModelInfo[]>([]);
  private readonly _selectedModelId = signal<string>(readStorage(STORAGE_KEY) ?? DEFAULT_MODEL_ID);
  private readonly _isChecking = signal(false);

  readonly models = this._models.asReadonly();
  readonly selectedModelId = this._selectedModelId.asReadonly();
  readonly isChecking = this._isChecking.asReadonly();

  readonly selectedModel = computed(() =>
    this._models().find((model) => model.id === this._selectedModelId()),
  );

  readonly downloadedModels = computed(() => this._models().filter((model) => model.isDownloaded));

  readonly downloadedCount = computed(() => this.downloadedModels().length);

  constructor() {
    void this.loadModels();
  }

  async loadModels(): Promise<void> {
    this._isChecking.set(true);

    try {
      const customModels = this.getCustomModelsFromStorage();
      const allModelConfigs = [...RECOMMENDED_MODELS, ...customModels];

      const modelsWithStatus = await Promise.all(
        allModelConfigs.map(async (model) => ({
          ...model,
          isDownloaded: await this.checkModelCached(model.id),
        })),
      );

      this._models.set(modelsWithStatus);
    } finally {
      this._isChecking.set(false);
    }
  }

  selectModel(modelId: string): void {
    this._selectedModelId.set(modelId);
    writeStorage(STORAGE_KEY, modelId);
  }

  async deleteModel(modelId: string): Promise<void> {
    await deleteModelAllInfoInCache(modelId);
    this._models.update((models) =>
      models.map((model) => (model.id === modelId ? { ...model, isDownloaded: false } : model)),
    );

    const isCustom = this.getCustomModelsFromStorage().some((model) => model.id === modelId);
    if (isCustom) {
      this.removeCustomModel(modelId);
    }
  }

  addCustomModel(config: CustomModelConfig): void {
    if (!config.modelId.trim()) return;

    const exists = this._models().some((model) => model.id === config.modelId);
    if (exists) return;

    const customModel: Omit<ModelInfo, 'isDownloaded'> = {
      id: config.modelId,
      name: config.modelId.split('/').pop() || config.modelId,
      size: 'Desconocido',
      description: config.modelUrl
        ? `Modelo personalizado · ${config.modelUrl}`
        : 'Modelo personalizado del registro MLC-AI.',
      isCustom: true,
    };

    const customModels = this.getCustomModelsFromStorage();
    writeJson(CUSTOM_MODELS_KEY, [...customModels, customModel]);

    this._models.update((models) => [...models, { ...customModel, isDownloaded: false }]);
  }

  getAvailableModelsFromRegistry(): string[] {
    return prebuiltAppConfig.model_list.map((model) => model.model_id);
  }

  async getStorageUsage(): Promise<{ used: number; quota: number } | null> {
    if ('storage' in navigator && 'estimate' in navigator.storage) {
      const estimate = await navigator.storage.estimate();
      return {
        used: estimate.usage ?? 0,
        quota: estimate.quota ?? 0,
      };
    }
    return null;
  }

  private async checkModelCached(modelId: string): Promise<boolean> {
    try {
      return await hasModelInCache(modelId);
    } catch {
      return false;
    }
  }

  private removeCustomModel(modelId: string): void {
    const customModels = this.getCustomModelsFromStorage().filter((model) => model.id !== modelId);
    writeJson(CUSTOM_MODELS_KEY, customModels);
    this._models.update((models) => models.filter((model) => model.id !== modelId));
  }

  private getCustomModelsFromStorage(): Omit<ModelInfo, 'isDownloaded'>[] {
    return readJson<Omit<ModelInfo, 'isDownloaded'>[]>(CUSTOM_MODELS_KEY, []);
  }
}
