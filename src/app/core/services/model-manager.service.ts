import { Injectable, signal, computed } from '@angular/core';
import { prebuiltAppConfig, deleteModelAllInfoInCache, hasModelInCache } from '@mlc-ai/web-llm';

export interface ModelInfo {
  id: string;
  name: string;
  size: string;
  description: string;
  isDownloaded: boolean;
  isCustom: boolean;
  vramRequired?: string;
}

export interface CustomModelConfig {
  modelId: string;
  modelUrl: string;
  wasmUrl?: string;
}

const STORAGE_KEY = 'webllm_selected_model';
const CUSTOM_MODELS_KEY = 'webllm_custom_models';

// Modelos recomendados para navegador (ligeros)
const RECOMMENDED_MODELS: Omit<ModelInfo, 'isDownloaded'>[] = [
  {
    id: 'Llama-3.2-1B-Instruct-q4f16_1-MLC',
    name: 'Llama 3.2 1B',
    size: '~700MB',
    description: 'Modelo compacto y rápido. Ideal para empezar.',
    isCustom: false,
    vramRequired: '2GB',
  },
  {
    id: 'Llama-3.2-3B-Instruct-q4f16_1-MLC',
    name: 'Llama 3.2 3B',
    size: '~1.8GB',
    description: 'Balance entre velocidad y calidad.',
    isCustom: false,
    vramRequired: '4GB',
  },
  {
    id: 'Phi-3.5-mini-instruct-q4f16_1-MLC',
    name: 'Phi 3.5 Mini',
    size: '~2.2GB',
    description: 'Modelo de Microsoft, excelente razonamiento.',
    isCustom: false,
    vramRequired: '4GB',
  },
  {
    id: 'Qwen2.5-1.5B-Instruct-q4f16_1-MLC',
    name: 'Qwen 2.5 1.5B',
    size: '~1GB',
    description: 'Modelo de Alibaba, buen soporte multilingüe.',
    isCustom: false,
    vramRequired: '3GB',
  },
  {
    id: 'SmolLM2-1.7B-Instruct-q4f16_1-MLC',
    name: 'SmolLM2 1.7B',
    size: '~1GB',
    description: 'Modelo ultra ligero de Hugging Face.',
    isCustom: false,
    vramRequired: '2GB',
  },
];


@Injectable({ providedIn: 'root' })
export class ModelManagerService {
  private readonly _models = signal<ModelInfo[]>([]);
  private readonly _selectedModelId = signal<string>(
    localStorage.getItem(STORAGE_KEY) || RECOMMENDED_MODELS[0].id
  );
  private readonly _isChecking = signal(false);

  readonly models = this._models.asReadonly();
  readonly selectedModelId = this._selectedModelId.asReadonly();
  readonly isChecking = this._isChecking.asReadonly();

  readonly selectedModel = computed(() =>
    this._models().find((m) => m.id === this._selectedModelId())
  );

  readonly downloadedModels = computed(() =>
    this._models().filter((m) => m.isDownloaded)
  );

  constructor() {
    this.loadModels();
  }

  async loadModels(): Promise<void> {
    this._isChecking.set(true);

    const customModels = this.getCustomModelsFromStorage();
    const allModelConfigs = [...RECOMMENDED_MODELS, ...customModels];

    const modelsWithStatus = await Promise.all(
      allModelConfigs.map(async (model) => ({
        ...model,
        isDownloaded: await this.checkModelCached(model.id),
      }))
    );

    this._models.set(modelsWithStatus);
    this._isChecking.set(false);
  }

  private async checkModelCached(modelId: string): Promise<boolean> {
    try {
      return await hasModelInCache(modelId);
    } catch {
      return false;
    }
  }

  selectModel(modelId: string): void {
    this._selectedModelId.set(modelId);
    localStorage.setItem(STORAGE_KEY, modelId);
  }

  async deleteModel(modelId: string): Promise<void> {
    try {
      await deleteModelAllInfoInCache(modelId);
      this._models.update((models) =>
        models.map((m) => (m.id === modelId ? { ...m, isDownloaded: false } : m))
      );

      // Si es custom, también lo eliminamos de la lista
      const customModels = this.getCustomModelsFromStorage();
      const isCustom = customModels.some((m) => m.id === modelId);
      if (isCustom) {
        this.removeCustomModel(modelId);
      }
    } catch (error) {
      console.error('Error deleting model:', error);
      throw error;
    }
  }


  addCustomModel(config: CustomModelConfig): void {
    const customModel: Omit<ModelInfo, 'isDownloaded'> = {
      id: config.modelId,
      name: config.modelId.split('/').pop() || config.modelId,
      size: 'Desconocido',
      description: `Modelo personalizado: ${config.modelUrl}`,
      isCustom: true,
    };

    const customModels = this.getCustomModelsFromStorage();
    customModels.push(customModel);
    localStorage.setItem(CUSTOM_MODELS_KEY, JSON.stringify(customModels));

    this._models.update((models) => [...models, { ...customModel, isDownloaded: false }]);
  }

  private removeCustomModel(modelId: string): void {
    const customModels = this.getCustomModelsFromStorage().filter((m) => m.id !== modelId);
    localStorage.setItem(CUSTOM_MODELS_KEY, JSON.stringify(customModels));
    this._models.update((models) => models.filter((m) => m.id !== modelId));
  }

  private getCustomModelsFromStorage(): Omit<ModelInfo, 'isDownloaded'>[] {
    try {
      const stored = localStorage.getItem(CUSTOM_MODELS_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch {
      return [];
    }
  }

  getAvailableModelsFromRegistry(): string[] {
    return prebuiltAppConfig.model_list.map((m) => m.model_id);
  }

  async getStorageUsage(): Promise<{ used: number; quota: number } | null> {
    if ('storage' in navigator && 'estimate' in navigator.storage) {
      const estimate = await navigator.storage.estimate();
      return {
        used: estimate.usage || 0,
        quota: estimate.quota || 0,
      };
    }
    return null;
  }
}
