import { Injectable, computed, signal } from '@angular/core';
import { DEFAULT_HF_MODELS } from '../config/model-catalog';
import { HFModelInfo } from '../models/model.model';
import { readJson, readStorage, writeJson, writeStorage } from '../utils/storage';

const STORAGE_KEY = 'hf_models_list';
const DEFAULT_MODEL_KEY = 'hf_default_model';

@Injectable({ providedIn: 'root' })
export class HuggingFaceModelService {
  private readonly _models = signal<HFModelInfo[]>([]);
  private readonly _defaultModelId = signal<string>(
    readStorage(DEFAULT_MODEL_KEY) ?? DEFAULT_HF_MODELS[0].repoId,
  );
  private readonly _isLoading = signal(false);
  private readonly _searchError = signal<string | null>(null);

  readonly models = this._models.asReadonly();
  readonly defaultModelId = this._defaultModelId.asReadonly();
  readonly isLoading = this._isLoading.asReadonly();
  readonly searchError = this._searchError.asReadonly();

  readonly defaultModel = computed(() =>
    this._models().find((model) => model.repoId === this._defaultModelId()),
  );

  constructor() {
    this.loadModels();
  }

  addModel(model: HFModelInfo): void {
    const exists = this._models().some((item) => item.repoId === model.repoId);
    if (exists) return;

    this._models.update((models) => [...models, model]);
    this.saveModels();
  }

  removeModel(repoId: string): void {
    if (DEFAULT_HF_MODELS.some((model) => model.repoId === repoId)) return;

    this._models.update((models) => models.filter((model) => model.repoId !== repoId));
    if (this._defaultModelId() === repoId) {
      this.setDefaultModel(DEFAULT_HF_MODELS[0].repoId);
    }
    this.saveModels();
  }

  setDefaultModel(repoId: string): void {
    if (!this._models().some((model) => model.repoId === repoId)) return;
    this._defaultModelId.set(repoId);
    writeStorage(DEFAULT_MODEL_KEY, repoId);
  }

  getCloneCommand(repoId: string): string {
    return `git clone https://huggingface.co/${repoId}`;
  }

  getModelUrl(repoId: string): string {
    return `https://huggingface.co/${repoId}`;
  }

  getPythonDownloadCode(repoId: string): string {
    return `from huggingface_hub import snapshot_download

model_path = snapshot_download(repo_id="${repoId}")
print(f"Modelo descargado en: {model_path}")`;
  }

  async searchModels(query: string): Promise<HFModelInfo[]> {
    const term = query.trim();
    if (!term) return [];

    this._isLoading.set(true);
    this._searchError.set(null);

    try {
      const response = await fetch(
        `https://huggingface.co/api/models?search=${encodeURIComponent(term)}&limit=10`,
      );

      if (!response.ok) {
        throw new Error(`Hugging Face respondió ${response.status}`);
      }

      const data = (await response.json()) as Array<{
        modelId?: string;
        id?: string;
        downloads?: number;
        tags?: string[];
      }>;

      return data.map((model) => {
        const repoId = model.modelId ?? model.id ?? '';
        return {
          repoId,
          name: repoId.split('/').pop() || repoId,
          description: `${(model.downloads ?? 0).toLocaleString('es-ES')} descargas · ${(model.tags ?? [])
            .slice(0, 3)
            .join(', ') || 'sin tags'}`,
          downloads: model.downloads,
        };
      });
    } catch (error) {
      this._searchError.set(
        error instanceof Error ? error.message : 'No se pudo conectar con Hugging Face.',
      );
      return [];
    } finally {
      this._isLoading.set(false);
    }
  }

  async getModelInfo(repoId: string): Promise<HFModelInfo | null> {
    try {
      const response = await fetch(`https://huggingface.co/api/models/${repoId}`);
      if (!response.ok) return null;

      const data = await response.json();
      return {
        repoId: data.modelId ?? data.id,
        name: (data.modelId ?? data.id ?? '').split('/').pop() || 'Desconocido',
        description: data.cardData?.description ?? 'Sin descripción',
        downloads: data.downloads,
      };
    } catch {
      return null;
    }
  }

  getDefaultModel(): HFModelInfo | undefined {
    return this.defaultModel();
  }

  private loadModels(): void {
    const customModels = readJson<HFModelInfo[]>(STORAGE_KEY, []);
    const customIds = new Set(customModels.map((model) => model.repoId));
    const baseModels = DEFAULT_HF_MODELS.filter((model) => !customIds.has(model.repoId));
    this._models.set([...baseModels, ...customModels]);

    if (!this._models().some((model) => model.repoId === this._defaultModelId())) {
      this._defaultModelId.set(DEFAULT_HF_MODELS[0].repoId);
    }
  }

  private saveModels(): void {
    const customModels = this._models().filter(
      (model) => !DEFAULT_HF_MODELS.some((defaultModel) => defaultModel.repoId === model.repoId),
    );
    writeJson(STORAGE_KEY, customModels);
  }
}
