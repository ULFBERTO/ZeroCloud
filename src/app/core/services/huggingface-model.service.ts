import { Injectable, signal } from '@angular/core';

export interface HFModelInfo {
  repoId: string;
  name: string;
  description: string;
  size?: string;
  downloads?: number;
  isDefault?: boolean;
}

export interface DownloadProgress {
  repoId: string;
  progress: number;
  status: 'downloading' | 'completed' | 'error';
  message: string;
}

const STORAGE_KEY = 'hf_models_list';
const DEFAULT_MODEL_KEY = 'hf_default_model';

// Modelo principal por defecto
const DEFAULT_MODELS: HFModelInfo[] = [
  {
    repoId: 'ULFBERTO/gpt-don-quijote',
    name: 'GPT OxideLLM_5M',
    description: 'Modelo GPT entrenado con el texto de OxideLLM_5M de la Mancha',
    size: '~50MB',
    isDefault: true,
  },
];

@Injectable({ providedIn: 'root' })
export class HuggingFaceModelService {
  private readonly _models = signal<HFModelInfo[]>([]);
  private readonly _defaultModelId = signal<string>(
    localStorage.getItem(DEFAULT_MODEL_KEY) || DEFAULT_MODELS[0].repoId
  );
  private readonly _downloadProgress = signal<DownloadProgress | null>(null);
  private readonly _isLoading = signal(false);

  readonly models = this._models.asReadonly();
  readonly defaultModelId = this._defaultModelId.asReadonly();
  readonly downloadProgress = this._downloadProgress.asReadonly();
  readonly isLoading = this._isLoading.asReadonly();

  constructor() {
    this.loadModels();
  }

  private loadModels(): void {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        const customModels = JSON.parse(stored) as HFModelInfo[];
        this._models.set([...DEFAULT_MODELS, ...customModels]);
      } catch {
        this._models.set([...DEFAULT_MODELS]);
      }
    } else {
      this._models.set([...DEFAULT_MODELS]);
    }
  }

  private saveModels(): void {
    const customModels = this._models().filter(
      (m) => !DEFAULT_MODELS.some((d) => d.repoId === m.repoId)
    );
    localStorage.setItem(STORAGE_KEY, JSON.stringify(customModels));
  }

  /**
   * Agrega un modelo de HuggingFace a la lista
   */
  addModel(model: HFModelInfo): void {
    const exists = this._models().some((m) => m.repoId === model.repoId);
    if (!exists) {
      this._models.update((models) => [...models, model]);
      this.saveModels();
    }
  }

  /**
   * Elimina un modelo de la lista (no elimina archivos descargados)
   */
  removeModel(repoId: string): void {
    // No permitir eliminar modelos por defecto
    if (DEFAULT_MODELS.some((m) => m.repoId === repoId)) {
      return;
    }
    this._models.update((models) => models.filter((m) => m.repoId !== repoId));
    this.saveModels();
  }

  /**
   * Establece el modelo por defecto
   */
  setDefaultModel(repoId: string): void {
    this._defaultModelId.set(repoId);
    localStorage.setItem(DEFAULT_MODEL_KEY, repoId);
  }

  /**
   * Genera el comando git clone para descargar un modelo
   */
  getCloneCommand(repoId: string): string {
    return `git clone https://huggingface.co/${repoId}`;
  }

  /**
   * Genera la URL del modelo en HuggingFace
   */
  getModelUrl(repoId: string): string {
    return `https://huggingface.co/${repoId}`;
  }

  /**
   * Genera el código Python para descargar el modelo
   */
  getPythonDownloadCode(repoId: string): string {
    return `from huggingface_hub import snapshot_download

# Descargar modelo
model_path = snapshot_download(repo_id="${repoId}")
print(f"Modelo descargado en: {model_path}")`;
  }

  /**
   * Busca modelos en HuggingFace Hub (requiere API)
   */
  async searchModels(query: string): Promise<HFModelInfo[]> {
    this._isLoading.set(true);
    try {
      const response = await fetch(
        `https://huggingface.co/api/models?search=${encodeURIComponent(query)}&limit=10`
      );
      
      if (!response.ok) {
        throw new Error('Error al buscar modelos');
      }

      const data = await response.json();
      return data.map((model: { modelId?: string; id?: string; downloads?: number; tags?: string[] }) => ({
        repoId: model.modelId || model.id,
        name: (model.modelId || model.id || '').split('/').pop() || 'Unknown',
        description: `Downloads: ${model.downloads || 0}`,
        downloads: model.downloads,
      }));
    } catch (error) {
      console.error('Error searching models:', error);
      return [];
    } finally {
      this._isLoading.set(false);
    }
  }

  /**
   * Obtiene información de un modelo específico
   */
  async getModelInfo(repoId: string): Promise<HFModelInfo | null> {
    try {
      const response = await fetch(`https://huggingface.co/api/models/${repoId}`);
      
      if (!response.ok) {
        return null;
      }

      const data = await response.json();
      return {
        repoId: data.modelId || data.id,
        name: (data.modelId || data.id || '').split('/').pop() || 'Unknown',
        description: data.description || data.cardData?.description || 'Sin descripción',
        downloads: data.downloads,
      };
    } catch {
      return null;
    }
  }

  /**
   * Obtiene el modelo por defecto
   */
  getDefaultModel(): HFModelInfo | undefined {
    return this._models().find((m) => m.repoId === this._defaultModelId());
  }
}
