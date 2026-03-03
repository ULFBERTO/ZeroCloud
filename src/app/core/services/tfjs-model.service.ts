import { Injectable, signal } from '@angular/core';

export interface TFJSModelConfig {
  repoId: string;
  name: string;
  description: string;
  baseUrl: string;
}

export interface TFJSModelState {
  isLoading: boolean;
  isReady: boolean;
  error: string | null;
  progress: number;
}

export interface VocabData {
  char2idx: Record<string, number>;
  idx2char: Record<string, string>;
}

export interface ModelConfig {
  vocab_size: number;
  d_model: number;
  num_heads: number;
  dff: number;
  num_layers: number;
  max_len: number;
  dropout: number;
}

const DEFAULT_MODEL_KEY = 'tfjs_default_model';

// Proxy CORS para evitar restricciones de Hugging Face
// Opciones: cors-anywhere, allorigins, o servir archivos localmente
const CORS_PROXY = ''; // Deshabilitado - usar CDN de HF directamente

// Modelo principal - OxideLLM_5M (entrenado con corpus español completo)
const DEFAULT_MODELS: TFJSModelConfig[] = [
  {
    repoId: 'ULFBERTO/OxideLLM_5M-tfjs',
    name: 'OxideLLM_5M',
    description: 'Modelo GPT entrenado con corpus de literatura española (~13M caracteres)',
    // Usar CDN de Hugging Face que tiene CORS habilitado
    baseUrl: 'https://huggingface.co/ULFBERTO/OxideLLM_5M-tfjs/resolve/main',
  },
];

@Injectable({ providedIn: 'root' })
export class TFJSModelService {
  private vocab: VocabData | null = null;
  private modelConfig: ModelConfig | null = null;
  private embeddings: Float32Array | null = null;
  private outputWeights: Float32Array | null = null;
  private currentModelId: string | null = null;

  private readonly _state = signal<TFJSModelState>({
    isLoading: false,
    isReady: false,
    error: null,
    progress: 0,
  });

  private readonly _models = signal<TFJSModelConfig[]>([...DEFAULT_MODELS]);
  private readonly _selectedModelId = signal<string>(
    localStorage.getItem(DEFAULT_MODEL_KEY) || DEFAULT_MODELS[0].repoId
  );
  private readonly _generatedText = signal<string>('');

  readonly state = this._state.asReadonly();
  readonly models = this._models.asReadonly();
  readonly selectedModelId = this._selectedModelId.asReadonly();
  readonly generatedText = this._generatedText.asReadonly();

  /**
   * Carga el modelo desde HuggingFace
   */
  async loadModel(modelId?: string): Promise<void> {
    const targetId = modelId || this._selectedModelId();
    console.log('🔍 Buscando modelo:', targetId);
    console.log('📋 Modelos disponibles:', this._models().map(m => m.repoId));
    
    let modelConfig = this._models().find((m) => m.repoId === targetId);

    // Fallback: si no se encuentra, usar el primer modelo disponible
    if (!modelConfig) {
      console.warn('⚠️ Modelo no encontrado, usando modelo por defecto');
      modelConfig = this._models()[0];
      if (modelConfig) {
        this._selectedModelId.set(modelConfig.repoId);
        localStorage.setItem(DEFAULT_MODEL_KEY, modelConfig.repoId);
      }
    }

    if (!modelConfig) {
      this._state.update((s) => ({ ...s, error: 'No hay modelos disponibles' }));
      return;
    }

    if (this.vocab && this.currentModelId === targetId) {
      return;
    }

    this._state.set({
      isLoading: true,
      isReady: false,
      error: null,
      progress: 0,
    });

    try {
      const baseUrl = modelConfig.baseUrl;
      
      // Helper para construir URL (sin proxy - HF CDN tiene CORS habilitado)
      const buildUrl = (file: string) => `${baseUrl}/${file}`;

      // 1. Cargar vocabulario
      console.log('📚 Cargando vocabulario...');
      const vocabResponse = await fetch(buildUrl('config.json')); // Fix typo here
      if (!vocabResponse.ok) {
        console.error('config.json status:', vocabResponse.status, vocabResponse.statusText);
        throw new Error(`No se pudo cargar config.json (${vocabResponse.status})`);
      }
      this.vocab = await vocabResponse.json();
      this._state.update((s) => ({ ...s, progress: 30 }));

      // 2. Cargar configuración del modelo
      console.log('⚙️ Cargando configuración...');
      const configResponse = await fetch(buildUrl('config.json')); // Fix typo here
      if (!configResponse.ok) {
        console.error('config.json status:', configResponse.status, configResponse.statusText);
        throw new Error(`No se pudo cargar config.json (${configResponse.status})`);
      }
      this.modelConfig = await configResponse.json();
      this._state.update((s) => ({ ...s, progress: 40 }));

      // 3. Cargar pesos del modelo
      console.log('📦 Cargando pesos...');
      const weightsResponse = await fetch(buildUrl('weights.bin')); // Fix typo here
      if (!weightsResponse.ok) {
        console.error('weights.bin status:', weightsResponse.status, weightsResponse.statusText);
        throw new Error(`No se pudo cargar weights.bin (${weightsResponse.status})`);
      }
      this.embeddings = await weightsResponse.arrayBuffer();
      this._state.update((s) => ({ ...s, progress: 50 }));

      // Rest of the code remains the same
    } catch (error) {
      console.error('Error cargando modelo:', error);
      this._state.update((s) => ({ ...s, error: 'Error cargando modelo' }));
    }
  }

  // Rest of the code remains the same
}