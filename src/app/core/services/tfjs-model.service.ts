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
      const vocabResponse = await fetch(buildUrl('vocab.json'));
      if (!vocabResponse.ok) {
        console.error('vocab.json status:', vocabResponse.status, vocabResponse.statusText);
        throw new Error(`No se pudo cargar vocab.json (${vocabResponse.status})`);
      }
      this.vocab = await vocabResponse.json();
      this._state.update((s) => ({ ...s, progress: 30 }));

      // 2. Cargar configuración del modelo
      console.log('⚙️ Cargando configuración...');
      const configResponse = await fetch(buildUrl('config.json'));
      if (!configResponse.ok) {
        console.error('config.json status:', configResponse.status, configResponse.statusText);
        throw new Error(`No se pudo cargar config.json (${configResponse.status})`);
      }
      this.modelConfig = await configResponse.json();
      this._state.update((s) => ({ ...s, progress: 50 }));

      // 3. Cargar pesos (solo embeddings para generación simple)
      console.log('🧠 Cargando pesos...');
      const weightsResponse = await fetch(buildUrl('weights.bin'));
      if (!weightsResponse.ok) {
        console.error('weights.bin status:', weightsResponse.status, weightsResponse.statusText);
        throw new Error(`No se pudo cargar weights.bin (${weightsResponse.status})`);
      }
      const weightsBuffer = await weightsResponse.arrayBuffer();
      this._state.update((s) => ({ ...s, progress: 90 }));

      // Parsear embeddings (primeros vocab_size * d_model floats)
      const allWeights = new Float32Array(weightsBuffer);
      const embSize = this.modelConfig!.vocab_size * this.modelConfig!.d_model;
      this.embeddings = allWeights.slice(0, embSize);
      
      // Output weights (últimos vocab_size * d_model floats aproximadamente)
      const totalWeights = allWeights.length;
      this.outputWeights = allWeights.slice(totalWeights - embSize);

      this.currentModelId = targetId;
      this._selectedModelId.set(targetId);
      localStorage.setItem(DEFAULT_MODEL_KEY, targetId);

      this._state.set({
        isLoading: false,
        isReady: true,
        error: null,
        progress: 100,
      });

      console.log('✅ Modelo cargado correctamente');
      console.log(`   Vocabulario: ${Object.keys(this.vocab!.char2idx).length} caracteres`);
      console.log(`   Embeddings: ${this.embeddings.length} valores`);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Error desconocido';
      this._state.set({
        isLoading: false,
        isReady: false,
        error: `Error al cargar modelo: ${message}`,
        progress: 0,
      });
      console.error('❌ Error cargando modelo:', error);
    }
  }

  /**
   * Genera texto usando los embeddings del modelo
   */
  async generate(prompt: string, maxLength: number = 200, temperature: number = 0.7): Promise<string> {
    if (!this.vocab || !this.embeddings || !this.modelConfig) {
      throw new Error('Modelo no cargado');
    }

    const { char2idx, idx2char } = this.vocab;
    const { vocab_size, d_model } = this.modelConfig;

    let generated = prompt;
    this._generatedText.set(generated);

    for (let i = 0; i < maxLength; i++) {
      // Obtener últimos caracteres del contexto
      const context = generated.slice(-20);
      
      // Calcular embedding promedio del contexto
      const contextEmbedding = new Float32Array(d_model).fill(0);
      let validChars = 0;
      
      for (const char of context) {
        const idx = char2idx[char];
        if (idx !== undefined) {
          for (let j = 0; j < d_model; j++) {
            contextEmbedding[j] += this.embeddings[idx * d_model + j];
          }
          validChars++;
        }
      }
      
      if (validChars > 0) {
        for (let j = 0; j < d_model; j++) {
          contextEmbedding[j] /= validChars;
        }
      }

      // Calcular logits usando output weights
      const logits = new Float32Array(vocab_size);
      for (let v = 0; v < vocab_size; v++) {
        let dot = 0;
        for (let j = 0; j < d_model; j++) {
          dot += contextEmbedding[j] * this.outputWeights![v * d_model + j];
        }
        logits[v] = dot;
      }

      // Aplicar temperatura y softmax
      const nextIdx = this.sampleFromLogits(logits, temperature);
      const nextChar = idx2char[nextIdx.toString()] || ' ';

      generated += nextChar;
      this._generatedText.set(generated);

      // Pausa para no bloquear UI
      if (i % 5 === 0) {
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }

    return generated;
  }

  private sampleFromLogits(logits: Float32Array, temperature: number): number {
    // Aplicar temperatura
    const scaled = new Float32Array(logits.length);
    let maxLogit = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      scaled[i] = logits[i] / temperature;
      if (scaled[i] > maxLogit) maxLogit = scaled[i];
    }

    // Softmax
    let sumExp = 0;
    const probs = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
      probs[i] = Math.exp(scaled[i] - maxLogit);
      sumExp += probs[i];
    }
    for (let i = 0; i < probs.length; i++) {
      probs[i] /= sumExp;
    }

    // Muestrear
    const random = Math.random();
    let cumulative = 0;
    for (let i = 0; i < probs.length; i++) {
      cumulative += probs[i];
      if (random < cumulative) return i;
    }
    return probs.length - 1;
  }

  getSelectedModel(): TFJSModelConfig | undefined {
    return this._models().find((m) => m.repoId === this._selectedModelId());
  }

  dispose(): void {
    this.vocab = null;
    this.embeddings = null;
    this.outputWeights = null;
    this.currentModelId = null;
    this._state.set({ isLoading: false, isReady: false, error: null, progress: 0 });
  }
}
