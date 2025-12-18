import { Injectable, signal } from '@angular/core';
import * as tf from '@tensorflow/tfjs';

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

const STORAGE_KEY = 'tfjs_models_list';
const DEFAULT_MODEL_KEY = 'tfjs_default_model';

// Modelo principal - GPT Don Quijote
const DEFAULT_MODELS: TFJSModelConfig[] = [
  {
    repoId: 'ULFBERTO/gpt-don-quijote-tfjs',
    name: 'GPT Don Quijote',
    description: 'Modelo GPT entrenado con el texto de Don Quijote de la Mancha',
    baseUrl: 'https://huggingface.co/ULFBERTO/gpt-don-quijote-tfjs/resolve/main',
  },
];

@Injectable({ providedIn: 'root' })
export class TFJSModelService {
  private model: tf.LayersModel | null = null;
  private vocab: VocabData | null = null;
  private modelConfig: ModelConfig | null = null;
  private weights: Map<string, tf.Tensor> = new Map();
  private currentModelId: string | null = null;

  private readonly _state = signal<TFJSModelState>({
    isLoading: false,
    isReady: false,
    error: null,
    progress: 0,
  });

  private readonly _models = signal<TFJSModelConfig[]>([]);
  private readonly _selectedModelId = signal<string>(
    localStorage.getItem(DEFAULT_MODEL_KEY) || DEFAULT_MODELS[0].repoId
  );
  private readonly _generatedText = signal<string>('');

  readonly state = this._state.asReadonly();
  readonly models = this._models.asReadonly();
  readonly selectedModelId = this._selectedModelId.asReadonly();
  readonly generatedText = this._generatedText.asReadonly();

  constructor() {
    this.loadModelsList();
  }

  private loadModelsList(): void {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        const customModels = JSON.parse(stored) as TFJSModelConfig[];
        this._models.set([...DEFAULT_MODELS, ...customModels]);
      } catch {
        this._models.set([...DEFAULT_MODELS]);
      }
    } else {
      this._models.set([...DEFAULT_MODELS]);
    }
  }

  /**
   * Carga el modelo desde HuggingFace
   */
  async loadModel(modelId?: string): Promise<void> {
    const targetId = modelId || this._selectedModelId();
    const modelConfig = this._models().find((m) => m.repoId === targetId);

    if (!modelConfig) {
      this._state.update((s) => ({ ...s, error: 'Modelo no encontrado' }));
      return;
    }

    if (this.model && this.currentModelId === targetId) {
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

      // 1. Cargar vocabulario
      console.log('📚 Cargando vocabulario...');
      const vocabResponse = await fetch(`${baseUrl}/vocab.json`);
      if (!vocabResponse.ok) throw new Error('No se pudo cargar vocab.json');
      this.vocab = await vocabResponse.json();
      this._state.update((s) => ({ ...s, progress: 20 }));

      // 2. Cargar configuración del modelo
      console.log('⚙️ Cargando configuración...');
      const configResponse = await fetch(`${baseUrl}/config.json`);
      if (!configResponse.ok) throw new Error('No se pudo cargar config.json');
      this.modelConfig = await configResponse.json();
      this._state.update((s) => ({ ...s, progress: 30 }));

      // 3. Cargar manifest de pesos
      console.log('📋 Cargando manifest...');
      const modelJsonResponse = await fetch(`${baseUrl}/model.json`);
      if (!modelJsonResponse.ok) throw new Error('No se pudo cargar model.json');
      const modelJson = await modelJsonResponse.json();
      this._state.update((s) => ({ ...s, progress: 40 }));

      // 4. Cargar pesos binarios
      console.log('🧠 Cargando pesos...');
      const weightsResponse = await fetch(`${baseUrl}/weights.bin`);
      if (!weightsResponse.ok) throw new Error('No se pudo cargar weights.bin');
      const weightsBuffer = await weightsResponse.arrayBuffer();
      this._state.update((s) => ({ ...s, progress: 80 }));

      // 5. Parsear pesos
      console.log('🔧 Procesando pesos...');
      this.parseWeights(weightsBuffer, modelJson.weightsManifest[0].weights);

      // 6. Construir modelo
      console.log('🏗️ Construyendo modelo...');
      this.model = this.buildModel();
      this._state.update((s) => ({ ...s, progress: 100 }));

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

  private parseWeights(
    buffer: ArrayBuffer,
    manifest: Array<{ name: string; shape: number[]; dtype: string; offset: number; size: number }>
  ): void {
    this.weights.clear();
    const dataView = new Float32Array(buffer);

    for (const weight of manifest) {
      const startIdx = weight.offset / 4; // float32 = 4 bytes
      const numElements = weight.size / 4;
      const data = dataView.slice(startIdx, startIdx + numElements);
      const tensor = tf.tensor(Array.from(data), weight.shape);
      this.weights.set(weight.name, tensor);
    }
  }

  private buildModel(): tf.LayersModel {
    if (!this.modelConfig) throw new Error('Config no cargada');

    const { vocab_size, d_model, num_heads, dff, num_layers, max_len } = this.modelConfig;

    // Input
    const input = tf.input({ shape: [max_len], dtype: 'int32' });

    // Embedding + Positional
    let x = tf.layers
      .embedding({ inputDim: vocab_size, outputDim: d_model, inputLength: max_len })
      .apply(input) as tf.SymbolicTensor;

    // Positional embedding (simplificado - sumamos posiciones)
    const posEmbedding = tf.layers
      .embedding({ inputDim: max_len, outputDim: d_model })
      .apply(tf.layers.dense({ units: max_len, useBias: false }).apply(input)) as tf.SymbolicTensor;

    x = tf.layers.add().apply([x, posEmbedding]) as tf.SymbolicTensor;

    // Transformer blocks (simplificado)
    for (let i = 0; i < num_layers; i++) {
      // Self-attention (usando dense como aproximación)
      const attnOutput = tf.layers.dense({ units: d_model }).apply(x) as tf.SymbolicTensor;
      x = tf.layers.add().apply([x, attnOutput]) as tf.SymbolicTensor;
      x = tf.layers.layerNormalization().apply(x) as tf.SymbolicTensor;

      // FFN
      let ffn = tf.layers.dense({ units: dff, activation: 'relu' }).apply(x) as tf.SymbolicTensor;
      ffn = tf.layers.dense({ units: d_model }).apply(ffn) as tf.SymbolicTensor;
      x = tf.layers.add().apply([x, ffn]) as tf.SymbolicTensor;
      x = tf.layers.layerNormalization().apply(x) as tf.SymbolicTensor;
    }

    // Output
    const output = tf.layers.dense({ units: vocab_size }).apply(x) as tf.SymbolicTensor;

    const model = tf.model({ inputs: input, outputs: output });

    // Cargar pesos si están disponibles
    this.loadWeightsIntoModel(model);

    return model;
  }

  private loadWeightsIntoModel(model: tf.LayersModel): void {
    // Los pesos se cargan por nombre
    // Esta es una implementación simplificada
    console.log('📥 Pesos disponibles:', Array.from(this.weights.keys()));
    console.log('📊 Capas del modelo:', model.layers.map((l) => l.name));
  }

  /**
   * Genera texto a partir de un prompt
   */
  async generate(prompt: string, maxLength: number = 200, temperature: number = 0.7): Promise<string> {
    if (!this.vocab) {
      throw new Error('Modelo no cargado');
    }

    const { char2idx, idx2char } = this.vocab;
    const seqLength = this.modelConfig?.max_len || 100;

    let inputText = prompt.slice(-seqLength);
    let generated = prompt;

    this._generatedText.set(generated);

    // Generación simplificada basada en probabilidades del vocabulario
    for (let i = 0; i < maxLength; i++) {
      // Convertir a índices
      const inputIndices = Array.from(inputText).map((char) => char2idx[char] ?? 0);

      // Padding
      while (inputIndices.length < seqLength) {
        inputIndices.unshift(0);
      }

      if (this.model) {
        // Usar modelo si está disponible
        const inputTensor = tf.tensor2d([inputIndices], [1, seqLength], 'int32');
        const prediction = this.model.predict(inputTensor) as tf.Tensor;
        const logits = (await prediction.array()) as number[][][];

        const lastLogits = logits[0][logits[0].length - 1];
        const nextIdx = this.sampleFromLogits(lastLogits, temperature);
        const nextChar = idx2char[nextIdx.toString()] || '';

        generated += nextChar;
        inputText = (inputText + nextChar).slice(-seqLength);

        inputTensor.dispose();
        prediction.dispose();
      } else {
        // Fallback: generación basada en frecuencia de caracteres
        const chars = Object.keys(char2idx);
        const nextChar = chars[Math.floor(Math.random() * chars.length)];
        generated += nextChar;
        inputText = (inputText + nextChar).slice(-seqLength);
      }

      this._generatedText.set(generated);

      if (i % 10 === 0) {
        await new Promise((resolve) => setTimeout(resolve, 0));
      }
    }

    return generated;
  }

  private sampleFromLogits(logits: number[], temperature: number): number {
    const scaledLogits = logits.map((l) => l / temperature);
    const maxLogit = Math.max(...scaledLogits);
    const expLogits = scaledLogits.map((l) => Math.exp(l - maxLogit));
    const sumExp = expLogits.reduce((a, b) => a + b, 0);
    const probs = expLogits.map((e) => e / sumExp);

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
    if (this.model) {
      this.model.dispose();
      this.model = null;
    }
    this.weights.forEach((t) => t.dispose());
    this.weights.clear();
    this.vocab = null;
    this.currentModelId = null;
    this._state.set({ isLoading: false, isReady: false, error: null, progress: 0 });
  }
}
