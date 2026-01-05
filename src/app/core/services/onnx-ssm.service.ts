import { Injectable, signal } from '@angular/core';

export interface SSMModelState {
  isLoading: boolean;
  isReady: boolean;
  error: string | null;
  progress: number;
}

export interface SSMTokenizer {
  model_name: string;
  iteration: number;
  vocab_size: number;
  char2idx: Record<string, number>;
  idx2char: Record<string, string>;
  special_tokens: string[];
  config: {
    dim: number;
    state_dim: number;
    n_layers: number;
  };
}

// URL base del modelo ONNX en HuggingFace
const MODEL_BASE_URL = 'https://huggingface.co/ULFBERTO/OxideLLM_TK_SSM_V1_ONNX/resolve/main';

@Injectable({ providedIn: 'root' })
export class OnnxSSMService {
  private session: any = null;
  private tokenizer: SSMTokenizer | null = null;
  private ort: any = null;

  private readonly _state = signal<SSMModelState>({
    isLoading: false,
    isReady: false,
    error: null,
    progress: 0,
  });

  private readonly _generatedText = signal<string>('');

  readonly state = this._state.asReadonly();
  readonly generatedText = this._generatedText.asReadonly();

  /**
   * Carga ONNX Runtime Web y el modelo SSM
   */
  async loadModel(): Promise<void> {
    if (this._state().isReady || this._state().isLoading) return;

    this._state.set({
      isLoading: true,
      isReady: false,
      error: null,
      progress: 0,
    });

    try {
      // 1. Cargar ONNX Runtime Web dinámicamente
      this._state.update(s => ({ ...s, progress: 10 }));
      await this.loadOnnxRuntime();

      // 2. Cargar tokenizer
      this._state.update(s => ({ ...s, progress: 30 }));
      await this.loadTokenizer();

      // 3. Cargar modelo ONNX
      this._state.update(s => ({ ...s, progress: 50 }));
      await this.loadOnnxModel();

      this._state.set({
        isLoading: false,
        isReady: true,
        error: null,
        progress: 100,
      });

      console.log('✅ Modelo SSM ONNX cargado correctamente');

    } catch (error) {
      const message = error instanceof Error ? error.message : 'Error desconocido';
      this._state.set({
        isLoading: false,
        isReady: false,
        error: message,
        progress: 0,
      });
      console.error('❌ Error cargando modelo SSM:', error);
    }
  }

  private async loadOnnxRuntime(): Promise<void> {
    if (this.ort) return;

    // Cargar ONNX Runtime Web desde CDN
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/ort.min.js';
      script.onload = () => {
        this.ort = (window as any).ort;
        if (this.ort) {
          console.log('ONNX Runtime Web cargado');
          resolve();
        } else {
          reject(new Error('ONNX Runtime no disponible'));
        }
      };
      script.onerror = () => reject(new Error('Error cargando ONNX Runtime'));
      document.head.appendChild(script);
    });
  }

  private async loadTokenizer(): Promise<void> {
    const response = await fetch(`${MODEL_BASE_URL}/tokenizer.json`);
    if (!response.ok) {
      throw new Error(`Error cargando tokenizer: ${response.status}`);
    }
    this.tokenizer = await response.json();
    console.log(`Tokenizer cargado: ${this.tokenizer!.vocab_size} tokens`);
  }

  private async loadOnnxModel(): Promise<void> {
    const modelUrl = `${MODEL_BASE_URL}/ssm_model.onnx`;
    
    // Configurar opciones de sesión
    const sessionOptions: any = {
      executionProviders: ['wasm'], // WebAssembly backend
      graphOptimizationLevel: 'all',
    };

    // Intentar usar WebGPU si está disponible
    if ('gpu' in navigator) {
      sessionOptions.executionProviders = ['webgpu', 'wasm'];
    }

    this.session = await this.ort.InferenceSession.create(modelUrl, sessionOptions);
    console.log('Modelo ONNX cargado');
  }

  /**
   * Genera texto a partir de un prompt
   */
  async generate(
    prompt: string,
    maxTokens: number = 100,
    temperature: number = 0.8,
    onToken?: (text: string) => void
  ): Promise<string> {
    if (!this.session || !this.tokenizer) {
      throw new Error('Modelo no cargado');
    }

    const { char2idx, idx2char, vocab_size } = this.tokenizer;

    // Codificar prompt
    let inputIds = prompt.split('').map(c => char2idx[c] ?? 0);
    let generated = prompt;
    this._generatedText.set(generated);

    for (let i = 0; i < maxTokens; i++) {
      // Crear tensor de entrada
      const inputTensor = new this.ort.Tensor(
        'int64',
        BigInt64Array.from(inputIds.map(x => BigInt(x))),
        [1, inputIds.length]
      );

      // Inferencia
      const outputs = await this.session.run({ input_ids: inputTensor });
      const logits = outputs.logits.data as Float32Array;

      // Obtener logits del último token
      const lastLogits = new Float32Array(vocab_size);
      const offset = (inputIds.length - 1) * vocab_size;
      for (let j = 0; j < vocab_size; j++) {
        lastLogits[j] = logits[offset + j];
      }

      // Muestrear siguiente token
      const nextIdx = this.sampleFromLogits(lastLogits, temperature);
      const nextChar = idx2char[nextIdx.toString()] || '';

      // Verificar token de fin
      if (nextChar === '<|end|>') break;

      // Agregar al resultado
      inputIds.push(nextIdx);
      generated += nextChar;
      this._generatedText.set(generated);

      // Callback para streaming
      if (onToken) {
        onToken(generated);
      }

      // Yield para no bloquear UI
      if (i % 3 === 0) {
        await new Promise(r => setTimeout(r, 0));
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

  /**
   * Obtiene información del modelo
   */
  getModelInfo(): { name: string; iteration: number; vocabSize: number; params: string } | null {
    if (!this.tokenizer) return null;
    return {
      name: this.tokenizer.model_name,
      iteration: this.tokenizer.iteration,
      vocabSize: this.tokenizer.vocab_size,
      params: '~770K',
    };
  }

  /**
   * Libera recursos
   */
  dispose(): void {
    this.session = null;
    this.tokenizer = null;
    this._state.set({ isLoading: false, isReady: false, error: null, progress: 0 });
    this._generatedText.set('');
  }
}
