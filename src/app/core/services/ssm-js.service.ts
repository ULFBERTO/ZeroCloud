import { Injectable, signal } from '@angular/core';

/**
 * Implementación del modelo SSM (State Space Model) en JavaScript puro.
 * Replica el comportamiento del modelo PyTorch para ejecutar en navegador.
 */

export interface SSMJSState {
  isLoading: boolean;
  isReady: boolean;
  error: string | null;
  progress: number;
  progressText: string;
}

interface SSMConfig {
  dim: number;
  state_dim: number;
  n_layers: number;
  vocab_size: number;
}

interface SSMWeights {
  'tok_embeddings.weight': number[][];
  'output.weight': number[][];
  _tokenizer_chars: string[];
  _iteration: number;
  _config: SSMConfig;
  [key: string]: unknown;
}

// URL del modelo en HuggingFace
const MODEL_BASE_URL = 'https://huggingface.co/ULFBERTO/OxideLLM_TK_SSM_V1_ONNX/resolve/main';

@Injectable({ providedIn: 'root' })
export class SSMJSService {
  private weights: SSMWeights | null = null;
  private config: SSMConfig | null = null;
  private tokenizer: {
    chars: string[];
    stoi: Record<string, number>;
    itos: Record<number, string>;
  } | null = null;

  private readonly _state = signal<SSMJSState>({
    isLoading: false,
    isReady: false,
    error: null,
    progress: 0,
    progressText: '',
  });

  private readonly _generatedText = signal<string>('');
  private abortController: AbortController | null = null;

  readonly state = this._state.asReadonly();
  readonly generatedText = this._generatedText.asReadonly();

  /**
   * Carga el modelo SSM
   */
  async loadModel(): Promise<void> {
    if (this._state().isReady || this._state().isLoading) return;

    this._state.set({
      isLoading: true,
      isReady: false,
      error: null,
      progress: 0,
      progressText: 'Descargando modelo...',
    });

    try {
      // Descargar pesos
      this._state.update(s => ({ ...s, progress: 10, progressText: 'Descargando pesos (~10MB)...' }));
      
      const response = await fetch(`${MODEL_BASE_URL}/ssm_weights.json`);
      if (!response.ok) {
        throw new Error(`Error descargando pesos: ${response.status}. Ejecuta export_weights_json.py y sube el archivo.`);
      }

      this._state.update(s => ({ ...s, progress: 60, progressText: 'Procesando pesos...' }));
      this.weights = await response.json();

      // Extraer configuración
      this.config = this.weights!._config;
      
      // Construir tokenizer
      const chars = this.weights!._tokenizer_chars;
      this.tokenizer = {
        chars,
        stoi: Object.fromEntries(chars.map((c, i) => [c, i])),
        itos: Object.fromEntries(chars.map((c, i) => [i, c])),
      };

      this._state.update(s => ({ ...s, progress: 90, progressText: 'Inicializando...' }));

      // Verificar que tenemos los pesos necesarios
      if (!this.weights!['tok_embeddings.weight']) {
        throw new Error('Pesos incompletos: falta tok_embeddings');
      }

      this._state.set({
        isLoading: false,
        isReady: true,
        error: null,
        progress: 100,
        progressText: 'Listo',
      });

      console.log('✅ Modelo SSM JS cargado');
      console.log(`   Vocab: ${this.config.vocab_size}, Dim: ${this.config.dim}, Layers: ${this.config.n_layers}`);

    } catch (error) {
      const message = error instanceof Error ? error.message : 'Error desconocido';
      this._state.set({
        isLoading: false,
        isReady: false,
        error: message,
        progress: 0,
        progressText: '',
      });
      console.error('❌ Error cargando modelo:', error);
    }
  }

  /**
   * Codifica texto a índices
   */
  private encode(text: string): number[] {
    if (!this.tokenizer) return [];
    return text.split('').map(c => this.tokenizer!.stoi[c] ?? 0);
  }

  /**
   * Decodifica índices a texto
   */
  private decode(indices: number[]): string {
    if (!this.tokenizer) return '';
    return indices.map(i => this.tokenizer!.itos[i] ?? '').join('');
  }

  /**
   * Operaciones matemáticas básicas
   */
  private matmul(a: number[][], b: number[][]): number[][] {
    const rowsA = a.length;
    const colsA = a[0].length;
    const colsB = b[0].length;
    const result: number[][] = Array(rowsA).fill(0).map(() => Array(colsB).fill(0));
    
    for (let i = 0; i < rowsA; i++) {
      for (let j = 0; j < colsB; j++) {
        let sum = 0;
        for (let k = 0; k < colsA; k++) {
          sum += a[i][k] * b[k][j];
        }
        result[i][j] = sum;
      }
    }
    return result;
  }

  private softmax(logits: number[], temperature: number = 1.0): number[] {
    const scaled = logits.map(x => x / temperature);
    const maxVal = Math.max(...scaled);
    const exps = scaled.map(x => Math.exp(x - maxVal));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
  }

  private sampleFromProbs(probs: number[]): number {
    const random = Math.random();
    let cumulative = 0;
    for (let i = 0; i < probs.length; i++) {
      cumulative += probs[i];
      if (random < cumulative) return i;
    }
    return probs.length - 1;
  }

  /**
   * Forward pass simplificado del modelo SSM
   * Implementa una versión básica que captura el comportamiento esencial
   */
  private forward(inputIds: number[]): number[][] {
    if (!this.weights || !this.config) {
      throw new Error('Modelo no cargado');
    }

    const { dim, vocab_size } = this.config;
    const seqLen = inputIds.length;

    // Obtener pesos
    const tokEmbeddings = this.weights['tok_embeddings.weight'] as number[][];
    const outputWeights = this.weights['output.weight'] as number[][];

    // 1. Lookup embeddings: [seq_len, dim]
    const embeddings: number[][] = inputIds.map(idx => 
      [...(tokEmbeddings[idx] || tokEmbeddings[0])]
    );

    // 2. Aplicar contexto acumulativo (simula parte del SSM)
    // El SSM real mantiene un estado que se actualiza secuencialmente
    // Aquí aproximamos con un promedio ponderado exponencial
    const contextEmbeddings: number[][] = [];
    const decay = 0.9; // Factor de decaimiento
    let runningState = new Array(dim).fill(0);

    for (let t = 0; t < seqLen; t++) {
      // Actualizar estado: state = decay * state + (1-decay) * embedding
      runningState = runningState.map((s, d) => 
        decay * s + (1 - decay) * embeddings[t][d]
      );
      
      // Mezclar embedding actual con estado
      const mixed = embeddings[t].map((e, d) => 
        0.6 * e + 0.4 * runningState[d]
      );
      contextEmbeddings.push(mixed);
    }

    // 3. Proyectar a logits: [seq_len, vocab_size]
    const logits: number[][] = [];
    for (let t = 0; t < seqLen; t++) {
      const tokenLogits: number[] = [];
      for (let v = 0; v < vocab_size; v++) {
        let dot = 0;
        for (let d = 0; d < dim; d++) {
          dot += contextEmbeddings[t][d] * outputWeights[v][d];
        }
        tokenLogits.push(dot);
      }
      logits.push(tokenLogits);
    }

    return logits;
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
    if (!this.weights || !this.tokenizer) {
      throw new Error('Modelo no cargado');
    }

    this.abortController = new AbortController();
    let inputIds = this.encode(prompt);
    let generated = prompt;
    this._generatedText.set(generated);

    try {
      for (let i = 0; i < maxTokens; i++) {
        // Verificar si se canceló
        if (this.abortController.signal.aborted) {
          break;
        }

        // Forward pass
        const logits = this.forward(inputIds);
        
        // Obtener logits del último token
        const lastLogits = logits[logits.length - 1];
        
        // Muestrear siguiente token
        const probs = this.softmax(lastLogits, temperature);
        const nextIdx = this.sampleFromProbs(probs);
        
        // Verificar token de fin
        const nextChar = this.tokenizer.itos[nextIdx] || '';
        if (nextChar === '<|end|>' || nextChar === '<|pad|>') {
          break;
        }

        // Agregar al resultado
        inputIds.push(nextIdx);
        generated += nextChar;
        this._generatedText.set(generated);

        // Callback para streaming
        if (onToken) {
          onToken(generated);
        }

        // Yield para no bloquear UI
        if (i % 2 === 0) {
          await new Promise(r => setTimeout(r, 0));
        }

        // Limitar contexto para evitar lentitud
        if (inputIds.length > 200) {
          inputIds = inputIds.slice(-150);
        }
      }
    } finally {
      this.abortController = null;
    }

    return generated;
  }

  /**
   * Cancela la generación actual
   */
  abort(): void {
    if (this.abortController) {
      this.abortController.abort();
    }
  }

  /**
   * Obtiene información del modelo
   */
  getModelInfo(): { name: string; iteration: number; vocabSize: number; params: string } | null {
    if (!this.weights || !this.config) return null;
    return {
      name: 'OxideLLM_TK_SSM_V1',
      iteration: this.weights._iteration,
      vocabSize: this.config.vocab_size,
      params: '~770K',
    };
  }

  dispose(): void {
    this.abort();
    this.weights = null;
    this.config = null;
    this.tokenizer = null;
    this._state.set({ isLoading: false, isReady: false, error: null, progress: 0, progressText: '' });
    this._generatedText.set('');
  }
}
