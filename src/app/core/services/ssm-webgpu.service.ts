import { Injectable, signal } from '@angular/core';

/**
 * Implementación completa del modelo SSM (State Space Model) en TypeScript.
 * Replica fielmente el comportamiento del modelo PyTorch incluyendo:
 * - RMSNorm
 * - SSMBlock con escaneo secuencial selectivo
 * - Múltiples capas con conexiones residuales
 */

export interface SSMState {
  isLoading: boolean;
  isReady: boolean;
  error: string | null;
  progress: number;
  progressText: string;
}

interface SSMConfig {
  vocab_size: number;
  dim: number;
  n_layers: number;
  state_dim: number;
  inner_dim: number; // expand * dim
}

// URL del modelo en HuggingFace
const MODEL_BASE_URL = 'https://huggingface.co/ULFBERTO/OxideLLM_TK_SSM_V1_ONNX/resolve/main';

@Injectable({ providedIn: 'root' })
export class SSMWebGPUService {
  private weights: Map<string, Float32Array> = new Map();
  private config: SSMConfig | null = null;
  private tokenizer: {
    chars: string[];
    stoi: Map<string, number>;
    itos: Map<number, string>;
  } | null = null;

  private readonly _state = signal<SSMState>({
    isLoading: false,
    isReady: false,
    error: null,
    progress: 0,
    progressText: '',
  });

  private readonly _generatedText = signal<string>('');
  private abortFlag = false;

  readonly state = this._state.asReadonly();
  readonly generatedText = this._generatedText.asReadonly();

  async loadModel(): Promise<void> {
    if (this._state().isReady || this._state().isLoading) return;

    this._state.set({
      isLoading: true,
      isReady: false,
      error: null,
      progress: 0,
      progressText: 'Iniciando...',
    });

    try {
      this._state.update(s => ({ ...s, progress: 5, progressText: 'Descargando pesos...' }));
      
      const response = await fetch(`${MODEL_BASE_URL}/ssm_weights.json`);
      if (!response.ok) {
        throw new Error('Ejecuta export_weights_json.py y sube ssm_weights.json a HuggingFace');
      }

      this._state.update(s => ({ ...s, progress: 40, progressText: 'Procesando pesos...' }));
      const data = await response.json();

      // Extraer configuración
      const configData = data['_config'];
      this.config = {
        vocab_size: configData.vocab_size,
        dim: configData.dim,
        n_layers: configData.n_layers,
        state_dim: configData.state_dim,
        inner_dim: configData.dim * 2, // expand = 2
      };

      // Construir tokenizer
      const chars = data['_tokenizer_chars'] as string[];
      this.tokenizer = {
        chars,
        stoi: new Map(chars.map((c, i) => [c, i])),
        itos: new Map(chars.map((c, i) => [i, c])),
      };

      this._state.update(s => ({ ...s, progress: 60, progressText: 'Convirtiendo pesos...' }));

      // Convertir pesos a Float32Array
      for (const [key, value] of Object.entries(data)) {
        if (key.startsWith('_')) continue;
        const flat = this.flattenArray(value as number[] | number[][] | number[][][]);
        this.weights.set(key, new Float32Array(flat));
      }

      this._state.update(s => ({ ...s, progress: 90, progressText: 'Verificando...' }));

      // Verificar pesos esenciales
      if (!this.weights.has('tok_embeddings.weight')) {
        throw new Error('Faltan pesos: tok_embeddings.weight');
      }
      if (!this.weights.has('output.weight')) {
        throw new Error('Faltan pesos: output.weight');
      }

      this._state.set({
        isLoading: false,
        isReady: true,
        error: null,
        progress: 100,
        progressText: 'Listo',
      });

      console.log('✅ SSM cargado:', this.config);

    } catch (error) {
      const msg = error instanceof Error ? error.message : 'Error desconocido';
      this._state.set({
        isLoading: false,
        isReady: false,
        error: msg,
        progress: 0,
        progressText: '',
      });
    }
  }

  private flattenArray(arr: number[] | number[][] | number[][][]): number[] {
    const result: number[] = [];
    const flatten = (a: unknown): void => {
      if (Array.isArray(a)) {
        for (const item of a) flatten(item);
      } else if (typeof a === 'number') {
        result.push(a);
      }
    };
    flatten(arr);
    return result;
  }

  // ============ Operaciones matemáticas ============

  private rmsNorm(x: Float32Array, weight: Float32Array, eps = 1e-6): Float32Array {
    const dim = weight.length;
    const result = new Float32Array(dim);
    
    // Calcular RMS
    let sumSq = 0;
    for (let i = 0; i < dim; i++) {
      sumSq += x[i] * x[i];
    }
    const rms = Math.sqrt(sumSq / dim + eps);
    
    // Normalizar y escalar
    for (let i = 0; i < dim; i++) {
      result[i] = (x[i] / rms) * weight[i];
    }
    return result;
  }

  private silu(x: number): number {
    return x / (1 + Math.exp(-x));
  }

  private softplus(x: number): number {
    return Math.log(1 + Math.exp(x));
  }

  private linear(input: Float32Array, weight: Float32Array, outDim: number, bias?: Float32Array): Float32Array {
    const inDim = input.length;
    const result = new Float32Array(outDim);
    
    for (let o = 0; o < outDim; o++) {
      let sum = bias ? bias[o] : 0;
      for (let i = 0; i < inDim; i++) {
        sum += input[i] * weight[o * inDim + i];
      }
      result[o] = sum;
    }
    return result;
  }

  // ============ SSM Block ============

  private ssmBlock(
    x: Float32Array[], // [seqLen][dim]
    layerIdx: number
  ): Float32Array[] {
    const { dim, state_dim, inner_dim } = this.config!;
    const seqLen = x.length;
    const prefix = `layers.${layerIdx}.ssm.`;

    // Obtener pesos
    const inProjW = this.weights.get(prefix + 'in_proj.weight')!;
    const conv1dW = this.weights.get(prefix + 'conv1d.weight')!;
    const conv1dB = this.weights.get(prefix + 'conv1d.bias')!;
    const dtProjW = this.weights.get(prefix + 'dt_proj.weight')!;
    const dtProjB = this.weights.get(prefix + 'dt_proj.bias')!;
    const ALog = this.weights.get(prefix + 'A_log')!;
    const BProjW = this.weights.get(prefix + 'B_proj.weight')!;
    const CProjW = this.weights.get(prefix + 'C_proj.weight')!;
    const outProjW = this.weights.get(prefix + 'out_proj.weight')!;

    // 1. Proyección inicial: [seqLen, dim] -> [seqLen, 2*inner_dim]
    const projected: Float32Array[] = [];
    const residuals: Float32Array[] = [];
    
    for (let t = 0; t < seqLen; t++) {
      const proj = this.linear(x[t], inProjW, inner_dim * 2);
      projected.push(new Float32Array(proj.slice(0, inner_dim)));
      residuals.push(new Float32Array(proj.slice(inner_dim)));
    }

    // 2. Convolución 1D (kernel_size=4, causal)
    const convOut: Float32Array[] = [];
    for (let t = 0; t < seqLen; t++) {
      const out = new Float32Array(inner_dim);
      for (let c = 0; c < inner_dim; c++) {
        let sum = conv1dB[c];
        for (let k = 0; k < 4; k++) {
          const srcT = t - k;
          if (srcT >= 0) {
            sum += projected[srcT][c] * conv1dW[c * 4 + k];
          }
        }
        out[c] = this.silu(sum);
      }
      convOut.push(out);
    }

    // 3. SSM Selectivo con escaneo secuencial
    // A = -exp(A_log)
    const A = new Float32Array(inner_dim * state_dim);
    for (let i = 0; i < ALog.length; i++) {
      A[i] = -Math.exp(ALog[i]);
    }

    // Estado inicial
    const state = new Float32Array(inner_dim * state_dim);
    const ssmOut: Float32Array[] = [];

    for (let t = 0; t < seqLen; t++) {
      const xT = convOut[t];
      
      // dt = softplus(dt_proj(x))
      const dtRaw = this.linear(xT, dtProjW, inner_dim, dtProjB);
      const dt = new Float32Array(inner_dim);
      for (let i = 0; i < inner_dim; i++) {
        dt[i] = this.softplus(dtRaw[i]);
      }

      // B = B_proj(x), C = C_proj(x)
      const B = this.linear(xT, BProjW, state_dim);
      const C = this.linear(xT, CProjW, state_dim);

      // Actualizar estado y calcular salida
      const y = new Float32Array(inner_dim);
      
      for (let d = 0; d < inner_dim; d++) {
        for (let s = 0; s < state_dim; s++) {
          const idx = d * state_dim + s;
          // A_bar = exp(A * dt)
          const aBar = Math.exp(A[idx] * dt[d]);
          // B_bar = dt * B
          const bBar = dt[d] * B[s];
          // state = A_bar * state + B_bar * x
          state[idx] = aBar * state[idx] + bBar * xT[d];
          // y += C * state
          y[d] += C[s] * state[idx];
        }
      }

      ssmOut.push(y);
    }

    // 4. Combinar con residuo y proyectar salida
    const output: Float32Array[] = [];
    for (let t = 0; t < seqLen; t++) {
      // out = y * silu(res)
      const gated = new Float32Array(inner_dim);
      for (let i = 0; i < inner_dim; i++) {
        gated[i] = ssmOut[t][i] * this.silu(residuals[t][i]);
      }
      // Proyección de salida
      output.push(this.linear(gated, outProjW, dim));
    }

    return output;
  }

  // ============ Forward completo ============

  private forward(inputIds: number[]): Float32Array[] {
    const { vocab_size, dim, n_layers } = this.config!;
    const seqLen = inputIds.length;

    // Embeddings
    const tokEmb = this.weights.get('tok_embeddings.weight')!;
    let x: Float32Array[] = inputIds.map(idx => {
      const emb = new Float32Array(dim);
      for (let d = 0; d < dim; d++) {
        emb[d] = tokEmb[idx * dim + d];
      }
      return emb;
    });

    // Capas SSM
    for (let l = 0; l < n_layers; l++) {
      const normW = this.weights.get(`layers.${l}.norm.weight`)!;
      
      // Normalizar
      const normed = x.map(xi => this.rmsNorm(xi, normW));
      
      // SSM Block
      const ssmOut = this.ssmBlock(normed, l);
      
      // Conexión residual
      x = x.map((xi, t) => {
        const res = new Float32Array(dim);
        for (let d = 0; d < dim; d++) {
          res[d] = xi[d] + ssmOut[t][d];
        }
        return res;
      });
    }

    // Normalización final
    const normFW = this.weights.get('norm_f.weight')!;
    x = x.map(xi => this.rmsNorm(xi, normFW));

    // Proyección a logits
    const outW = this.weights.get('output.weight')!;
    const logits = x.map(xi => this.linear(xi, outW, vocab_size));

    return logits;
  }

  // ============ Generación ============

  async generate(
    prompt: string,
    maxTokens = 100,
    temperature = 0.8
  ): Promise<string> {
    if (!this.config || !this.tokenizer) {
      throw new Error('Modelo no cargado');
    }

    this.abortFlag = false;
    
    // Codificar prompt
    let inputIds = prompt.split('')
      .map(c => this.tokenizer!.stoi.get(c) ?? 0);
    
    let generated = prompt;
    this._generatedText.set(generated);

    for (let i = 0; i < maxTokens && !this.abortFlag; i++) {
      // Forward
      const logits = this.forward(inputIds);
      const lastLogits = logits[logits.length - 1];

      // Muestrear
      const nextIdx = this.sample(lastLogits, temperature);
      const nextChar = this.tokenizer.itos.get(nextIdx) ?? '';

      if (nextChar === '<|end|>' || nextChar === '<|pad|>') break;

      inputIds.push(nextIdx);
      generated += nextChar;
      this._generatedText.set(generated);

      // Limitar contexto
      if (inputIds.length > 150) {
        inputIds = inputIds.slice(-100);
      }

      // Yield para UI
      if (i % 2 === 0) {
        await new Promise(r => setTimeout(r, 0));
      }
    }

    return generated;
  }

  private sample(logits: Float32Array, temperature: number): number {
    const scaled = new Float32Array(logits.length);
    let maxVal = -Infinity;
    
    for (let i = 0; i < logits.length; i++) {
      scaled[i] = logits[i] / temperature;
      if (scaled[i] > maxVal) maxVal = scaled[i];
    }

    let sumExp = 0;
    const probs = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
      probs[i] = Math.exp(scaled[i] - maxVal);
      sumExp += probs[i];
    }
    for (let i = 0; i < probs.length; i++) {
      probs[i] /= sumExp;
    }

    const r = Math.random();
    let cum = 0;
    for (let i = 0; i < probs.length; i++) {
      cum += probs[i];
      if (r < cum) return i;
    }
    return probs.length - 1;
  }

  abort(): void {
    this.abortFlag = true;
  }

  getModelInfo() {
    if (!this.config) return null;
    return {
      name: 'OxideLLM_TK_SSM_V1',
      iteration: 1200,
      vocabSize: this.config.vocab_size,
      params: '~770K',
    };
  }

  dispose(): void {
    this.abort();
    this.weights.clear();
    this.config = null;
    this.tokenizer = null;
    this._state.set({ isLoading: false, isReady: false, error: null, progress: 0, progressText: '' });
  }
}
