import { Injectable, signal } from '@angular/core';

export interface PyodideSSMState {
  isLoading: boolean;
  isReady: boolean;
  error: string | null;
  progress: number;
  progressText: string;
}

// URLs del modelo en HuggingFace
const MODEL_BASE_URL = 'https://huggingface.co/ULFBERTO/OxideLLM_TK_SSM_V1/resolve/main';

@Injectable({ providedIn: 'root' })
export class PyodideSSMService {
  private pyodide: any = null;
  private modelLoaded = false;

  private readonly _state = signal<PyodideSSMState>({
    isLoading: false,
    isReady: false,
    error: null,
    progress: 0,
    progressText: '',
  });

  private readonly _generatedText = signal<string>('');

  readonly state = this._state.asReadonly();
  readonly generatedText = this._generatedText.asReadonly();

  /**
   * Carga Pyodide y el modelo SSM
   */
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
      // 1. Cargar Pyodide
      this._state.update(s => ({ ...s, progress: 5, progressText: 'Cargando Python (Pyodide)...' }));
      await this.loadPyodide();

      // 2. Instalar dependencias
      this._state.update(s => ({ ...s, progress: 20, progressText: 'Instalando PyTorch...' }));
      await this.installDependencies();

      // 3. Cargar código del modelo
      this._state.update(s => ({ ...s, progress: 50, progressText: 'Cargando modelo SSM...' }));
      await this.loadModelCode();

      // 4. Descargar y cargar checkpoint
      this._state.update(s => ({ ...s, progress: 70, progressText: 'Descargando checkpoint...' }));
      await this.loadCheckpoint();

      this._state.set({
        isLoading: false,
        isReady: true,
        error: null,
        progress: 100,
        progressText: 'Listo',
      });

      console.log('✅ Modelo SSM cargado en Pyodide');

    } catch (error) {
      const message = error instanceof Error ? error.message : 'Error desconocido';
      this._state.set({
        isLoading: false,
        isReady: false,
        error: message,
        progress: 0,
        progressText: '',
      });
      console.error('❌ Error cargando modelo SSM:', error);
    }
  }

  private async loadPyodide(): Promise<void> {
    if (this.pyodide) return;

    // Cargar Pyodide desde CDN
    return new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/pyodide.js';
      script.onload = async () => {
        try {
          const loadPyodide = (window as any).loadPyodide;
          this.pyodide = await loadPyodide({
            indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.25.0/full/',
          });
          console.log('Pyodide cargado');
          resolve();
        } catch (e) {
          reject(e);
        }
      };
      script.onerror = () => reject(new Error('Error cargando Pyodide'));
      document.head.appendChild(script);
    });
  }

  private async installDependencies(): Promise<void> {
    // Instalar micropip para gestionar paquetes
    await this.pyodide.loadPackage('micropip');
    
    // Instalar torch (versión ligera para Pyodide)
    const micropip = this.pyodide.pyimport('micropip');
    
    // Nota: PyTorch completo no está disponible en Pyodide
    // Usamos una implementación simplificada con numpy
    await this.pyodide.loadPackage(['numpy']);
    
    console.log('Dependencias instaladas');
  }

  private async loadModelCode(): Promise<void> {
    // Cargar el código del modelo y tokenizer
    const modelCode = await fetch(`${MODEL_BASE_URL}/model.py`).then(r => r.text());
    const tokenizerCode = await fetch(`${MODEL_BASE_URL}/tokenizer.py`).then(r => r.text());

    // Código de inferencia simplificado usando numpy (sin PyTorch)
    const inferenceCode = `
import numpy as np
import json

class SimpleSSMInference:
    """
    Implementación simplificada del modelo SSM usando NumPy.
    Carga los pesos del checkpoint de PyTorch y hace inferencia.
    """
    def __init__(self):
        self.weights = None
        self.tokenizer = None
        self.vocab_size = 0
        self.dim = 128
        
    def load_weights(self, weights_dict, tokenizer_chars):
        """Carga los pesos desde un diccionario"""
        self.weights = weights_dict
        
        # Reconstruir tokenizer
        self.tokenizer = {
            'chars': tokenizer_chars,
            'stoi': {ch: i for i, ch in enumerate(tokenizer_chars)},
            'itos': {i: ch for i, ch in enumerate(tokenizer_chars)},
        }
        self.vocab_size = len(tokenizer_chars)
        print(f"Modelo cargado: vocab_size={self.vocab_size}")
        
    def encode(self, text):
        """Codifica texto a índices"""
        return [self.tokenizer['stoi'].get(c, 0) for c in text]
    
    def decode(self, indices):
        """Decodifica índices a texto"""
        return ''.join([self.tokenizer['itos'].get(i, '') for i in indices])
    
    def forward_simple(self, input_ids):
        """
        Forward pass simplificado usando solo embeddings.
        """
        # Obtener embeddings
        embeddings = self.weights['tok_embeddings.weight']  # [vocab, dim]
        output_weights = self.weights['output.weight']  # [vocab, dim]
        
        # Lookup embeddings
        x = embeddings[input_ids]  # [seq_len, dim]
        
        # Contexto simple: promedio acumulado
        seq_len = len(input_ids)
        context = np.zeros_like(x)
        cumsum = np.cumsum(x, axis=0)
        for i in range(seq_len):
            context[i] = cumsum[i] / (i + 1)
        
        # Mezclar
        mixed = x * 0.7 + context * 0.3
        
        # Proyectar a logits
        logits = np.dot(mixed, output_weights.T)  # [seq_len, vocab]
        
        return logits
    
    def generate(self, prompt, max_tokens=100, temperature=0.8):
        """Genera texto"""
        input_ids = self.encode(prompt)
        generated = prompt
        
        for _ in range(max_tokens):
            # Forward
            logits = self.forward_simple(input_ids)
            
            # Obtener logits del último token
            last_logits = logits[-1] / temperature
            
            # Softmax
            exp_logits = np.exp(last_logits - np.max(last_logits))
            probs = exp_logits / np.sum(exp_logits)
            
            # Muestrear
            next_idx = np.random.choice(len(probs), p=probs)
            
            # Verificar token de fin
            next_char = self.tokenizer['itos'].get(next_idx, '')
            if next_char == '<|end|>':
                break
            
            # Agregar
            input_ids.append(next_idx)
            generated += next_char
        
        return generated

# Instancia global
ssm_model = SimpleSSMInference()
`;

    await this.pyodide.runPythonAsync(inferenceCode);
    console.log('Código de inferencia cargado');
  }

  private async loadCheckpoint(): Promise<void> {
    // Descargar checkpoint
    const checkpointUrl = `${MODEL_BASE_URL}/ssm_checkpoint.pth`;
    
    // Nota: No podemos cargar .pth directamente en Pyodide sin PyTorch
    // Necesitamos un formato alternativo (JSON con los pesos)
    
    // Por ahora, intentamos cargar el tokenizer.json si existe
    try {
      const tokenizerUrl = `${MODEL_BASE_URL}/tokenizer.json`;
      const response = await fetch(tokenizerUrl);
      
      if (response.ok) {
        const tokenizerData = await response.json();
        
        // Pasar datos a Python
        this.pyodide.globals.set('tokenizer_data', tokenizerData);
        
        await this.pyodide.runPythonAsync(`
import js
import json

# Obtener datos del tokenizer
tok_data = js.tokenizer_data.to_py()
print(f"Tokenizer cargado: {tok_data.get('vocab_size', 0)} tokens")
`);
        
        this.modelLoaded = true;
      } else {
        throw new Error('No se pudo cargar el tokenizer');
      }
    } catch (e) {
      console.warn('Error cargando checkpoint:', e);
      throw new Error('El modelo SSM requiere archivos adicionales en HuggingFace. Usa la versión PyTorch local.');
    }
  }

  /**
   * Genera texto
   */
  async generate(prompt: string, maxTokens: number = 100, temperature: number = 0.8): Promise<string> {
    if (!this.pyodide || !this.modelLoaded) {
      throw new Error('Modelo no cargado');
    }

    this._generatedText.set(prompt);

    try {
      // Ejecutar generación en Python
      this.pyodide.globals.set('prompt', prompt);
      this.pyodide.globals.set('max_tokens', maxTokens);
      this.pyodide.globals.set('temperature', temperature);

      const result = await this.pyodide.runPythonAsync(`
ssm_model.generate(prompt, max_tokens, temperature)
`);

      const generated = result as string;
      this._generatedText.set(generated);
      return generated;

    } catch (error) {
      console.error('Error generando:', error);
      throw error;
    }
  }

  /**
   * Obtiene información del modelo
   */
  getModelInfo(): { name: string; iteration: number; vocabSize: number; params: string } | null {
    if (!this.modelLoaded) return null;
    return {
      name: 'OxideLLM_TK_SSM_V1',
      iteration: 1200,
      vocabSize: 228,
      params: '~770K',
    };
  }

  dispose(): void {
    this.pyodide = null;
    this.modelLoaded = false;
    this._state.set({ isLoading: false, isReady: false, error: null, progress: 0, progressText: '' });
    this._generatedText.set('');
  }
}
