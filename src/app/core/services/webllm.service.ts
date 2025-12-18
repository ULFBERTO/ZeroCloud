import { Injectable, signal, inject } from '@angular/core';
import {
  ChatBackendInterface,
  ChatBackendState,
  ChatMessage,
} from '../interfaces/chat-backend.interface';
import {
  MLCEngine,
  InitProgressReport,
  ChatCompletionMessageParam,
} from '@mlc-ai/web-llm';
import { ModelManagerService } from './model-manager.service';

export interface GPUInfo {
  id: string;
  name: string;
  powerPreference: 'high-performance' | 'low-power';
}

const GPU_PREFERENCE_KEY = 'webllm_gpu_preference';

@Injectable({ providedIn: 'root' })
export class WebLLMService extends ChatBackendInterface {
  private readonly modelManager = inject(ModelManagerService);
  private engine: MLCEngine | null = null;
  private abortController: AbortController | null = null;
  private currentModelId: string | null = null;

  private readonly _state = signal<ChatBackendState>({
    isInitialized: false,
    isLoading: false,
    isGenerating: false,
    error: null,
    loadingProgress: null,
  });

  private readonly _currentResponse = signal<string>('');
  private readonly _availableGPUs = signal<GPUInfo[]>([]);
  private readonly _selectedGPU = signal<string>(
    localStorage.getItem(GPU_PREFERENCE_KEY) || 'high-performance'
  );

  readonly availableGPUs = this._availableGPUs.asReadonly();
  readonly selectedGPU = this._selectedGPU.asReadonly();

  readonly state = this._state.asReadonly();
  readonly currentResponse = this._currentResponse.asReadonly();

  async detectAvailableGPUs(): Promise<GPUInfo[]> {
    interface GPUAdapterLike {
      requestAdapterInfo?: () => Promise<{ vendor?: string; architecture?: string; device?: string }>;
    }

    const nav = navigator as Navigator & {
      gpu?: {
        requestAdapter: (options?: { powerPreference?: string }) => Promise<GPUAdapterLike | null>;
      };
    };

    if (!nav.gpu) return [];

    const gpus: GPUInfo[] = [];

    // Intentar obtener GPU de alto rendimiento (dedicada - NVIDIA/AMD)
    try {
      const highPerfAdapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (highPerfAdapter) {
        const info = await highPerfAdapter.requestAdapterInfo?.();
        const name = info
          ? `${info.vendor || ''} ${info.architecture || info.device || ''}`.trim() || 'GPU Dedicada'
          : 'GPU Dedicada';
        gpus.push({ id: 'high-performance', name, powerPreference: 'high-performance' });
      }
    } catch {}

    // Intentar obtener GPU de bajo consumo (integrada - Intel)
    try {
      const lowPowerAdapter = await nav.gpu.requestAdapter({ powerPreference: 'low-power' });
      if (lowPowerAdapter) {
        const info = await lowPowerAdapter.requestAdapterInfo?.();
        const name = info
          ? `${info.vendor || ''} ${info.architecture || info.device || ''}`.trim() || 'GPU Integrada'
          : 'GPU Integrada';
        // Solo agregar si es diferente a la de alto rendimiento
        if (gpus.length === 0 || gpus[0].name !== name) {
          gpus.push({ id: 'low-power', name, powerPreference: 'low-power' });
        }
      }
    } catch {}

    this._availableGPUs.set(gpus);
    return gpus;
  }

  selectGPU(gpuId: string): void {
    this._selectedGPU.set(gpuId);
    localStorage.setItem(GPU_PREFERENCE_KEY, gpuId);
  }

  async checkWebGPUSupport(): Promise<{ supported: boolean; error?: string; adapterInfo?: string }> {
    interface GPUAdapterLike {
      requestAdapterInfo?: () => Promise<{ vendor?: string; architecture?: string; device?: string }>;
    }

    const nav = navigator as Navigator & {
      gpu?: {
        requestAdapter: (options?: { powerPreference?: string }) => Promise<GPUAdapterLike | null>;
      };
    };

    if (!nav.gpu) {
      return {
        supported: false,
        error: 'Tu navegador no soporta WebGPU. Usa Chrome 113+ (Android/PC) o Edge 113+.',
      };
    }

    try {
      // Usar la GPU seleccionada por el usuario
      const preference = this._selectedGPU() as 'high-performance' | 'low-power';
      const adapter = await nav.gpu.requestAdapter({ powerPreference: preference });

      if (!adapter) {
        return {
          supported: false,
          error: 'No se encontró una GPU compatible con WebGPU.',
        };
      }

      const info = await adapter.requestAdapterInfo?.();
      const adapterInfo = info
        ? `${info.vendor || ''} ${info.architecture || info.device || ''}`.trim() || 'GPU detectada'
        : 'GPU detectada';

      return { supported: true, adapterInfo };
    } catch {
      return {
        supported: false,
        error: 'Error al acceder a la GPU.',
      };
    }
  }

  async isWebGPUSupported(): Promise<boolean> {
    const result = await this.checkWebGPUSupport();
    return result.supported;
  }


  async initialize(): Promise<void> {
    const gpuCheck = await this.checkWebGPUSupport();
    if (!gpuCheck.supported) {
      this._state.update((s) => ({
        ...s,
        error: gpuCheck.error || 'WebGPU no disponible',
      }));
      throw new Error(gpuCheck.error);
    }

    this._state.update((s) => ({
      ...s,
      isLoading: true,
      error: null,
      loadingProgress: { progress: 0, text: `Iniciando con ${gpuCheck.adapterInfo}...` },
    }));

    try {
      this.engine = new MLCEngine();

      this.engine.setInitProgressCallback((report: InitProgressReport) => {
        this._state.update((s) => ({
          ...s,
          loadingProgress: {
            progress: Math.round(report.progress * 100),
            text: report.text,
          },
        }));
      });

      const modelId = this.modelManager.selectedModelId();
      await this.engine.reload(modelId);
      this.currentModelId = modelId;

      // Actualizar estado de descarga en el manager
      await this.modelManager.loadModels();

      this._state.update((s) => ({
        ...s,
        isInitialized: true,
        isLoading: false,
        loadingProgress: null,
      }));
    } catch (error) {
      const errorMessage = this.parseModelError(error);
      this._state.update((s) => ({
        ...s,
        isLoading: false,
        error: errorMessage,
        loadingProgress: null,
      }));
      throw error;
    }
  }

  private parseModelError(error: unknown): string {
    const message = error instanceof Error ? error.message : String(error);
    
    // Errores de shader/GPU - pueden ser temporales
    if (message.includes('ShaderModule') || message.includes('shader')) {
      return 'Error al compilar el modelo. Esto puede ser temporal. Intenta recargar la página o reiniciar el navegador.';
    }
    
    // Errores de memoria
    if (message.includes('out of memory') || message.includes('OOM') || message.includes('allocation')) {
      return 'No hay suficiente memoria en la GPU. Cierra otras pestañas o aplicaciones y prueba con un modelo más pequeño.';
    }
    
    // Errores de red
    if (message.includes('fetch') || message.includes('network') || message.includes('Failed to load')) {
      return 'Error de conexión al descargar el modelo. Verifica tu conexión a internet e intenta de nuevo.';
    }
    
    // Error genérico pero amigable
    return `No se pudo cargar el modelo. ${message.length < 100 ? message : 'Intenta recargar la página.'}`;
  }


  async sendMessage(prompt: string, history: ChatMessage[] = []): Promise<string> {
    if (!this.engine || !this._state().isInitialized) {
      throw new Error('El motor no está inicializado');
    }

    this._state.update((s) => ({ ...s, isGenerating: true, error: null }));
    this._currentResponse.set('');
    this.abortController = new AbortController();

    try {
      const messages: ChatCompletionMessageParam[] = [
        {
          role: 'system',
          content: 'Eres un asistente útil y amigable. Responde de forma concisa y clara en español.',
        },
        ...history.map((msg) => ({
          role: msg.role as 'user' | 'assistant',
          content: msg.content,
        })),
        { role: 'user', content: prompt },
      ];

      const stream = await this.engine.chat.completions.create({
        messages,
        stream: true,
        temperature: 0.7,
        max_tokens: 1024,
      });

      let fullResponse = '';

      for await (const chunk of stream) {
        if (this.abortController?.signal.aborted) break;
        const delta = chunk.choices[0]?.delta?.content || '';
        fullResponse += delta;
        this._currentResponse.set(fullResponse);
      }

      this._state.update((s) => ({ ...s, isGenerating: false }));
      return fullResponse;
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Error desconocido';
      this._state.update((s) => ({
        ...s,
        isGenerating: false,
        error: `Error en generación: ${message}`,
      }));
      throw error;
    }
  }

  abortGeneration(): void {
    this.abortController?.abort();
    this._state.update((s) => ({ ...s, isGenerating: false }));
  }

  resetChat(): void {
    this.engine?.resetChat();
    this._currentResponse.set('');
  }

  async switchModel(modelId: string): Promise<void> {
    if (this.currentModelId === modelId && this._state().isInitialized) {
      return;
    }

    this.modelManager.selectModel(modelId);
    this._state.update((s) => ({
      ...s,
      isInitialized: false,
      isLoading: false,
      error: null,
    }));

    await this.initialize();
  }

  getCurrentModelId(): string | null {
    return this.currentModelId;
  }
}
