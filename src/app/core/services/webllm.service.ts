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
  supportsF16?: boolean;
  vendor?: string;
}

const GPU_PREFERENCE_KEY = 'webllm_gpu_preference';
const F16_SUPPORT_KEY = 'webllm_f16_support';

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
  private readonly _supportsF16 = signal<boolean>(
    localStorage.getItem(F16_SUPPORT_KEY) !== 'false'
  );

  readonly availableGPUs = this._availableGPUs.asReadonly();
  readonly selectedGPU = this._selectedGPU.asReadonly();
  readonly supportsF16 = this._supportsF16.asReadonly();

  readonly state = this._state.asReadonly();
  readonly currentResponse = this._currentResponse.asReadonly();

  async detectAvailableGPUs(): Promise<GPUInfo[]> {
    interface GPUAdapterLike {
      features?: { has: (key: string) => boolean };
      info?: { vendor?: string; architecture?: string; device?: string; description?: string };
      requestAdapterInfo?: () => Promise<{ vendor?: string; architecture?: string; device?: string; description?: string }>;
    }

    const nav = navigator as Navigator & {
      gpu?: {
        requestAdapter: (options?: { powerPreference?: string; forceFallbackAdapter?: boolean }) => Promise<GPUAdapterLike | null>;
      };
    };

    if (!nav.gpu) return [];

    const gpus: GPUInfo[] = [];
    const seenGPUs = new Set<string>();

    const getGPUInfo = async (adapter: GPUAdapterLike): Promise<{ name: string; vendor: string; supportsF16: boolean; uniqueId: string }> => {
      let info = adapter.info;
      if (!info && adapter.requestAdapterInfo) {
        info = await adapter.requestAdapterInfo();
      }
      const vendor = info?.vendor?.toLowerCase() || '';
      const architecture = info?.architecture || '';
      const device = info?.device || info?.description || '';
      const name = info
        ? `${info.vendor || ''} ${architecture || device}`.trim()
        : '';
      
      // Crear ID único basado en vendor + architecture/device
      const uniqueId = `${vendor}-${architecture}-${device}`.toLowerCase();
      
      // Detectar soporte F16 - Intel Gen9 y anteriores no lo soportan bien
      const isIntel = vendor.includes('intel');
      const isOldIntel = isIntel && (architecture.includes('gen-9') || architecture.includes('gen-8') || architecture.includes('gen9') || architecture.includes('gen8'));
      const hasF16Feature = adapter.features?.has('shader-f16') ?? false;
      const supportsF16 = !isOldIntel && (hasF16Feature || !isIntel);

      return { name: name || 'GPU', vendor, supportsF16, uniqueId };
    };

    // Intentar obtener GPU de alto rendimiento (dedicada - NVIDIA/AMD)
    try {
      const highPerfAdapter = await nav.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (highPerfAdapter) {
        const { name, vendor, supportsF16, uniqueId } = await getGPUInfo(highPerfAdapter as GPUAdapterLike);
        if (!seenGPUs.has(uniqueId)) {
          seenGPUs.add(uniqueId);
          gpus.push({
            id: 'high-performance',
            name: name || 'GPU Dedicada',
            powerPreference: 'high-performance',
            supportsF16,
            vendor,
          });
        }
      }
    } catch (e) {
      console.warn('Error detecting high-performance GPU:', e);
    }

    // Intentar obtener GPU de bajo consumo (integrada - Intel)
    try {
      const lowPowerAdapter = await nav.gpu.requestAdapter({ powerPreference: 'low-power' });
      if (lowPowerAdapter) {
        const { name, vendor, supportsF16, uniqueId } = await getGPUInfo(lowPowerAdapter as GPUAdapterLike);
        // Solo agregar si es diferente
        if (!seenGPUs.has(uniqueId)) {
          seenGPUs.add(uniqueId);
          gpus.push({
            id: 'low-power',
            name: name || 'GPU Integrada',
            powerPreference: 'low-power',
            supportsF16,
            vendor,
          });
        }
      }
    } catch (e) {
      console.warn('Error detecting low-power GPU:', e);
    }

    // Intentar obtener adaptador por defecto (sin preferencia)
    try {
      const defaultAdapter = await nav.gpu.requestAdapter();
      if (defaultAdapter) {
        const { name, vendor, supportsF16, uniqueId } = await getGPUInfo(defaultAdapter as GPUAdapterLike);
        if (!seenGPUs.has(uniqueId)) {
          seenGPUs.add(uniqueId);
          gpus.push({
            id: 'default',
            name: name || 'GPU por defecto',
            powerPreference: 'high-performance',
            supportsF16,
            vendor,
          });
        }
      }
    } catch (e) {
      console.warn('Error detecting default GPU:', e);
    }

    this._availableGPUs.set(gpus);

    // Log para debug
    console.log('GPUs detectadas:', gpus.map(g => `${g.name} (${g.id})`));

    // Actualizar soporte F16 basado en GPU seleccionada
    const selectedGpu = gpus.find(g => g.id === this._selectedGPU()) || gpus[0];
    if (selectedGpu) {
      this._supportsF16.set(selectedGpu.supportsF16 ?? true);
      localStorage.setItem(F16_SUPPORT_KEY, String(selectedGpu.supportsF16 ?? true));
    }

    return gpus;
  }

  selectGPU(gpuId: string): void {
    this._selectedGPU.set(gpuId);
    localStorage.setItem(GPU_PREFERENCE_KEY, gpuId);

    // Actualizar soporte F16 basado en GPU seleccionada
    const selectedGpu = this._availableGPUs().find(g => g.id === gpuId);
    if (selectedGpu) {
      this._supportsF16.set(selectedGpu.supportsF16 ?? true);
      localStorage.setItem(F16_SUPPORT_KEY, String(selectedGpu.supportsF16 ?? true));
    }
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
      // Log de GPU seleccionada
      console.log(`🖥️ Inicializando con GPU: ${gpuCheck.adapterInfo}`);
      
      // Crear engine - web-llm usa la GPU por defecto del sistema
      this.engine = new MLCEngine({
        initProgressCallback: (report: InitProgressReport) => {
          this._state.update((s) => ({
            ...s,
            loadingProgress: {
              progress: Math.round(report.progress * 100),
              text: report.text,
            },
          }));
          // Log de progreso para debug
          if (report.progress === 1) {
            console.log('✅ Modelo cargado en GPU');
          }
        },
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
    
    // Errores de shader/GPU
    if (message.includes('ShaderModule') || message.includes('shader')) {
      return 'Error de compatibilidad con la GPU. Prueba: 1) Seleccionar otra GPU, 2) Probar otro modelo (Qwen o SmolLM), 3) Limpiar caché del modelo.';
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

  async clearModelCache(): Promise<void> {
    const modelId = this.modelManager.selectedModelId();
    try {
      await this.modelManager.deleteModel(modelId);
    } catch {
      // Ignorar errores de borrado
    }
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
