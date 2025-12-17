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

  readonly state = this._state.asReadonly();
  readonly currentResponse = this._currentResponse.asReadonly();

  async isWebGPUSupported(): Promise<boolean> {
    const nav = navigator as Navigator & { gpu?: { requestAdapter: () => Promise<unknown | null> } };
    if (!nav.gpu) return false;
    try {
      const adapter = await nav.gpu.requestAdapter();
      return adapter !== null;
    } catch {
      return false;
    }
  }


  async initialize(): Promise<void> {
    const supported = await this.isWebGPUSupported();
    if (!supported) {
      this._state.update((s) => ({
        ...s,
        error: 'WebGPU no está soportado en este navegador. Usa Chrome/Edge 113+ con GPU compatible.',
      }));
      throw new Error('WebGPU not supported');
    }

    this._state.update((s) => ({
      ...s,
      isLoading: true,
      error: null,
      loadingProgress: { progress: 0, text: 'Iniciando...' },
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
      const message = error instanceof Error ? error.message : 'Error desconocido';
      this._state.update((s) => ({
        ...s,
        isLoading: false,
        error: `Error al cargar el modelo: ${message}`,
        loadingProgress: null,
      }));
      throw error;
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
