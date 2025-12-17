import { Signal } from '@angular/core';

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
}

export interface LoadingProgress {
  progress: number;
  text: string;
  timeElapsed?: number;
}

export interface ChatBackendState {
  isInitialized: boolean;
  isLoading: boolean;
  isGenerating: boolean;
  error: string | null;
  loadingProgress: LoadingProgress | null;
}

export abstract class ChatBackendInterface {
  abstract readonly state: Signal<ChatBackendState>;
  abstract readonly currentResponse: Signal<string>;

  abstract initialize(): Promise<void>;
  abstract sendMessage(prompt: string, history?: ChatMessage[]): Promise<string>;
  abstract abortGeneration(): void;
  abstract resetChat(): void;
  abstract isWebGPUSupported(): Promise<boolean>;
}
