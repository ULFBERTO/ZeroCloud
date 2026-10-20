export interface ModelInfo {
  id: string;
  name: string;
  size: string;
  description: string;
  isDownloaded: boolean;
  isCustom: boolean;
  vramRequired?: string;
  requiresF16?: boolean;
}

export interface CustomModelConfig {
  modelId: string;
  modelUrl: string;
  wasmUrl?: string;
}

export interface HFModelInfo {
  repoId: string;
  name: string;
  description: string;
  size?: string;
  downloads?: number;
  isDefault?: boolean;
}

export interface DownloadProgress {
  repoId: string;
  progress: number;
  status: 'downloading' | 'completed' | 'error';
  message: string;
}
