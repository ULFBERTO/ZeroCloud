import { HFModelInfo, ModelInfo } from '../models/model.model';

export type ModelCatalogEntry = Omit<ModelInfo, 'isDownloaded'>;

/**
 * Modelos disponibles en el catálogo local. Mantener los identificadores
 * sincronizados con el registro de MLC-AI.
 */
export const RECOMMENDED_MODELS: ModelCatalogEntry[] = [
  {
    id: 'OxideLLM_TK_SSM_V1_ONNX',
    name: 'OxideLLM SSM V1',
    size: '~4 MB',
    description: 'State space model O(n) lineal. Ultra ligero y experimental.',
    isCustom: false,
    vramRequired: '256 MB',
    requiresF16: false,
  },
  {
    id: 'Llama-3.2-1B-Instruct-q4f32_1-MLC',
    name: 'Llama 3.2 1B (f32)',
    size: '~800 MB',
    description: 'Compatibilidad máxima con GPUs Intel y hardware antiguo.',
    isCustom: false,
    vramRequired: '2 GB',
    requiresF16: false,
  },
  {
    id: 'Llama-3.2-1B-Instruct-q4f16_1-MLC',
    name: 'Llama 3.2 1B',
    size: '~700 MB',
    description: 'Rápido y equilibrado. Requiere soporte float16.',
    isCustom: false,
    vramRequired: '2 GB',
    requiresF16: true,
  },
  {
    id: 'Llama-3.2-3B-Instruct-q4f16_1-MLC',
    name: 'Llama 3.2 3B',
    size: '~1.8 GB',
    description: 'Mejor calidad de razonamiento. Requiere float16.',
    isCustom: false,
    vramRequired: '4 GB',
    requiresF16: true,
  },
  {
    id: 'Phi-3.5-mini-instruct-q4f16_1-MLC',
    name: 'Phi 3.5 Mini',
    size: '~2.2 GB',
    description: 'Modelo compacto de Microsoft para tareas de razonamiento.',
    isCustom: false,
    vramRequired: '4 GB',
    requiresF16: true,
  },
  {
    id: 'Qwen2.5-1.5B-Instruct-q4f32_1-MLC',
    name: 'Qwen 2.5 1.5B (f32)',
    size: '~1.2 GB',
    description: 'Alternativa compatible con GPUs sin float16.',
    isCustom: false,
    vramRequired: '3 GB',
    requiresF16: false,
  },
  {
    id: 'SmolLM2-1.7B-Instruct-q4f16_1-MLC',
    name: 'SmolLM2 1.7B',
    size: '~1 GB',
    description: 'Modelo ligero para equipos con VRAM limitada.',
    isCustom: false,
    vramRequired: '2 GB',
    requiresF16: true,
  },
];

export const DEFAULT_HF_MODELS: HFModelInfo[] = [
  {
    repoId: 'ULFBERTO/OxideLLM_5M-tfjs',
    name: 'OxideLLM 5M',
    description: 'Modelo GPT entrenado con el texto de OxideLLM 5M de la Mancha.',
    size: '~50 MB',
    isDefault: true,
  },
  {
    repoId: 'ULFBERTO/OxideLLM_TK_SSM_V1',
    name: 'OxideLLM TK SSM V1',
    description: 'SSM selectivo (Mamba-like) con complejidad lineal.',
    size: '~9 MB',
    isDefault: true,
  },
];
