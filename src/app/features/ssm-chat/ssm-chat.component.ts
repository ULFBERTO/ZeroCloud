import { Component, inject, signal, computed } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DecimalPipe } from '@angular/common';
import { Router } from '@angular/router';
import { OnnxSSMService } from '../../core/services/onnx-ssm.service';

@Component({
  selector: 'app-ssm-chat',
  standalone: true,
  imports: [FormsModule, DecimalPipe],
  template: `
    <div class="min-h-screen bg-gray-900 text-white flex flex-col">
      <!-- Header -->
      <header class="bg-gray-800 border-b border-gray-700 p-4">
        <div class="max-w-4xl mx-auto flex items-center justify-between">
          <div class="flex items-center gap-3">
            <button
              (click)="goBack()"
              class="p-2 hover:bg-gray-700 rounded-lg transition-colors">
              ← Volver
            </button>
            <div>
              <h1 class="text-xl font-bold">🦀 OxideLLM_TK_SSM_V1</h1>
              <p class="text-sm text-gray-400">State Space Model (Mamba-like)</p>
            </div>
          </div>
          
          <div class="flex items-center gap-2">
            @if (state().isReady && modelInfo()) {
              <span class="px-3 py-1 bg-green-600/20 text-green-400 rounded-full text-sm">
                ✓ Iter {{ modelInfo()!.iteration }}
              </span>
              <span class="px-3 py-1 bg-orange-600/20 text-orange-400 rounded-full text-sm">
                ONNX
              </span>
            }
            <a 
              href="https://huggingface.co/ULFBERTO/OxideLLM_TK_SSM_V1_ONNX"
              target="_blank"
              class="px-3 py-1.5 bg-yellow-600/20 text-yellow-400 rounded-lg text-sm hover:bg-yellow-600/30 transition-colors">
              🤗 HuggingFace
            </a>
          </div>
        </div>
      </header>

      <!-- Main Content -->
      <main class="flex-1 max-w-4xl mx-auto w-full p-4 flex flex-col">
        <!-- Loading State -->
        @if (state().isLoading) {
          <div class="flex-1 flex items-center justify-center">
            <div class="text-center">
              <div class="animate-spin w-12 h-12 border-4 border-purple-500 border-t-transparent rounded-full mx-auto mb-4"></div>
              <p class="text-lg mb-2">Cargando modelo SSM...</p>
              <div class="w-64 bg-gray-700 rounded-full h-2 mx-auto">
                <div
                  class="bg-purple-500 h-full rounded-full transition-all"
                  [style.width.%]="state().progress">
                </div>
              </div>
              <p class="text-sm text-gray-400 mt-2">{{ state().progress | number:'1.0-0' }}%</p>
            </div>
          </div>
        }

        <!-- Error State -->
        @if (state().error) {
          <div class="flex-1 flex items-center justify-center">
            <div class="text-center max-w-md">
              <div class="text-6xl mb-4">❌</div>
              <h2 class="text-xl font-bold mb-2">Error al cargar</h2>
              <p class="text-gray-400 mb-4">{{ state().error }}</p>
              <p class="text-sm text-gray-500 mb-4">
                El modelo ONNX aún no está disponible en HuggingFace.
                Ejecuta <code class="bg-gray-800 px-2 py-1 rounded">python convert_to_onnx.py</code> 
                y sube los archivos.
              </p>
              <button
                (click)="loadModel()"
                class="px-6 py-3 bg-purple-600 hover:bg-purple-500 rounded-xl transition-colors">
                Reintentar
              </button>
            </div>
          </div>
        }

        <!-- Not Loaded State -->
        @if (!state().isLoading && !state().isReady && !state().error) {
          <div class="flex-1 flex items-center justify-center">
            <div class="text-center max-w-lg">
              <div class="text-6xl mb-4">🦀</div>
              <h2 class="text-2xl font-bold mb-2">OxideLLM_TK_SSM_V1</h2>
              <p class="text-gray-400 mb-4">
                Modelo de lenguaje experimental basado en State Space Models (SSM).
                Arquitectura Mamba-like con complejidad O(n) lineal.
              </p>
              
              <!-- ONNX Notice -->
              <div class="bg-orange-900/20 border border-orange-500/30 rounded-lg p-3 mb-4 text-sm text-left">
                <p class="text-orange-400 font-medium mb-1">⚠️ Versión ONNX</p>
                <p class="text-gray-400">
                  Esta es la versión ONNX convertida para navegador. 
                  No hay soporte nativo para PyTorch en web, por lo que se usa ONNX Runtime.
                </p>
              </div>
              
              <!-- SSM vs Transformer -->
              <div class="bg-gray-800 rounded-xl p-4 mb-6 text-left">
                <h3 class="font-semibold mb-3 text-center">⚡ SSM vs Transformer</h3>
                <div class="grid grid-cols-2 gap-4 text-sm">
                  <div>
                    <p class="text-gray-500 mb-1">Transformer</p>
                    <p>Complejidad O(n²)</p>
                  </div>
                  <div>
                    <p class="text-purple-400 mb-1">SSM ✓</p>
                    <p>Complejidad O(n)</p>
                  </div>
                </div>
              </div>

              <button
                (click)="loadModel()"
                class="px-8 py-3 bg-purple-600 hover:bg-purple-500 rounded-xl font-medium transition-colors">
                🚀 Cargar Modelo
              </button>
              <p class="text-xs text-gray-500 mt-3">
                Se ejecuta en tu navegador con ONNX Runtime Web
              </p>
            </div>
          </div>
        }

        <!-- Chat Interface -->
        @if (state().isReady) {
          <div class="flex-1 flex flex-col">
            <!-- Model Info -->
            @if (modelInfo()) {
              <div class="bg-gray-800/50 rounded-lg p-3 mb-4 flex items-center justify-between text-sm">
                <div class="flex items-center gap-4">
                  <span class="text-gray-400">Modelo: <span class="text-white">{{ modelInfo()!.name }}</span></span>
                  <span class="text-gray-400">Params: <span class="text-white">{{ modelInfo()!.params }}</span></span>
                  <span class="text-gray-400">Vocab: <span class="text-white">{{ modelInfo()!.vocabSize }}</span></span>
                </div>
              </div>
            }

            <!-- Generated Text Display -->
            <div class="flex-1 bg-gray-800 rounded-xl p-4 mb-4 overflow-y-auto min-h-[300px]">
              @if (generatedText()) {
                <pre class="whitespace-pre-wrap font-serif text-lg leading-relaxed">{{ generatedText() }}</pre>
              } @else {
                <p class="text-gray-500 italic">
                  Escribe un texto inicial y el modelo SSM continuará generando...
                </p>
              }
            </div>

            <!-- Controls -->
            <div class="space-y-4">
              <!-- Input -->
              <div class="flex gap-2">
                <textarea
                  [ngModel]="prompt()"
                  (ngModelChange)="prompt.set($event)"
                  (keydown)="onKeyDown($event)"
                  placeholder="Escribe el inicio del texto..."
                  rows="3"
                  class="flex-1 px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl resize-none focus:outline-none focus:border-purple-500"
                  [disabled]="isGenerating()">
                </textarea>
              </div>

              <!-- Settings -->
              <div class="flex items-center gap-4 text-sm">
                <label class="flex items-center gap-2">
                  <span class="text-gray-400">Longitud:</span>
                  <input
                    type="number"
                    [ngModel]="maxLength()"
                    (ngModelChange)="maxLength.set($event)"
                    min="20"
                    max="300"
                    class="w-20 px-2 py-1 bg-gray-800 border border-gray-700 rounded"
                  />
                </label>
                <label class="flex items-center gap-2">
                  <span class="text-gray-400">Temperatura:</span>
                  <input
                    type="range"
                    [ngModel]="temperature()"
                    (ngModelChange)="temperature.set($event)"
                    min="0.1"
                    max="1.5"
                    step="0.1"
                    class="w-24"
                  />
                  <span>{{ temperature() }}</span>
                </label>
              </div>

              <!-- Buttons -->
              <div class="flex gap-2">
                <button
                  (click)="generate()"
                  [disabled]="!canGenerate()"
                  class="flex-1 px-6 py-3 bg-purple-600 hover:bg-purple-500 rounded-xl font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
                  @if (isGenerating()) {
                    <span class="flex items-center justify-center gap-2">
                      <span class="animate-spin">⏳</span> Generando...
                    </span>
                  } @else {
                    🦀 Generar con SSM
                  }
                </button>
                <button
                  (click)="clear()"
                  class="px-4 py-3 bg-gray-700 hover:bg-gray-600 rounded-xl transition-colors">
                  🗑️ Limpiar
                </button>
              </div>
            </div>
          </div>
        }
      </main>

      <!-- Footer Info -->
      <footer class="bg-gray-800 border-t border-gray-700 p-3 text-center text-sm text-gray-500">
        🦀 SSM Transformer Killer • Versión ONNX • Complejidad O(n) lineal • 100% en tu navegador
      </footer>
    </div>
  `,
})
export class SSMChatComponent {
  private readonly ssmService = inject(OnnxSSMService);
  private readonly router = inject(Router);

  readonly state = this.ssmService.state;
  readonly generatedText = this.ssmService.generatedText;

  readonly prompt = signal('Hola, ');
  readonly maxLength = signal(100);
  readonly temperature = signal(0.8);
  readonly isGenerating = signal(false);

  readonly canGenerate = computed(
    () => this.state().isReady && !this.isGenerating() && this.prompt().trim().length > 0
  );

  modelInfo() {
    return this.ssmService.getModelInfo();
  }

  async loadModel(): Promise<void> {
    await this.ssmService.loadModel();
  }

  async generate(): Promise<void> {
    if (!this.canGenerate()) return;

    this.isGenerating.set(true);
    try {
      await this.ssmService.generate(
        this.prompt(),
        this.maxLength(),
        this.temperature()
      );
    } catch (error) {
      console.error('Error generating:', error);
    } finally {
      this.isGenerating.set(false);
    }
  }

  clear(): void {
    this.prompt.set('');
  }

  onKeyDown(event: KeyboardEvent): void {
    if (event.key === 'Enter' && event.ctrlKey) {
      event.preventDefault();
      this.generate();
    }
  }

  goBack(): void {
    this.router.navigate(['/']);
  }
}
