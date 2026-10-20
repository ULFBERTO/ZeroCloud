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
    <div class="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <!-- Header -->
      <header class="bg-slate-900 border-b border-slate-800 p-4 shrink-0">
        <div class="max-w-4xl mx-auto flex items-center justify-between">
          <div class="flex items-center gap-3">
            <button
              (click)="goBack()"
              class="p-2 text-slate-400 hover:text-slate-200 hover:bg-slate-800 rounded-xl transition-colors">
              ← Volver
            </button>
            <div>
              <h1 class="text-xl font-bold text-slate-100">🦀 OxideLLM_TK_SSM_V1</h1>
              <p class="text-xs text-slate-500">State Space Model (Mamba-like)</p>
            </div>
          </div>

          <div class="flex items-center gap-2">
            @if (state().isReady && modelInfo()) {
              <span class="px-3 py-1.5 rounded-full bg-green-500/10 text-green-400 border border-green-500/20 text-xs font-medium">
                ✓ Iter {{ modelInfo()!.iteration }}
              </span>
              <span class="px-3 py-1.5 rounded-full bg-purple-500/10 text-purple-400 border border-purple-500/20 text-xs font-medium">
                ONNX
              </span>
            }
            <a
              href="https://huggingface.co/ULFBERTO/OxideLLM_TK_SSM_V1_ONNX"
              target="_blank"
              class="px-3 py-1.5 rounded-xl bg-amber-500/10 text-amber-400 border border-amber-500/20 text-xs font-medium hover:bg-amber-500/20 transition-colors">
              🤗 HuggingFace
            </a>
          </div>
        </div>
      </header>

      <!-- Main Content -->
      <main class="flex-1 max-w-4xl mx-auto w-full p-4 md:p-6 flex flex-col">
        <!-- Loading State -->
        @if (state().isLoading) {
          <div class="flex-1 flex items-center justify-center">
            <div class="text-center">
              <div class="w-10 h-10 border-4 border-purple-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
              <p class="text-lg text-slate-200 mb-2">Cargando modelo SSM...</p>
              <div class="w-64 bg-slate-800 rounded-full h-1.5 mx-auto overflow-hidden">
                <div class="bg-purple-500 h-full rounded-full transition-all" [style.width.%]="state().progress"></div>
              </div>
              <p class="text-sm text-slate-500 mt-2">{{ state().progress | number:'1.0-0' }}%</p>
            </div>
          </div>
        }

        <!-- Error State -->
        @if (state().error) {
          <div class="flex-1 flex items-center justify-center">
            <div class="text-center max-w-md">
              <div class="text-6xl mb-4">❌</div>
              <h2 class="text-xl font-bold text-red-400 mb-2">Error al cargar</h2>
              <p class="text-slate-500 mb-4">{{ state().error }}</p>
              <p class="text-[10px] text-slate-500 mb-5">
                El modelo ONNX aún no está disponible en HuggingFace. Ejecuta <code class="bg-slate-800 px-2 py-1 rounded text-indigo-400">python convert_to_onnx.py</code>
                y sube los archivos.
              </p>
              <button
                (click)="loadModel()"
                class="px-6 py-3 bg-purple-600 text-slate-100 font-medium rounded-xl hover:bg-purple-500 transition-all duration-200 shadow-lg shadow-purple-500/20">
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
              <h2 class="text-2xl font-bold text-slate-100 mb-2">OxideLLM_TK_SSM_V1</h2>
              <p class="text-slate-500 mb-6">
                Modelo de lenguaje experimental basado en State Space Models (SSM). Arquitectura Mamba-like con complejidad O(n) lineal.
              </p>

              <!-- ONNX Notice -->
              <div class="rounded-2xl p-4 mb-6 border border-amber-500/30 bg-amber-500/10 text-left text-sm">
                <p class="text-amber-400 font-medium mb-1">⚠️ Versión ONNX</p>
                <p class="text-slate-500">
                  Esta es la versión ONNX convertida para navegador. No hay soporte nativo para PyTorch en web, por lo que se usa ONNX Runtime.
                </p>
              </div>

              <!-- SSM vs Transformer -->
              <div class="rounded-2xl p-5 mb-6 border border-slate-800 bg-slate-900/50 text-left">
                <h3 class="font-semibold text-slate-200 mb-3 text-center">⚡ SSM vs Transformer</h3>
                <div class="grid grid-cols-2 gap-4 text-xs">
                  <div>
                    <p class="text-slate-500 mb-1">Transformer</p>
                    <p class="text-slate-400">Complejidad O(n²)</p>
                  </div>
                  <div>
                    <p class="text-purple-400 mb-1 font-medium">SSM ✓</p>
                    <p class="text-slate-400">Complejidad O(n)</p>
                  </div>
                </div>
              </div>

              <button
                (click)="loadModel()"
                class="w-full px-8 py-3.5 bg-purple-600 text-slate-100 font-medium rounded-2xl hover:bg-purple-500 transition-all duration-200 shadow-lg shadow-purple-500/20">
                🚀 Cargar Modelo
              </button>
              <p class="text-[10px] text-slate-500 mt-3">Se ejecuta en tu navegador con ONNX Runtime Web</p>
            </div>
          </div>
        }

        <!-- Chat Interface -->
        @if (state().isReady) {
          <div class="flex-1 flex flex-col">
            <!-- Model Info -->
            @if (modelInfo()) {
              <div class="mb-5 p-3.5 rounded-2xl bg-slate-900/50 border border-slate-800 flex items-center justify-between text-[10px]">
                <div class="flex items-center gap-4">
                  <span class="text-slate-500">Modelo:</span>
                  <span class="text-slate-200">{{ modelInfo()!.name }}</span>
                  <span class="text-slate-500">Params:</span>
                  <span class="text-slate-200">{{ modelInfo()!.params }}</span>
                  <span class="text-slate-500">Vocab:</span>
                  <span class="text-slate-200">{{ modelInfo()!.vocabSize }}</span>
                </div>
              </div>
            }

            <!-- Generated Text Display -->
            <div class="flex-1 bg-slate-900 rounded-3xl p-5 mb-5 overflow-y-auto border border-slate-800 min-h-[300px]">
              @if (generatedText()) {
                <pre class="whitespace-pre-wrap font-serif text-sm leading-relaxed text-slate-200">{{ generatedText() }}</pre>
              } @else {
                <p class="text-slate-500 text-sm italic">Escribe un texto inicial y el modelo SSM continuará generando...</p>
              }
            </div>

            <!-- Controls -->
            <div class="space-y-4">
              <!-- Input -->
              <div class="flex gap-3">
                <textarea
                  [ngModel]="prompt()"
                  (ngModelChange)="prompt.set($event)"
                  (keydown)="onKeyDown($event)"
                  placeholder="Escribe el inicio del texto..."
                  rows="3"
                  class="flex-1 px-4 py-3.5 bg-slate-800 border border-slate-600 rounded-2xl resize-none focus:outline-none focus:border-purple-500 transition-colors text-sm"
                  [disabled]="isGenerating()">
                </textarea>
              </div>

              <!-- Settings -->
              <div class="flex items-center gap-5 text-xs">
                <label class="flex items-center gap-2">
                  <span class="text-slate-500">Longitud:</span>
                  <input
                    type="number"
                    [ngModel]="maxLength()"
                    (ngModelChange)="maxLength.set($event)"
                    min="20"
                    max="300"
                    class="w-20 px-2 py-1.5 bg-slate-800 border border-slate-600 rounded text-xs text-slate-100 focus:outline-none focus:border-indigo-500"
                  />
                </label>
                <label class="flex items-center gap-2">
                  <span class="text-slate-500">Temperatura:</span>
                  <input
                    type="range"
                    [ngModel]="temperature()"
                    (ngModelChange)="temperature.set($event)"
                    min="0.1"
                    max="1.5"
                    step="0.1"
                    class="w-24 accent-purple-500"
                  />
                  <span class="text-slate-300">{{ temperature() }}</span>
                </label>
              </div>

              <!-- Buttons -->
              <div class="flex gap-3">
                <button
                  (click)="generate()"
                  [disabled]="!canGenerate()"
                  class="flex-1 px-6 py-3.5 bg-purple-600 text-slate-100 font-medium rounded-2xl hover:bg-purple-500 disabled:bg-slate-800 disabled:text-slate-500 disabled:cursor-not-allowed transition-all duration-200 shadow-lg shadow-purple-500/20">
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
                  class="px-4 py-3.5 bg-slate-800 text-slate-400 hover:text-slate-200 hover:bg-slate-700 rounded-2xl transition-colors">
                  🗑️ Limpiar
                </button>
              </div>
            </div>
          </div>
        }
      </main>

      <!-- Footer Info -->
      <footer class="bg-slate-900 border-t border-slate-800 p-3 text-center text-[10px] text-slate-500 shrink-0">
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
