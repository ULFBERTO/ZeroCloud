import { Component, inject, signal, computed } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DecimalPipe } from '@angular/common';
import { Router } from '@angular/router';
import { TFJSModelService } from '../../core/services/tfjs-model.service';

@Component({
  selector: 'app-tfjs-chat',
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
              <h1 class="text-xl font-bold text-slate-100">🧠 GPT OxideLLM_5M</h1>
              <p class="text-xs text-slate-500">Modelo TFJS en navegador</p>
            </div>
          </div>

          <div class="flex items-center gap-2">
            @if (state().isReady) {
              <span class="px-3 py-1.5 rounded-full bg-green-500/10 text-green-400 border border-green-500/20 text-xs font-medium">
                ✓ Modelo cargado
              </span>
            }
          </div>
        </div>
      </header>

      <!-- Main Content -->
      <main class="flex-1 max-w-4xl mx-auto w-full p-4 md:p-6 flex flex-col">
        <!-- Loading State -->
        @if (state().isLoading) {
          <div class="flex-1 flex items-center justify-center">
            <div class="text-center">
              <div class="w-10 h-10 border-4 border-indigo-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
              <p class="text-lg text-slate-200 mb-2">Cargando modelo...</p>
              <div class="w-64 bg-slate-800 rounded-full h-1.5 mx-auto overflow-hidden">
                <div class="bg-indigo-600 h-full rounded-full transition-all" [style.width.%]="state().progress"></div>
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
              <p class="text-slate-500 mb-5">{{ state().error }}</p>
              <button
                (click)="loadModel()"
                class="px-6 py-3 bg-indigo-600 text-slate-100 font-medium rounded-xl hover:bg-indigo-500 transition-all duration-200 shadow-lg shadow-indigo-500/20">
                Reintentar
              </button>
            </div>
          </div>
        }

        <!-- Not Loaded State -->
        @if (!state().isLoading && !state().isReady && !state().error) {
          <div class="flex-1 flex items-center justify-center">
            <div class="text-center max-w-md">
              <div class="text-6xl mb-4">🤖</div>
              <h2 class="text-xl font-bold text-slate-100 mb-2">GPT OxideLLM_5M</h2>
              <p class="text-slate-500 mb-6">
                Modelo de lenguaje entrenado con el texto de OxideLLM_5M de la Mancha. Se ejecuta completamente en tu navegador.
              </p>
              <button
                (click)="loadModel()"
                class="px-8 py-3.5 bg-indigo-600 text-slate-100 font-medium rounded-2xl hover:bg-indigo-500 transition-all duration-200 shadow-lg shadow-indigo-500/20">
                🚀 Cargar Modelo
              </button>
              <p class="text-[10px] text-slate-500 mt-3">100% en tu navegador • Sin envío de datos a servidores</p>
            </div>
          </div>
        }

        <!-- Chat Interface -->
        @if (state().isReady) {
          <div class="flex-1 flex flex-col">
            <!-- Generated Text Display -->
            <div class="flex-1 bg-slate-900 rounded-3xl p-5 mb-5 overflow-y-auto border border-slate-800 min-h-[300px]">
              @if (generatedText()) {
                <pre class="whitespace-pre-wrap font-serif text-sm leading-relaxed text-slate-200">{{ generatedText() }}</pre>
              } @else {
                <p class="text-slate-500 text-sm italic">Escribe un texto inicial y el modelo continuará generando...</p>
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
                  placeholder="Escribe el inicio del texto... (ej: En un lugar de la Mancha)"
                  rows="3"
                  class="flex-1 px-4 py-3.5 bg-slate-800 border border-slate-600 rounded-2xl resize-none focus:outline-none focus:border-indigo-500 transition-colors text-sm"
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
                    min="50"
                    max="500"
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
                    class="w-24 accent-indigo-500"
                  />
                  <span class="text-slate-300">{{ temperature() }}</span>
                </label>
              </div>

              <!-- Buttons -->
              <div class="flex gap-3">
                <button
                  (click)="generate()"
                  [disabled]="!canGenerate()"
                  class="flex-1 px-6 py-3.5 bg-indigo-600 text-slate-100 font-medium rounded-2xl hover:bg-indigo-500 disabled:bg-slate-800 disabled:text-slate-500 disabled:cursor-not-allowed transition-all duration-200 shadow-lg shadow-indigo-500/20">
                  @if (isGenerating()) {
                    <span class="flex items-center justify-center gap-2">
                      <span class="animate-spin">⏳</span> Generando...
                    </span>
                  } @else {
                    ✨ Generar Texto
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
        Modelo ejecutándose 100% en tu navegador con TensorFlow.js • Sin envío de datos a servidores
      </footer>
    </div>
  `,
})
export class TFJSChatComponent {
  private readonly tfjsService = inject(TFJSModelService);
  private readonly router = inject(Router);

  readonly state = this.tfjsService.state;
  readonly generatedText = this.tfjsService.generatedText;

  readonly prompt = signal('En un lugar de la Mancha, de cuyo nombre ');
  readonly maxLength = signal(200);
  readonly temperature = signal(0.7);
  readonly isGenerating = signal(false);

  readonly canGenerate = computed(
    () => this.state().isReady && !this.isGenerating() && this.prompt().trim().length > 0
  );

  async loadModel(): Promise<void> {
    await this.tfjsService.loadModel();
  }

  async generate(): Promise<void> {
    if (!this.canGenerate()) return;

    this.isGenerating.set(true);
    try {
      await this.tfjsService.generate(
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
