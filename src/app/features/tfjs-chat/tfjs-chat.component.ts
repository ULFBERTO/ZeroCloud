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
              <h1 class="text-xl font-bold">🧠 GPT Don Quijote</h1>
              <p class="text-sm text-gray-400">Modelo TFJS en navegador</p>
            </div>
          </div>
          
          <div class="flex items-center gap-2">
            @if (state().isReady) {
              <span class="px-3 py-1 bg-green-600/20 text-green-400 rounded-full text-sm">
                ✓ Modelo cargado
              </span>
            }
          </div>
        </div>
      </header>

      <!-- Main Content -->
      <main class="flex-1 max-w-4xl mx-auto w-full p-4 flex flex-col">
        <!-- Loading State -->
        @if (state().isLoading) {
          <div class="flex-1 flex items-center justify-center">
            <div class="text-center">
              <div class="animate-spin w-12 h-12 border-4 border-blue-500 border-t-transparent rounded-full mx-auto mb-4"></div>
              <p class="text-lg mb-2">Cargando modelo...</p>
              <div class="w-64 bg-gray-700 rounded-full h-2 mx-auto">
                <div
                  class="bg-blue-500 h-full rounded-full transition-all"
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
              <button
                (click)="loadModel()"
                class="px-6 py-3 bg-blue-600 hover:bg-blue-500 rounded-xl transition-colors">
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
              <h2 class="text-xl font-bold mb-2">GPT Don Quijote</h2>
              <p class="text-gray-400 mb-4">
                Modelo de lenguaje entrenado con el texto de Don Quijote de la Mancha.
                Se ejecuta completamente en tu navegador.
              </p>
              <button
                (click)="loadModel()"
                class="px-6 py-3 bg-blue-600 hover:bg-blue-500 rounded-xl transition-colors">
                🚀 Cargar Modelo
              </button>
            </div>
          </div>
        }

        <!-- Chat Interface -->
        @if (state().isReady) {
          <div class="flex-1 flex flex-col">
            <!-- Generated Text Display -->
            <div class="flex-1 bg-gray-800 rounded-xl p-4 mb-4 overflow-y-auto">
              @if (generatedText()) {
                <pre class="whitespace-pre-wrap font-serif text-lg leading-relaxed">{{ generatedText() }}</pre>
              } @else {
                <p class="text-gray-500 italic">
                  Escribe un texto inicial y el modelo continuará generando...
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
                  placeholder="Escribe el inicio del texto... (ej: En un lugar de la Mancha)"
                  rows="3"
                  class="flex-1 px-4 py-3 bg-gray-800 border border-gray-700 rounded-xl resize-none focus:outline-none focus:border-blue-500"
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
                    min="50"
                    max="500"
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
                  class="flex-1 px-6 py-3 bg-blue-600 hover:bg-blue-500 rounded-xl font-medium transition-colors disabled:opacity-50 disabled:cursor-not-allowed">
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
