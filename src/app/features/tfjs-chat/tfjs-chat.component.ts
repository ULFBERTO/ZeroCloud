import { DecimalPipe } from '@angular/common';
import { ChangeDetectionStrategy, Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Router } from '@angular/router';
import { TFJSModelService } from '../../core/services/tfjs-model.service';
import { BadgeComponent, IconComponent } from '../../shared/components';

@Component({
  selector: 'app-tfjs-chat',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [FormsModule, DecimalPipe, BadgeComponent, IconComponent],
  template: `
    <div class="flex min-h-[calc(100dvh-3.5rem)] flex-col bg-paper lg:min-h-dvh">
      <header class="flex h-14 shrink-0 items-center justify-between gap-3 border-b border-line bg-surface px-3 sm:px-5">
        <div class="flex min-w-0 items-center gap-2">
          <button
            type="button"
            class="rounded-[2px] p-1.5 text-muted transition-colors hover:bg-ink/5 hover:text-ink"
            aria-label="Volver al inicio"
            (click)="goBack()"
          >
            <app-icon name="arrow-left" [size]="16" />
          </button>
          <span class="hidden h-4 w-px bg-line sm:block"></span>
          <div class="flex items-center gap-2 font-mono text-[9px] uppercase tracking-[0.14em]">
            <span class="text-ink">Oxide TFJS</span>
            <span class="text-faint">/</span>
            <span class="hidden text-muted sm:inline">TensorFlow.js</span>
          </div>
        </div>

        <div class="flex items-center gap-2">
          @if (state().isReady) {
            <app-badge label="Listo" variant="success" [dot]="true" />
          }
          <app-badge label="TFJS" variant="neutral" />
        </div>
      </header>

      <main class="mx-auto flex w-full max-w-4xl flex-1 flex-col px-5 py-8 sm:px-8">
        @if (state().isLoading) {
          <div class="flex flex-1 flex-col justify-center py-12">
            <div class="flex items-end justify-between gap-6">
              <div>
                <p class="label-mono">Model / Loading</p>
                <h1 class="mt-2 text-3xl font-semibold tracking-[-0.04em] text-ink">
                  Cargando TFJS
                </h1>
              </div>
              <span class="font-mono text-5xl leading-none tracking-[-0.06em] text-ink">
                {{ state().progress | number: '1.0-0' }}%
              </span>
            </div>

            <div class="mt-6 h-2 w-full bg-line">
              <div
                class="h-full bg-ink transition-[width] duration-300"
                [style.width.%]="state().progress"
              ></div>
            </div>
            <p class="mt-3 text-[11px] leading-relaxed text-muted">
              Los pesos se descargan desde HuggingFace y se ejecutan con TensorFlow.js.
            </p>
          </div>
        }

        @if (state().error) {
          <div class="flex flex-1 flex-col justify-center py-8">
            <p class="label-mono">Runtime / Error</p>
            <h1 class="editorial-title mt-3 text-[clamp(2.25rem,5vw,3.5rem)] text-negative">
              El modelo no cargó.
            </h1>
            <p class="mt-3 max-w-xl text-[13px] leading-relaxed text-muted">{{ state().error }}</p>

            <div class="mt-6 flex flex-wrap gap-2">
              <button
                type="button"
                class="inline-flex h-9 items-center gap-2 bg-ink px-4 font-mono text-[9px] uppercase tracking-[0.12em] text-paper hover:bg-graphite"
                (click)="loadModel()"
              >
                <app-icon name="refresh" [size]="13" />
                Reintentar
              </button>
              <a
                href="https://huggingface.co/ULFBERTO/OxideLLM_5M-tfjs"
                target="_blank"
                rel="noopener noreferrer"
                class="inline-flex h-9 items-center gap-2 border border-line bg-surface px-4 font-mono text-[9px] uppercase tracking-[0.12em] text-graphite hover:border-ink hover:text-ink"
              >
                Abrir HuggingFace
                <app-icon name="arrow-up-right" [size]="12" />
              </a>
            </div>
          </div>
        }

        @if (!state().isLoading && !state().isReady && !state().error) {
          <div class="flex flex-1 flex-col justify-center py-8">
            <div class="flex items-center gap-2 font-mono text-[9px] uppercase tracking-[0.16em] text-muted">
              <span class="text-ink">+</span>
              Lab / Oxide TFJS
            </div>

            <h1 class="editorial-title mt-5 max-w-[15ch] text-[clamp(2.75rem,6vw,4.5rem)] text-ink">
              Un GPT de 5M en tu pestaña.
            </h1>
            <p class="mt-4 max-w-xl text-sm leading-relaxed text-muted">
              Entrenado con un corpus de literatura española. Los pesos se descargan una sola vez y
              la generación ocurre por completo en el navegador.
            </p>

            <div class="mt-8 grid grid-cols-3 border border-line bg-surface">
              <div class="border-r border-line p-4">
                <p class="label-mono">Parámetros</p>
                <p class="mt-2 text-[13px] font-medium text-ink">~5M</p>
              </div>
              <div class="border-r border-line p-4">
                <p class="label-mono">Runtime</p>
                <p class="mt-2 text-[13px] font-medium text-ink">TensorFlow.js</p>
              </div>
              <div class="p-4">
                <p class="label-mono">Corpus</p>
                <p class="mt-2 text-[13px] font-medium text-ink">Español</p>
              </div>
            </div>

            <div class="mt-8 flex flex-wrap items-center gap-3">
              <button
                type="button"
                class="inline-flex h-10 items-center gap-2 bg-ink px-5 font-mono text-[9px] uppercase tracking-[0.12em] text-paper transition-colors hover:bg-graphite"
                (click)="loadModel()"
              >
                <app-icon name="play" [size]="13" />
                Cargar modelo
              </button>
              <span class="font-mono text-[8px] uppercase tracking-[0.1em] text-faint">
                Sin envío de datos · caché del navegador
              </span>
            </div>
          </div>
        }

        @if (state().isReady) {
          <div class="flex flex-1 flex-col">
            <div class="mb-4 flex flex-wrap items-center gap-x-5 gap-y-1 border border-line bg-surface px-4 py-3 font-mono text-[8px] uppercase tracking-[0.08em]">
              <span class="text-faint">MODELO <span class="text-graphite">OxideLLM_5M</span></span>
              <span class="text-faint">PARAMS <span class="text-graphite">~5M</span></span>
              <span class="text-faint">VOCAB <span class="text-graphite">char-level</span></span>
            </div>

            <div class="min-h-[300px] flex-1 overflow-y-auto border border-line bg-raised p-5">
              @if (generatedText()) {
                <pre class="whitespace-pre-wrap font-sans text-[13px] leading-relaxed text-graphite">{{ generatedText() }}</pre>
              } @else {
                <div class="flex h-full min-h-[260px] flex-col items-center justify-center text-center">
                  <div class="relative flex h-14 w-14 items-center justify-center">
                    <div class="iridescent blob absolute inset-0 border border-ink/50"></div>
                    <span class="relative flex h-6 w-6 items-center justify-center bg-ink font-mono text-[9px] text-paper">O</span>
                  </div>
                  <p class="mt-4 text-[13px] font-medium text-ink">Escribe un inicio de texto</p>
                  <p class="mt-1 max-w-xs text-[11px] leading-relaxed text-muted">
                    El modelo continuará el texto carácter a carácter, de forma local.
                  </p>
                </div>
              }
            </div>

            <div class="mt-4 border border-line bg-surface p-3">
              <textarea
                [ngModel]="prompt()"
                (ngModelChange)="prompt.set($event)"
                (keydown)="onKeyDown($event)"
                placeholder="En un lugar de la Mancha, de cuyo nombre…"
                rows="3"
                class="command-input w-full resize-none border-0 bg-transparent px-1 py-1 text-[13px] focus:outline-none"
                [disabled]="isGenerating()"
                aria-label="Texto inicial"
              ></textarea>

              <div class="mt-2 flex flex-wrap items-end justify-between gap-3 border-t border-line pt-3">
                <div class="flex flex-wrap items-center gap-4">
                  <label class="flex items-center gap-2">
                    <span class="font-mono text-[8px] uppercase tracking-[0.1em] text-faint">Longitud</span>
                    <input
                      type="number"
                      [ngModel]="maxLength()"
                      (ngModelChange)="maxLength.set($event)"
                      min="50"
                      max="500"
                      class="command-input h-8 w-20 px-2 font-mono text-[11px]"
                    />
                  </label>
                  <label class="flex items-center gap-2">
                    <span class="font-mono text-[8px] uppercase tracking-[0.1em] text-faint">Temp</span>
                    <input
                      type="range"
                      [ngModel]="temperature()"
                      (ngModelChange)="temperature.set($event)"
                      min="0.1"
                      max="1.5"
                      step="0.1"
                      class="w-24 accent-ink"
                    />
                    <span class="font-mono text-[10px] text-graphite">{{ temperature() }}</span>
                  </label>
                </div>

                <div class="flex items-center gap-1.5">
                  <button
                    type="button"
                    class="inline-flex h-8 items-center gap-1.5 px-2.5 font-mono text-[9px] uppercase tracking-[0.1em] text-muted transition-colors hover:bg-ink/5 hover:text-ink"
                    (click)="clear()"
                  >
                    <app-icon name="trash" [size]="12" />
                    Limpiar
                  </button>
                  <button
                    type="button"
                    class="inline-flex h-8 items-center gap-1.5 bg-ink px-3 font-mono text-[9px] uppercase tracking-[0.1em] text-paper transition-colors hover:bg-graphite disabled:cursor-not-allowed disabled:opacity-30"
                    [disabled]="!canGenerate()"
                    (click)="generate()"
                  >
                    <app-icon name="sparkles" [size]="12" />
                    {{ isGenerating() ? 'Generando…' : 'Generar' }}
                  </button>
                </div>
              </div>
            </div>

            <p class="mt-2 text-center font-mono text-[8px] uppercase tracking-[0.1em] text-faint">
              Ctrl + Enter para generar · 100% local
            </p>
          </div>
        }
      </main>
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
    () => this.state().isReady && !this.isGenerating() && this.prompt().trim().length > 0,
  );

  async loadModel(): Promise<void> {
    await this.tfjsService.loadModel();
  }

  async generate(): Promise<void> {
    if (!this.canGenerate()) return;

    this.isGenerating.set(true);
    try {
      await this.tfjsService.generate(this.prompt(), this.maxLength(), this.temperature());
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
      void this.generate();
    }
  }

  goBack(): void {
    this.router.navigate(['/']);
  }
}
