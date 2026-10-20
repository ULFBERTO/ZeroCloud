import {
  ChangeDetectionStrategy,
  Component,
  DestroyRef,
  effect,
  HostListener,
  inject,
  input,
  output,
} from '@angular/core';
import { IconComponent } from './icon.component';

@Component({
  selector: 'app-sheet',
  standalone: true,
  changeDetection: ChangeDetectionStrategy.OnPush,
  imports: [IconComponent],
  template: `
    @if (open()) {
      <div class="fixed inset-0 z-50 flex justify-end" role="presentation">
        <button
          type="button"
          class="animate-fade absolute inset-0 cursor-default bg-ink/35 backdrop-blur-[2px]"
          aria-label="Cerrar panel"
          (click)="close.emit()"
        ></button>

        <section
          class="animate-slide-in relative flex h-full w-full flex-col border-l border-line bg-surface shadow-float"
          [class]="widthClass()"
          role="dialog"
          aria-modal="true"
          [attr.aria-label]="title()"
        >
          <header class="flex items-start justify-between gap-6 border-b border-line px-6 py-5">
            <div>
              @if (eyebrow()) {
                <p class="label-mono mb-1.5">{{ eyebrow() }}</p>
              }
              <h2 class="text-xl font-semibold tracking-[-0.02em] text-ink">{{ title() }}</h2>
            </div>
            <button
              type="button"
              class="rounded-[2px] p-2 text-muted transition-colors hover:bg-ink/5 hover:text-ink"
              aria-label="Cerrar"
              autofocus
              (click)="close.emit()"
            >
              <app-icon name="close" [size]="18" />
            </button>
          </header>

          <div class="flex-1 overflow-y-auto px-6 py-6">
            <ng-content />
          </div>

          <ng-content select="[footer]" />
        </section>
      </div>
    }
  `,
})
export class SheetComponent {
  readonly open = input<boolean>(false);
  readonly title = input<string>('');
  readonly eyebrow = input<string>('');
  readonly size = input<'md' | 'lg' | 'xl'>('md');
  readonly close = output<void>();

  private readonly destroyRef = inject(DestroyRef);

  widthClass(): string {
    const widths = {
      md: 'max-w-[440px]',
      lg: 'max-w-[600px]',
      xl: 'max-w-[760px]',
    };
    return widths[this.size()];
  }

  constructor() {
    effect((onCleanup) => {
      if (!this.open()) return;
      const previous = document.body.style.overflow;
      document.body.style.overflow = 'hidden';
      onCleanup(() => {
        document.body.style.overflow = previous;
      });
    });

    this.destroyRef.onDestroy(() => {
      document.body.style.overflow = '';
    });
  }

  @HostListener('document:keydown.escape')
  onEscape(): void {
    if (this.open()) {
      this.close.emit();
    }
  }
}
